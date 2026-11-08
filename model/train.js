const tf = require('./tf');
const fs = require('fs');
const path = require('path');

const config = require('./config');
const { BPETokenizer } = require('./tokenizer');
const { loadConversations, buildTrainingWindows, splitConversations } = require('./dataset');
const { ZarKosLM } = require('./transformer');
const { saveCheckpoint, checkpointExists, loadCheckpointInto, saveTrainingMeta, loadTrainingMeta } = require('./checkpoint');

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function* batches(windows, batchSize) {
  const shuffled = shuffle(windows.slice());
  for (let i = 0; i < shuffled.length; i += batchSize) {
    yield shuffled.slice(i, i + batchSize);
  }
}

function trainLoss(model, inputIds, targetIds, vocabSize, lossMask = null) {
  return tf.tidy(() => {
    const logits = model.forward(inputIds);
    const oneHot = tf.oneHot(targetIds, vocabSize);
    const logProbs = tf.logSoftmax(logits, -1);
    const perToken = oneHot.mul(logProbs).sum(-1).neg();
    if (!lossMask) return perToken.mean();
    const masked = perToken.mul(lossMask);
    const denom = lossMask.sum().maximum(1);
    return masked.sum().div(denom);
  });
}

async function evalLoss(model, windows, batchSize, vocabSize) {
  if (windows.length === 0) return null;
  let total = 0;
  let count = 0;
  for (let i = 0; i < windows.length; i += batchSize) {
    const batch = windows.slice(i, i + batchSize);
    const inputIds = tf.tensor2d(batch.map((w) => w.input), [batch.length, batch[0].input.length], 'int32');
    const targetIds = tf.tensor2d(batch.map((w) => w.target), [batch.length, batch[0].target.length], 'int32');
    const maskIds = tf.tensor2d(batch.map((w) => w.lossMask || new Array(w.input.length).fill(1)), [batch.length, batch[0].input.length], 'float32');
    const loss = trainLoss(model, inputIds, targetIds, vocabSize, maskIds);
    total += await loss.data().then((d) => d[0]);
    count += 1;
    inputIds.dispose();
    targetIds.dispose();
    maskIds.dispose();
    loss.dispose();
  }
  return total / count;
}

function logWindowUtilization(windows, contextLength) {
  if (windows.length === 0) return;
  let realTokens = 0;
  let paddedWindows = 0;
  for (const w of windows) {
    const real = w.lossMask.reduce((a, b) => a + b, 0);
    realTokens += real;
    if (real < contextLength) paddedWindows++;
  }
  const totalSlots = windows.length * contextLength;
  const utilization = (100 * realTokens) / totalSlots;
  console.log(
    `Window stats: ${paddedWindows}/${windows.length} windows are padded (short conversations). ` +
    `Real-token utilization: ${utilization.toFixed(1)}%.`
  );
  if (utilization < 60) {
    console.log(
      `TIP: utilization is low — most conversations are shorter than CONTEXT_LENGTH=${contextLength}. ` +
      `Lowering CONTEXT_LENGTH in .env (then clearing checkpoint/ and retraining) can speed training up a lot without losing data.`
    );
  }
}

async function main() {
  if (!fs.existsSync(config.checkpointDir)) fs.mkdirSync(config.checkpointDir, { recursive: true });

  console.log('Loading conversations from', config.dataDir, '...');
  console.log(`Config: batchSize=${config.batchSize}, contextLength=${config.contextLength}, nEmbd=${config.nEmbd}, nLayer=${config.nLayer}, vocabSize=${config.vocabSize}`);
  const allConversations = loadConversations(config.dataDir);
  console.log(`Loaded ${allConversations.length} conversations.`);

  const { train: conversations, val: valConversations } = splitConversations(allConversations, config.valFraction);
  console.log(`Split: ${conversations.length} train conversations, ${valConversations.length} validation (held out, never trained on).`);

  let tokenizer;
  const tokenizerPath = path.join(config.checkpointDir, 'tokenizer.json');
  if (fs.existsSync(tokenizerPath)) {
    console.log('Found existing tokenizer, loading it...');
    tokenizer = BPETokenizer.load(config.checkpointDir);
  } else {
    console.log(`Training BPE tokenizer (target vocab size ${config.vocabSize})...`);
    tokenizer = new BPETokenizer(config.specialTokens).train(conversations, config.vocabSize);
    tokenizer.save(config.checkpointDir);
    console.log(`Tokenizer trained: ${tokenizer.vocabSize} tokens.`);
    if (tokenizer.vocabSize < config.vocabSize) {
      console.log(
        `Note: target VOCAB_SIZE was ${config.vocabSize}, but the current dataset only supports ` +
        `${tokenizer.vocabSize} tokens (not enough repeated pairs to merge further). This is not an ` +
        `error — training will use ${tokenizer.vocabSize} — but more/varied data would let it grow further.`
      );
    }
  }

  config.vocabSize = tokenizer.vocabSize;

  console.log('Building training windows...');
  const windows = buildTrainingWindows(conversations, tokenizer, config.contextLength);
  console.log(`Built ${windows.length} training windows of length ${config.contextLength}.`);
  if (windows.length === 0) {
    throw new Error('No training windows produced — is your dataset too small for contextLength?');
  }
  logWindowUtilization(windows, config.contextLength);
  const valWindows = buildTrainingWindows(valConversations, tokenizer, config.contextLength);
  console.log(`Built ${valWindows.length} validation windows (held out from training).`);

  const model = new ZarKosLM(config);
  console.log(`Model initialized: ~${(model.countParams() / 1e6).toFixed(2)}M parameters.`);

  const optimizer = tf.train.adam(config.learningRate);

  const bestCheckpointDir = config.checkpointDir + '-best';

  let completedEpochs = 0;
  let globalStep = 0;
  let bestValLoss = null;
  if (checkpointExists(config.checkpointDir)) {
    console.log('Resuming from existing checkpoint...');
    loadCheckpointInto(model, config.checkpointDir);
    const meta = await loadTrainingMeta(config.checkpointDir, optimizer);
    completedEpochs = meta.completedEpochs;
    globalStep = meta.globalStep;
    bestValLoss = meta.bestValLoss;
    console.log(`Checkpoint had ${completedEpochs} completed epoch(s), ${globalStep} total step(s) so far.`);
    if (bestValLoss !== null) {
      console.log(`Best validation loss so far: ${bestValLoss.toFixed(4)} (saved separately in ${bestCheckpointDir}).`);
    }
  }

  if (completedEpochs >= config.epochs) {
    console.log(
      `Checkpoint already has ${completedEpochs} completed epochs, which meets or exceeds ` +
      `EPOCHS=${config.epochs} in .env. Nothing to do — raise EPOCHS in .env if you want to train further.`
    );
    return;
  }

  for (let epoch = completedEpochs + 1; epoch <= config.epochs; epoch++) {
    let epochLoss = 0;
    let steps = 0;

    for (const batch of batches(windows, config.batchSize)) {
      const inputArr = batch.map((w) => w.input);
      const targetArr = batch.map((w) => w.target);

      const inputIds = tf.tensor2d(inputArr, [batch.length, config.contextLength], 'int32');
      const targetIds = tf.tensor2d(targetArr, [batch.length, config.contextLength], 'int32');
      const lossMask = tf.tensor2d(batch.map((w) => w.lossMask || new Array(config.contextLength).fill(1)), [batch.length, config.contextLength], 'float32');

      const lossValue = optimizer.minimize(
        () => trainLoss(model, inputIds, targetIds, config.vocabSize, lossMask),
        true
      );

      epochLoss += await lossValue.data().then((d) => d[0]);
      steps += 1;
      globalStep += 1;

      inputIds.dispose();
      targetIds.dispose();
      lossMask.dispose();
      lossValue.dispose();

      if (steps % 20 === 0) {
        console.log(`  epoch ${epoch} step ${steps}/${Math.ceil(windows.length / config.batchSize)} - loss ${(epochLoss / steps).toFixed(4)}`);
      }

      if (config.checkpointEvery > 0 && steps % config.checkpointEvery === 0) {
        await saveCheckpoint(model, config.checkpointDir);
        await saveTrainingMeta(config.checkpointDir, { completedEpochs: epoch - 1, globalStep, bestValLoss }, optimizer);
      }
    }

    console.log(`Epoch ${epoch}/${config.epochs} done — avg train loss ${(epochLoss / steps).toFixed(4)}`);
    const valLossValue = await evalLoss(model, valWindows, config.batchSize, config.vocabSize);
    if (valLossValue !== null) {
      console.log(`Epoch ${epoch}/${config.epochs} — validation loss ${valLossValue.toFixed(4)} (on held-out data, never trained on)`);
      if (bestValLoss === null || valLossValue < bestValLoss) {
        const improved = bestValLoss === null ? null : (bestValLoss - valLossValue).toFixed(4);
        bestValLoss = valLossValue;
        await saveCheckpoint(model, bestCheckpointDir);
        console.log(
          improved === null
            ? `New best validation loss (${bestValLoss.toFixed(4)}) — saved to ${bestCheckpointDir}.`
            : `Validation loss improved by ${improved} — new best (${bestValLoss.toFixed(4)}) saved to ${bestCheckpointDir}.`
        );
      } else {
        console.log(
          `Validation loss did not improve (best remains ${bestValLoss.toFixed(4)} in ${bestCheckpointDir}) — ` +
          `model may be starting to overfit.`
        );
      }
    }
    await saveCheckpoint(model, config.checkpointDir);
    await saveTrainingMeta(config.checkpointDir, { completedEpochs: epoch, globalStep, bestValLoss }, optimizer);
  }

  console.log('Training complete.');
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

module.exports = { main };
