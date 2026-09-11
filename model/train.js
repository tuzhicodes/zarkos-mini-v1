const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const config = require('./config');
const { BPETokenizer } = require('./tokenizer');
const { loadConversations, buildTrainingWindows, splitConversations } = require('./dataset');
const { TinyGPT } = require('./transformer');
const { saveCheckpoint, checkpointExists, loadCheckpointInto } = require('./checkpoint');

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
    const logits = model.forward(inputIds); // [B,S,V]
    const oneHot = tf.oneHot(targetIds, vocabSize); // [B,S,V]
    const logProbs = tf.logSoftmax(logits, -1);
    const perToken = oneHot.mul(logProbs).sum(-1).neg(); // [B,S]
    if (!lossMask) return perToken.mean();
    const masked = perToken.mul(lossMask);
    const denom = lossMask.sum().maximum(1);
    return masked.sum().div(denom);
  });
}

/**
 * Forward-only loss over a set of windows, no gradient step. Used to report
 * validation loss so training loss numbers can't be mistaken for a measure
 * of how the model does on conversations it hasn't trained on.
 */
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

async function main() {
  if (!fs.existsSync(config.checkpointDir)) fs.mkdirSync(config.checkpointDir, { recursive: true });

  console.log('Loading conversations from', config.dataDir, '...');
  console.log(`Config: batchSize=${config.batchSize}, contextLength=${config.contextLength}, nEmbd=${config.nEmbd}, nLayer=${config.nLayer}, vocabSize=${config.vocabSize}`);
  const allConversations = loadConversations(config.dataDir);
  console.log(`Loaded ${allConversations.length} conversations.`);

  // BUG FIX: previously every conversation went straight into training —
  // there was no held-out data, so a falling "loss" number could only ever
  // tell you the model was memorizing what it had already seen, never
  // whether it generalizes. A small validation split (default 5%) is set
  // aside BEFORE windowing and never trained on; its loss is reported
  // separately each epoch.
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
  }
  // Keep model vocab size in sync with the actual tokenizer that was built/loaded
  config.vocabSize = tokenizer.vocabSize;

  console.log('Building training windows...');
  const windows = buildTrainingWindows(conversations, tokenizer, config.contextLength);
  console.log(`Built ${windows.length} training windows of length ${config.contextLength}.`);
  if (windows.length === 0) {
    throw new Error('No training windows produced — is your dataset too small for contextLength?');
  }
  const valWindows = buildTrainingWindows(valConversations, tokenizer, config.contextLength);
  console.log(`Built ${valWindows.length} validation windows (held out from training).`);

  const model = new TinyGPT(config);
  console.log(`Model initialized: ~${(model.countParams() / 1e6).toFixed(2)}M parameters.`);

  if (checkpointExists(config.checkpointDir)) {
    console.log('Resuming from existing checkpoint...');
    loadCheckpointInto(model, config.checkpointDir);
  }

  const optimizer = tf.train.adam(config.learningRate);

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
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

      inputIds.dispose();
      targetIds.dispose();
      lossMask.dispose();
      lossValue.dispose();

      if (steps % 20 === 0) {
        console.log(`  epoch ${epoch} step ${steps}/${Math.ceil(windows.length / config.batchSize)} - loss ${(epochLoss / steps).toFixed(4)}`);
      }

      // Mid-epoch checkpoint so long epochs don't lose all progress on a
      // crash/restart. Does not touch the end-of-epoch save below.
      if (config.checkpointEvery > 0 && steps % config.checkpointEvery === 0) {
        await saveCheckpoint(model, config.checkpointDir);
      }
    }

    console.log(`Epoch ${epoch}/${config.epochs} done — avg train loss ${(epochLoss / steps).toFixed(4)}`);
    const valLossValue = await evalLoss(model, valWindows, config.batchSize, config.vocabSize);
    if (valLossValue !== null) {
      console.log(`Epoch ${epoch}/${config.epochs} — validation loss ${valLossValue.toFixed(4)} (on held-out data, never trained on)`);
    }
    await saveCheckpoint(model, config.checkpointDir);
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
