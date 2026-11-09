const tf = require('./tf');

function sampleNextToken(logits1D, temperature = 0.8, topK = 40) {
  return tf.tidy(() => {
    const scaled = logits1D.div(Math.max(temperature, 1e-6));
    const k = Math.min(topK, scaled.shape[0]);
    const { values, indices } = tf.topk(scaled, k);
    const probs = tf.softmax(values);
    const probsArr = probs.arraySync();
    const idxArr = indices.arraySync();

    let r = Math.random();
    let chosen = idxArr[idxArr.length - 1];
    for (let i = 0; i < probsArr.length; i++) {
      r -= probsArr[i];
      if (r <= 0) {
        chosen = idxArr[i];
        break;
      }
    }
    return chosen;
  });
}

function applyRepetitionPenalty(logits1D, previousIds, penalty = 1.3) {
  if (penalty === 1 || previousIds.length === 0) return logits1D;
  const arr = logits1D.arraySync();
  for (const id of new Set(previousIds)) {
    if (id < 0 || id >= arr.length) continue;
    arr[id] = arr[id] > 0 ? arr[id] / penalty : arr[id] * penalty;
  }
  return tf.tensor1d(arr);
}

function applyFrequencyPresencePenalty(logits1D, previousIds, freqPenalty = 0, presPenalty = 0) {
  if ((freqPenalty === 0 && presPenalty === 0) || previousIds.length === 0) return logits1D;
  const arr = logits1D.arraySync();
  const counts = new Map();
  for (const id of previousIds) counts.set(id, (counts.get(id) || 0) + 1);
  for (const [id, count] of counts) {
    if (id < 0 || id >= arr.length) continue;
    arr[id] -= freqPenalty * count;
    if (count > 0) arr[id] -= presPenalty;
  }
  return tf.tensor1d(arr);
}

function blockedNgramTokens(previousIds, n = 3) {
  if (n <= 0 || previousIds.length < n) return null;
  const blocked = new Set();

  const gramContinuations = new Map();
  for (let i = 0; i + n <= previousIds.length; i++) {

    const prefix = previousIds.slice(i, i + n - 1).join('\u0001');
    const cont = previousIds[i + n - 1];
    if (!gramContinuations.has(prefix)) gramContinuations.set(prefix, new Set());
    gramContinuations.get(prefix).add(cont);
  }

  const prefix = previousIds.slice(-(n - 1)).join('\u0001');
  const conts = gramContinuations.get(prefix);
  if (conts) for (const c of conts) blocked.add(c);

  return blocked.size ? blocked : null;
}

function blockTokens(logits1D, blockedSet) {
  if (!blockedSet || blockedSet.size === 0) return logits1D;
  const arr = logits1D.arraySync();
  for (const id of blockedSet) {
    if (id >= 0 && id < arr.length) arr[id] = -Infinity;
  }
  return tf.tensor1d(arr);
}

const yieldToEventLoop = () => new Promise((resolve) => setImmediate(resolve));

async function generateReply(model, tokenizer, config, userMessage, opts = {}) {
  const {
    maxNewTokens = 100,
    temperature = 0.8,
    topK = 40,
    history = [],
    repetitionPenalty = 1.3,
    frequencyPenalty = 0.5,
    presencePenalty = 0.3,
    noRepeatNgramSize = 3,
    onToken = null,
  } = opts;
  const endId = tokenizer.tokenId('<|end|>');
  const bosId = tokenizer.tokenId('<bos>');
  const assistantId = tokenizer.tokenId('<|assistant|>');

  let promptText = '';
  for (const turn of history) {
    if (turn.role === 'user') promptText += `<|user|>${turn.content}<|end|>`;
    else if (turn.role === 'assistant') promptText += `<|assistant|>${turn.content}<|end|>`;
  }
  promptText += `<|user|>${userMessage}<|end|><|assistant|>`;

  let ids = [bosId, ...tokenizer.encode(promptText)];

  const genStart = ids.length;

  for (let step = 0; step < maxNewTokens; step++) {
    await yieldToEventLoop();

    const context = ids.slice(-config.contextLength);
    const seqLen = context.length;

    const inputIds = tf.tensor2d([context], [1, seqLen], 'int32');
    const logits = model.forward(inputIds);
    const lastLogits = logits.slice([0, seqLen - 1, 0], [1, 1, config.vocabSize]).reshape([config.vocabSize]);

    const genIds = ids.slice(genStart);

    let work = lastLogits;
    let allocated = [];

    if (noRepeatNgramSize > 0) {
      const blocked = blockedNgramTokens(genIds, noRepeatNgramSize);
      if (blocked) {
        const blockedT = blockTokens(work, blocked);
        if (work !== lastLogits) allocated.push(work);
        work = blockedT;
      }
    }

    const repT = applyRepetitionPenalty(work, context, repetitionPenalty);
    if (repT !== work) { allocated.push(work); work = repT; }

    const fpT = applyFrequencyPresencePenalty(work, genIds, frequencyPenalty, presencePenalty);
    if (fpT !== work) { allocated.push(work); work = fpT; }

    const nextId = sampleNextToken(work, temperature, topK);

    inputIds.dispose();
    logits.dispose();
    lastLogits.dispose();
    for (const t of allocated) t.dispose();
    if (work !== lastLogits && !allocated.includes(work)) work.dispose();

    if (nextId === endId) break;
    ids.push(nextId);
    if (typeof onToken === 'function') {
      try {
        const piece = typeof tokenizer.decodeToken === 'function'
          ? tokenizer.decodeToken(nextId)
          : tokenizer.decode([nextId]);
        onToken(piece);
      } catch (_) {}
    }
  }

  const generatedIds = ids.slice(genStart);
  return tokenizer.decode(generatedIds);
}

module.exports = {
  generateReply,
  sampleNextToken,
  applyRepetitionPenalty,
  applyFrequencyPresencePenalty,
  blockedNgramTokens,
  blockTokens,
};
