const tf = require('@tensorflow/tfjs');

/**
 * Sample the next token id from logits using temperature + top-k.
 *
 * Returns { id } — just the sampled token id.
 */
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

/**
 * Penalize tokens that already appear in `previousIds` (CTRL-style
 * repetition penalty: divide positive logits / multiply negative ones by
 * `penalty`, so the token becomes less likely either way).
 */
function applyRepetitionPenalty(logits1D, previousIds, penalty = 1.3) {
  if (penalty === 1 || previousIds.length === 0) return logits1D;
  const arr = logits1D.arraySync();
  for (const id of new Set(previousIds)) {
    if (id < 0 || id >= arr.length) continue;
    arr[id] = arr[id] > 0 ? arr[id] / penalty : arr[id] * penalty;
  }
  return tf.tensor1d(arr);
}

/**
 * Frequency+presence penalty (OpenAI-style). For every token that has
 * appeared `freq` times in `previousIds`:
 *   - frequency penalty:  logit -= freqPenalty * count
 *   - presence penalty:   logit -= presPenalty   (once, if count > 0)
 *
 * Unlike repetition penalty (which is a multiplicative factor applied once
 * per unique token), frequency scales with HOW MANY times a token was
 * repeated — this is what actually breaks the "ka ka ka ka ka" loops that
 * an undertrained model falls into, because each repeat pushes the token
 * further down.
 */
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

/**
 * Block any token id that would complete an n-gram already seen in
 * `previousIds`. This is the single most effective technique against
 * "same phrase over and over" degenerate output in small/undertrained LLMs,
 * because it makes the exact repetition structurally impossible rather than
 * merely unlikely.
 *
 * Standard implementation (same as HuggingFace `no_repeat_ngram_size`):
 *   - Build the set of n-grams of size n in previousIds.
 *   - For each candidate next-token id, check if the last (n-1) tokens of
 *     previousIds + this candidate forms an already-seen n-gram.
 *   - If yes, set its logit to -Infinity so it can never be sampled.
 *
 * Returns a Set of blocked token ids (caller applies -inf), or null if
 * no blocking applies (too few tokens / n <= 0).
 */
function blockedNgramTokens(previousIds, n = 3) {
  if (n <= 0 || previousIds.length < n) return null;
  const blocked = new Set();

  // Build a map of every seen n-gram's *continuation* token.
  // key = last (n-1) tokens joined, value = Set of token ids that followed.
  const gramContinuations = new Map();
  for (let i = 0; i + n <= previousIds.length; i++) {
    // n-gram = previousIds[i .. i+n-1]; continuation = previousIds[i+n-1]
    // For blocking, we care about: given last (n-1) tokens, which (n)th
    // tokens have we already seen?
    const prefix = previousIds.slice(i, i + n - 1).join('\u0001');
    const cont = previousIds[i + n - 1];
    if (!gramContinuations.has(prefix)) gramContinuations.set(prefix, new Set());
    gramContinuations.get(prefix).add(cont);
  }

  // The candidate next token would form an n-gram with the last (n-1) tokens.
  const prefix = previousIds.slice(-(n - 1)).join('\u0001');
  const conts = gramContinuations.get(prefix);
  if (conts) for (const c of conts) blocked.add(c);

  return blocked.size ? blocked : null;
}

/** Apply a Set of blocked token ids to a logits tensor by setting -Infinity. */
function blockTokens(logits1D, blockedSet) {
  if (!blockedSet || blockedSet.size === 0) return logits1D;
  const arr = logits1D.arraySync();
  for (const id of blockedSet) {
    if (id >= 0 && id < arr.length) arr[id] = -Infinity;
  }
  return tf.tensor1d(arr);
}

/**
 * Generate a reply to a user message using the <|user|>...<|end|><|assistant|>
 * chat template the model was trained on.
 *
 * Anti-repetition layering (applied in order before sampling):
 *   1. no-repeat-ngram blocking (noRepeatNgramSize, default 3) — makes
 *      exact phrase loops impossible.
 *   2. repetition penalty (CTRL-style multiplicative).
 *   3. frequency + presence penalties (additive, scales with repeat count).
 *
 * All three are ON by default with conservative values. For a better-trained
 * model you can lower/disable them; for an undertrained model these are what
 * keep output from collapsing into "ka ka ka".
 */
function generateReply(model, tokenizer, config, userMessage, opts = {}) {
  const {
    maxNewTokens = 100,
    temperature = 0.8,
    topK = 40,
    history = [],
    repetitionPenalty = 1.3,
    frequencyPenalty = 0.5,
    presencePenalty = 0.3,
    noRepeatNgramSize = 3,
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

  // Track only the GENERATED ids (after <|assistant|>) for n-gram blocking,
  // so we don't accidentally block tokens that legitimately appear in the
  // user's prompt / chat history.
  const genStart = ids.length;

  for (let step = 0; step < maxNewTokens; step++) {
    const context = ids.slice(-config.contextLength);
    const seqLen = context.length;

    const inputIds = tf.tensor2d([context], [1, seqLen], 'int32');
    const logits = model.forward(inputIds); // [1, S, V]
    const lastLogits = logits.slice([0, seqLen - 1, 0], [1, 1, config.vocabSize]).reshape([config.vocabSize]);

    // generated-so-far ids (for penalties + n-gram), bounded to context window
    const genIds = ids.slice(genStart);

    let work = lastLogits;
    let allocated = [];

    // 1) no-repeat-ngram blocking on generated ids
    if (noRepeatNgramSize > 0) {
      const blocked = blockedNgramTokens(genIds, noRepeatNgramSize);
      if (blocked) {
        const blockedT = blockTokens(work, blocked);
        if (work !== lastLogits) allocated.push(work);
        work = blockedT;
      }
    }

    // 2) repetition penalty (multiplicative) on full context
    const repT = applyRepetitionPenalty(work, context, repetitionPenalty);
    if (repT !== work) { allocated.push(work); work = repT; }

    // 3) frequency + presence penalties (additive, scales with repeat count)
    const fpT = applyFrequencyPresencePenalty(work, genIds, frequencyPenalty, presencePenalty);
    if (fpT !== work) { allocated.push(work); work = fpT; }

    const nextId = sampleNextToken(work, temperature, topK);

    // cleanup
    inputIds.dispose();
    logits.dispose();
    lastLogits.dispose();
    for (const t of allocated) t.dispose();
    if (work !== lastLogits && !allocated.includes(work)) work.dispose();

    if (nextId === endId) break;
    ids.push(nextId);
  }

  // Only decode the tokens generated after the prompt's <|assistant|> marker
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
