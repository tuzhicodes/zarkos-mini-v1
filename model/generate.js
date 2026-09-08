const tf = require('@tensorflow/tfjs');

/** Sample the next token id from logits using temperature + top-k. */
function sampleNextToken(logits1D, temperature = 0.8, topK = 40) {
  return tf.tidy(() => {
    const scaled = logits1D.div(Math.max(temperature, 1e-6));
    const { values, indices } = tf.topk(scaled, Math.min(topK, scaled.shape[0]));
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
 * Generate a reply to a user message using the <|user|>...<|end|><|assistant|>
 * chat template the model was trained on.
 *
 * BUG FIX: this used to build a prompt from ONLY the current message, with
 * no way to include earlier turns — so a UI showing a running conversation
 * was cosmetic only; each reply was generated with zero knowledge of what
 * was said before it. `history` (an array of {role, content}, oldest
 * first) is now optional but supported: every turn gets encoded the same
 * way training data was, ending with the new user message + <|assistant|>.
 * Because contextLength is small, `ids.slice(-config.contextLength)` below
 * already keeps only what fits — passing more history than fits is safe,
 * it just naturally truncates to the most recent turns.
 */
function generateReply(model, tokenizer, config, userMessage, opts = {}) {
  const { maxNewTokens = 100, temperature = 0.8, topK = 40, history = [] } = opts;
  const endId = tokenizer.tokenId('<|end|>');
  const bosId = tokenizer.tokenId('<bos>');

  let promptText = '';
  for (const turn of history) {
    if (turn.role === 'user') promptText += `<|user|>${turn.content}<|end|>`;
    else if (turn.role === 'assistant') promptText += `<|assistant|>${turn.content}<|end|>`;
  }
  promptText += `<|user|>${userMessage}<|end|><|assistant|>`;

  let ids = [bosId, ...tokenizer.encode(promptText)];

  for (let step = 0; step < maxNewTokens; step++) {
    // No left-padding: training windows were always full, unpadded
    // contextLength chunks with positions 0..contextLength-1. Left-padding
    // here would push real tokens into position indices they never
    // occupied during training, feeding the model out-of-distribution
    // positional embeddings. forward() already handles a shorter seqLen
    // dynamically, so just use the real context length.
    const context = ids.slice(-config.contextLength);
    const seqLen = context.length;

    const inputIds = tf.tensor2d([context], [1, seqLen], 'int32');
    const logits = model.forward(inputIds); // [1, S, V]
    const lastLogits = logits.slice([0, seqLen - 1, 0], [1, 1, config.vocabSize]).reshape([config.vocabSize]);

    const nextId = sampleNextToken(lastLogits, temperature, topK);

    inputIds.dispose();
    logits.dispose();
    lastLogits.dispose();

    if (nextId === endId) break;
    ids.push(nextId);
  }

  // Only decode the tokens generated after the prompt's <|assistant|> marker
  const generatedIds = ids.slice(ids.indexOf(tokenizer.tokenId('<|assistant|>')) + 1);
  return tokenizer.decode(generatedIds);
}

module.exports = { generateReply, sampleNextToken };
