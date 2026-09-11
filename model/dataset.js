/**
 * Loads every *.jsonl file inside dataDir. Each line must be a JSON object:
 *   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
 *
 * Multiple files are simply concatenated, so you can split data across
 * as many *.jsonl files as you like (e.g. topic1.jsonl, topic2.jsonl, ...).
 */

const fs = require('fs');
const path = require('path');

function listJsonlFiles(dataDir) {
  return fs
    .readdirSync(dataDir)
    .filter((f) => f.endsWith('.jsonl'))
    .map((f) => path.join(dataDir, f));
}

/** Turn one conversation's messages into a single training string. */
function messagesToText(messages) {
  let out = '';
  for (const { role, content } of messages) {
    if (role === 'user') out += `<|user|>${content}<|end|>`;
    else if (role === 'assistant') out += `<|assistant|>${content}<|end|>`;
    // system role (if present) is prepended silently as plain context
    else out += content;
  }
  return out;
}

/** Read every conversation from every jsonl file in dataDir as raw text blocks. */
function loadConversations(dataDir) {
  const files = listJsonlFiles(dataDir);
  if (files.length === 0) {
    throw new Error(`No .jsonl files found in ${dataDir}`);
  }

  const texts = [];
  for (const file of files) {
    const lines = fs.readFileSync(file, 'utf-8').split('\n').filter((l) => l.trim());
    for (const line of lines) {
      let obj;
      try {
        obj = JSON.parse(line);
      } catch (e) {
        console.warn(`Skipping malformed line in ${file}: ${e.message}`);
        continue;
      }
      if (!obj.messages || !Array.isArray(obj.messages)) {
        console.warn(`Skipping line without "messages" array in ${file}`);
        continue;
      }
      texts.push(messagesToText(obj.messages));
    }
  }
  return texts;
}

/**
 * Tokenize all conversations and slice into fixed-length (contextLength + 1)
 * windows for next-token-prediction training. Returns { inputs, targets }
 * as plain arrays of arrays (caller converts to tensors in batches).
 */
/**
 * Tokenize all conversations and slice into fixed-length (contextLength + 1)
 * windows for next-token-prediction training. Returns { inputs, targets }
 * as plain arrays of arrays (caller converts to tensors in batches).
 *
 * BUG FIX: this used to concatenate every conversation into ONE long token
 * stream and then slide a fixed window across the whole thing. That meant a
 * window could start near the end of conversation A's answer, cross the
 * <eos><bos> boundary, and end inside conversation B's question — the model
 * would then be trained to predict tokens of an unrelated conversation from
 * a different one's context. Windows are now built PER conversation, so a
 * window never spans two unrelated conversations. Short conversations that
 * don't fill a whole window are still kept (right-padded conceptually by
 * simply using whatever length they have — see contextLength note below).
 */
function buildTrainingWindows(conversations, tokenizer, contextLength) {
  const bosId = tokenizer.tokenId('<bos>');
  const eosId = tokenizer.tokenId('<eos>');
  const padId = tokenizer.tokenId('<pad>') ?? eosId;

  const windows = [];
  const step = Math.max(1, Math.floor(contextLength / 2)); // 50% overlap

  for (const text of conversations) {
    const stream = [bosId, ...tokenizer.encode(text), eosId];

    if (stream.length <= contextLength) {
      // Too short for a full window on its own — pad so the conversation is
      // still seen in full at least once, instead of being silently dropped.
      const input = stream.slice(0, stream.length - 1);
      const target = stream.slice(1);
      while (input.length < contextLength) input.push(padId);
      while (target.length < contextLength) target.push(padId);
      windows.push({ input: input.slice(0, contextLength), target: target.slice(0, contextLength), lossMask: Array.from({ length: contextLength }, (_, i) => i < stream.length - 1 ? 1 : 0) });
      continue;
    }

    for (let i = 0; i + contextLength < stream.length; i += step) {
      const chunk = stream.slice(i, i + contextLength + 1);
      windows.push({
        input: chunk.slice(0, contextLength),
        target: chunk.slice(1, contextLength + 1),
        lossMask: new Array(contextLength).fill(1),
      });
    }
  }
  return windows;
}

/**
 * Splits conversations into train/validation sets BEFORE windowing, so
 * validation windows never come from a conversation the model trained on.
 * Shuffles with a fixed-ish approach (caller passes an already-loaded
 * array; we shuffle a copy) and takes the last `valFraction` as validation.
 */
function splitConversations(conversations, valFraction = 0.05) {
  const copy = conversations.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  const valCount = Math.max(1, Math.floor(copy.length * valFraction));
  return {
    train: copy.slice(0, copy.length - valCount),
    val: copy.slice(copy.length - valCount),
  };
}

module.exports = { listJsonlFiles, loadConversations, buildTrainingWindows, splitConversations, messagesToText };
