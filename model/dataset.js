const fs = require('fs');
const path = require('path');

function listJsonlFiles(dataDir) {

  return fs
    .readdirSync(dataDir)
    .filter((f) => f.endsWith('.jsonl'))
    .sort()
    .map((f) => path.join(dataDir, f));
}

function messagesToText(messages) {
  let out = '';
  for (const { role, content } of messages) {
    if (role === 'user') out += `<|user|>${content}<|end|>`;
    else if (role === 'assistant') out += `<|assistant|>${content}<|end|>`;

    else out += content;
  }
  return out;
}

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

function buildTrainingWindows(conversations, tokenizer, contextLength) {
  const bosId = tokenizer.tokenId('<bos>');
  const eosId = tokenizer.tokenId('<eos>');
  const padId = tokenizer.tokenId('<pad>') ?? eosId;

  const windows = [];
  const step = Math.max(1, Math.floor(contextLength / 2));

  for (const text of conversations) {
    const stream = [bosId, ...tokenizer.encode(text), eosId];

    if (stream.length <= contextLength) {

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

function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function splitConversations(conversations, valFraction = 0.05, seed = 20260914) {
  const rng = mulberry32(seed);
  const copy = conversations.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  const valCount = Math.max(1, Math.floor(copy.length * valFraction));
  return {
    train: copy.slice(0, copy.length - valCount),
    val: copy.slice(copy.length - valCount),
  };
}

module.exports = { listJsonlFiles, loadConversations, buildTrainingWindows, splitConversations, messagesToText };
