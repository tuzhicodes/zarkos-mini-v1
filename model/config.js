const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const num = (v, d) => (v === undefined ? d : Number(v));

// Project root = one level up from /model. Same reasoning as the .env fix
// in index.js/server.js: if something launches this app from a different
// CWD (hosting panel, cron job, `npm run train` from another folder, VS
// Code "run" button, etc.), a plain relative path like './data' resolves
// against THAT cwd, not the project folder — so training would silently
// read/write the wrong "data"/"checkpoint" directory (or create empty new
// ones) depending on where the process happened to be launched from.
// path.resolve() anchors it to the project root instead, while still
// respecting an absolute path if the user sets one in .env.
const projectRoot = path.join(__dirname, '..');

const config = {
  // --- Tokenizer / data ---
  vocabSize: num(process.env.VOCAB_SIZE, 8000),
  dataDir: path.resolve(projectRoot, process.env.DATA_DIR || 'data'),

  // --- Architecture ---
  contextLength: num(process.env.CONTEXT_LENGTH, 128),
  nEmbd: num(process.env.N_EMBD, 128),
  nHead: num(process.env.N_HEAD, 4),
  nLayer: num(process.env.N_LAYER, 4),
  ffnHidden: num(process.env.FFN_HIDDEN, 512),

  // --- Training ---
  batchSize: num(process.env.BATCH_SIZE, 16),
  epochs: num(process.env.EPOCHS, 5),
  learningRate: num(process.env.LEARNING_RATE, 3e-4),
  // Fraction of conversations held out as a validation set (never trained
  // on). Lets you tell "model is memorizing training data" apart from
  // "model is actually generalizing" — see model/train.js.
  valFraction: num(process.env.VAL_FRACTION, 0.05),

  // --- Special tokens (added to vocab regardless of BPE merges) ---
  specialTokens: ['<pad>', '<unk>', '<bos>', '<eos>', '<|user|>', '<|assistant|>', '<|end|>'],

  // --- Checkpoint ---
  checkpointDir: path.resolve(projectRoot, process.env.CHECKPOINT_DIR || 'checkpoint'),
  // How often (in steps) to save a checkpoint DURING an epoch, in addition
  // to the save that already happens at the end of every epoch. Useful when
  // a single epoch has thousands of steps (like now) and you don't want to
  // lose hours of progress to a crash/restart before the epoch finishes.
  // Set to 0 in .env to disable mid-epoch saving and keep old behavior.
  checkpointEvery: num(process.env.CHECKPOINT_EVERY, 500),
};

// Sanity check: nEmbd must be divisible by nHead
if (config.nEmbd % config.nHead !== 0) {
  throw new Error(
    `nEmbd (${config.nEmbd}) must be divisible by nHead (${config.nHead})`
  );
}
config.headDim = config.nEmbd / config.nHead;

module.exports = config;
