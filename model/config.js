const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const num = (v, d) => (v === undefined ? d : Number(v));

const projectRoot = path.join(__dirname, '..');

const device = String(process.env.DEVICE || 'cpu').trim().toLowerCase();
if (!['cpu', 'gpu'].includes(device)) {
  throw new Error(`Invalid DEVICE=\"${device}\". Use DEVICE=cpu or DEVICE=gpu.`);
}

const config = {
  device,

  modelName: process.env.MODEL_NAME || 'ZarKos',
  modelId: process.env.MODEL_ID || 'tuzhi/zarkos-mini-v1',
  modelOwner: process.env.MODEL_OWNER || 'TuZhi Codes',
  modelBrand: process.env.MODEL_BRAND || 'TuZhi Studio',

  vocabSize: num(process.env.VOCAB_SIZE, 8000),
  dataDir: path.resolve(projectRoot, process.env.DATA_DIR || 'data'),

  contextLength: num(process.env.CONTEXT_LENGTH, 128),
  nEmbd: num(process.env.N_EMBD, 256),
  nHead: num(process.env.N_HEAD, 8),
  nLayer: num(process.env.N_LAYER, 6),
  ffnHidden: num(process.env.FFN_HIDDEN, 1024),

  batchSize: num(process.env.BATCH_SIZE, 2),
  epochs: num(process.env.EPOCHS, 10),
  learningRate: num(process.env.LEARNING_RATE, 3e-4),

  valFraction: num(process.env.VAL_FRACTION, 0.05),

  specialTokens: ['<pad>', '<unk>', '<bos>', '<eos>', '<|user|>', '<|assistant|>', '<|end|>'],

  checkpointDir: path.resolve(projectRoot, process.env.CHECKPOINT_DIR || 'checkpoint'),

  checkpointEvery: num(process.env.CHECKPOINT_EVERY, 200),
};

if (config.nEmbd % config.nHead !== 0) {
  throw new Error(
    `nEmbd (${config.nEmbd}) must be divisible by nHead (${config.nHead})`
  );
}
config.headDim = config.nEmbd / config.nHead;

module.exports = config;
