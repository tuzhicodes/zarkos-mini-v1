const fs = require('fs');
const path = require('path');
const tf = require('./tf');

async function saveCheckpoint(model, dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const manifest = { config: model.config, tensors: [] };
  const buffers = [];

  for (const { name, variable } of model.namedVariables()) {
    const data = await variable.data();
    manifest.tensors.push({ name, shape: variable.shape });
    buffers.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  }

  const manifestPath = path.join(dir, 'manifest.json');
  const weightsPath = path.join(dir, 'weights.bin');
  const manifestTmp = manifestPath + '.tmp';
  const weightsTmp = weightsPath + '.tmp';

  fs.writeFileSync(manifestTmp, JSON.stringify(manifest, null, 2));
  fs.writeFileSync(weightsTmp, Buffer.concat(buffers));
  fs.renameSync(weightsTmp, weightsPath);
  fs.renameSync(manifestTmp, manifestPath);

  console.log(`Saved checkpoint to ${dir} (${buffers.length} tensors)`);
}

function loadCheckpointInto(model, dir) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf-8'));
  const buffer = fs.readFileSync(path.join(dir, 'weights.bin'));

  const expected = model.config;
  const saved = manifest.config || {};
  const shapeKeys = ['vocabSize', 'contextLength', 'nEmbd', 'nHead', 'nLayer', 'ffnHidden'];
  const changed = shapeKeys.filter((key) => Number(saved[key]) !== Number(expected[key]));
  if (changed.length) {
    throw new Error(
      `Checkpoint architecture mismatch for: ${changed.join(', ')}. ` +
      `Saved checkpoint cannot be used with the current .env model settings. Archive/delete checkpoint/ and train a fresh model.`
    );
  }

  const named = model.namedVariables();
  if (named.length !== manifest.tensors.length) {
    throw new Error(
      `Checkpoint has ${manifest.tensors.length} tensors but model expects ${named.length}. ` +
        `Did the config change since this checkpoint was saved?`
    );
  }

  let offset = 0;
  named.forEach(({ name, variable }, i) => {
    const meta = manifest.tensors[i];
    if (meta.name !== name) {
      throw new Error(`Checkpoint tensor order mismatch: expected "${name}", found "${meta.name}"`);
    }
    const numel = meta.shape.reduce((a, b) => a * b, 1);
    const byteLength = numel * 4;

    const arr = new Float32Array(numel);
    const view = new DataView(buffer.buffer, buffer.byteOffset + offset, byteLength);
    for (let i = 0; i < numel; i++) {
      arr[i] = view.getFloat32(i * 4, true);
    }
    variable.assign(tf.tensor(arr, meta.shape));
    offset += byteLength;
  });

  console.log(`Loaded checkpoint from ${dir}`);
}

function checkpointExists(dir) {
  return fs.existsSync(path.join(dir, 'manifest.json')) && fs.existsSync(path.join(dir, 'weights.bin'));
}

async function saveTrainingMeta(dir, { completedEpochs, globalStep, bestValLoss }, optimizer) {
  const metaPath = path.join(dir, 'meta.json');
  const optPath = path.join(dir, 'optimizer.bin');
  const metaTmp = metaPath + '.tmp';
  const optTmp = optPath + '.tmp';

  let weights = [];
  if (optimizer && typeof optimizer.getWeights === 'function') {
    try {
      weights = await optimizer.getWeights();
    } catch (err) {
      console.warn(`Could not read optimizer state (will resume without momentum): ${err.message}`);
      weights = [];
    }
  }

  const meta = {
    completedEpochs,
    globalStep,

    bestValLoss: bestValLoss ?? null,
    optimizer: weights.length
      ? { tensors: weights.map((w) => ({ name: w.name, shape: w.tensor.shape })) }
      : null,
  };

  try {
    if (weights.length) {
      const buffers = [];
      for (const w of weights) {
        const data = await w.tensor.data();
        buffers.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
      }
      fs.writeFileSync(optTmp, Buffer.concat(buffers));
      fs.renameSync(optTmp, optPath);
    } else if (fs.existsSync(optPath)) {
      fs.unlinkSync(optPath);
    }
    fs.writeFileSync(metaTmp, JSON.stringify(meta, null, 2));
    fs.renameSync(metaTmp, metaPath);
  } catch (err) {
    console.warn(`Could not save training progress/optimizer state: ${err.message}`);
  }
}

async function loadTrainingMeta(dir, optimizer) {
  const metaPath = path.join(dir, 'meta.json');
  const optPath = path.join(dir, 'optimizer.bin');
  const fallback = { completedEpochs: 0, globalStep: 0, bestValLoss: null };
  if (!fs.existsSync(metaPath)) return fallback;

  let meta;
  try {
    meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  } catch (err) {
    console.warn(`Could not read meta.json (starting epoch/step count from 0): ${err.message}`);
    return fallback;
  }

  const result = {
    completedEpochs: Number(meta.completedEpochs) || 0,
    globalStep: Number(meta.globalStep) || 0,

    bestValLoss: typeof meta.bestValLoss === 'number' ? meta.bestValLoss : null,
  };

  if (optimizer && meta.optimizer && Array.isArray(meta.optimizer.tensors) && fs.existsSync(optPath)) {
    try {
      const buffer = fs.readFileSync(optPath);
      let offset = 0;
      const weightValues = meta.optimizer.tensors.map((t) => {
        const numel = t.shape.reduce((a, b) => a * b, 1);
        const byteLength = numel * 4;
        const arr = new Float32Array(numel);
        const view = new DataView(buffer.buffer, buffer.byteOffset + offset, byteLength);
        for (let i = 0; i < numel; i++) arr[i] = view.getFloat32(i * 4, true);
        offset += byteLength;
        return { name: t.name, tensor: tf.tensor(arr, t.shape) };
      });
      await optimizer.setWeights(weightValues);
      console.log(`Restored optimizer momentum (${weightValues.length} tensors) from checkpoint.`);
    } catch (err) {
      console.warn(`Could not restore optimizer momentum from checkpoint (resuming with fresh momentum): ${err.message}`);
    }
  }

  return result;
}

module.exports = { saveCheckpoint, loadCheckpointInto, checkpointExists, saveTrainingMeta, loadTrainingMeta };
