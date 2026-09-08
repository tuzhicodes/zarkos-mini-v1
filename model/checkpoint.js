const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

async function saveCheckpoint(model, dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const manifest = { config: model.config, tensors: [] };
  const buffers = [];

  for (const { name, variable } of model.namedVariables()) {
    const data = await variable.data(); // Float32Array
    manifest.tensors.push({ name, shape: variable.shape });
    buffers.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  }

  // BUG FIX: previously wrote manifest.json and weights.bin directly with
  // fs.writeFileSync. If anything reads the checkpoint while that write is
  // in progress — e.g. the chat server hot-reloading the latest weights
  // WHILE training is still running in a separate process — it could see a
  // half-written weights.bin (wrong byte length) and crash, or read stale
  // data. Writing to .tmp files and then renaming into place is atomic on
  // the same filesystem: a concurrent reader only ever sees either the
  // complete old pair or the complete new pair, never a partial file.
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
    const byteLength = numel * 4; // float32
    // Read via DataView.getFloat32 at explicit byte offsets instead of
    // reinterpreting the Buffer's backing ArrayBuffer as a Float32Array.
    // The latter only works when the view's byteOffset is a multiple of 4,
    // which happens to hold today (Node aligns Buffer pool offsets to 8,
    // and readFileSync gives byteOffset 0 for real-sized files) but isn't
    // guaranteed by any spec — this way it's correct regardless.
    const arr = new Float32Array(numel);
    const view = new DataView(buffer.buffer, buffer.byteOffset + offset, byteLength);
    for (let i = 0; i < numel; i++) {
      arr[i] = view.getFloat32(i * 4, true); // little-endian, matches saveCheckpoint's native write
    }
    variable.assign(tf.tensor(arr, meta.shape)); // pass the typed array directly, no Array.from needed
    offset += byteLength;
  });

  console.log(`Loaded checkpoint from ${dir}`);
}

function checkpointExists(dir) {
  return fs.existsSync(path.join(dir, 'manifest.json')) && fs.existsSync(path.join(dir, 'weights.bin'));
}

module.exports = { saveCheckpoint, loadCheckpointInto, checkpointExists };
