const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const requested = String(process.env.DEVICE || 'cpu').trim().toLowerCase();
const device = requested === 'gpu' ? 'gpu' : 'cpu';

let tf;

let gpuSignal = null;

if (device === 'gpu') {
  try {

    tf = require('@tensorflow/tfjs-node-gpu');
  } catch (err) {

    const isTermux = !!(process.env.PREFIX && process.env.PREFIX.includes('com.termux'))
      || !!process.env.TERMUX_VERSION;
    if (isTermux) {
      throw new Error(
        'DEVICE=gpu was requested, but this is running on Termux/Android. ' +
        '@tensorflow/tfjs-node-gpu requires an NVIDIA CUDA GPU and only ships native ' +
        'binaries for desktop/server Linux, Windows, and macOS (x64) — phone GPUs are not ' +
        'CUDA-capable, so this can never work here, no matter how many times the package is ' +
        're-installed. Set DEVICE=cpu in .env instead — the pure-JS backend runs fully on ' +
        'Termux and produces byte-identical checkpoints to any other DEVICE=cpu machine.\n' +
        `Original error: ${err.message}`
      );
    }
    throw new Error(
      'DEVICE=gpu was requested, but @tensorflow/tfjs-node-gpu could not be loaded. ' +
      'Install the optional GPU dependency and use a compatible Linux + NVIDIA/CUDA runtime.\n' +
      `Original error: ${err.message}`
    );
  }

  if (tf.getBackend() !== 'tensorflow') {
    throw new Error(
      `DEVICE=gpu was requested and @tensorflow/tfjs-node-gpu loaded, but the active backend ` +
      `is "${tf.getBackend()}", not "tensorflow". This shouldn't happen — please report it.`
    );
  }

  const backend = tf.backend();
  if (backend && typeof backend.isUsingGpuDevice === 'boolean') {
    gpuSignal = backend.isUsingGpuDevice;
    if (gpuSignal === false) {
      console.warn(
        'WARNING: @tensorflow/tfjs-node-gpu loaded, but it reports it is not using a GPU device. ' +
        'Check `nvidia-smi` and CUDA/cuDNN driver compatibility — training may be running on CPU ' +
        'despite DEVICE=gpu.'
      );
    }
  } else {
    console.warn(
      'Note: could not confirm GPU usage from tfjs-node-gpu directly (no reliable API for it). ' +
      'The native GPU backend loaded successfully, which is the strongest signal available from ' +
      'here — for certainty, watch `nvidia-smi` while training runs.'
    );
  }
} else {

  tf = require('@tensorflow/tfjs');

  tf.backend();
}

function deviceInfo() {
  return {
    device,
    backend: tf.getBackend(),
    gpuConfirmed: gpuSignal === true ? true : gpuSignal === false ? false : null,
  };
}

module.exports = tf;
module.exports.device = device;
module.exports.deviceInfo = deviceInfo;
