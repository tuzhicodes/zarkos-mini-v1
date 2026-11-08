const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

console.log('--- ZarKos Device Verification ---');

try {
  const tf = require('./model/tf');
  const info = tf.deviceInfo();

  console.log(`Requested device (from .env): ${info.device}`);
  console.log(`Active TensorFlow backend:    ${info.backend}`);

  if (info.device === 'gpu') {
    if (info.gpuConfirmed === true) {
      console.log('GPU Confirmation:             Confirmed active GPU device');
    } else if (info.gpuConfirmed === false) {
      console.log('GPU Confirmation:             Warning: Running on CPU despite GPU request');
    } else {
      console.log('GPU Confirmation:             Native TensorFlow backend loaded');
    }
  }

  const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
  const b = tf.tensor2d([[5, 6], [7, 8]], [2, 2]);
  const c = a.matMul(b);
  const result = c.dataSync();

  a.dispose();
  b.dispose();
  c.dispose();

  if (result[0] === 19 && result[3] === 50) {
    console.log('Tensor computation check:     PASSED');
  } else {
    throw new Error('Tensor computation produced incorrect values.');
  }

  console.log('\n✅ Verification successful! The environment is ready for training.\n');
  process.exit(0);
} catch (err) {
  console.error('\n❌ Device verification failed:');
  console.error(err.message);
  console.error('\nIf running on a phone/Termux or machine without NVIDIA GPU, ensure DEVICE=cpu in .env.');
  console.error('If running on Google Colab, ensure Runtime is set to GPU and @tensorflow/tfjs-node-gpu is installed.\n');
  process.exit(1);
}
