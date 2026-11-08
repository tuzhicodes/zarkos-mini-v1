const fs = require('fs');
const path = require('path');

const NEW_TOKENS = [
  '🔥',

];

const ADD_BOTH_FORMS = true;

const CHECKPOINT_DIR = './checkpoint';
const MANIFEST_PATH = path.join(CHECKPOINT_DIR, 'manifest.json');
const TOKENIZER_PATH = path.join(CHECKPOINT_DIR, 'tokenizer.json');
const WEIGHTS_PATH = path.join(CHECKPOINT_DIR, 'weights.bin');

function fail(msg) {
  console.error('\n❌ STOPPED — nothing was changed.\n' + msg + '\n');
  process.exit(1);
}

if (!fs.existsSync(MANIFEST_PATH)) fail(`Can't find ${MANIFEST_PATH}`);
if (!fs.existsSync(TOKENIZER_PATH)) fail(`Can't find ${TOKENIZER_PATH}`);
if (!fs.existsSync(WEIGHTS_PATH)) fail(`Can't find ${WEIGHTS_PATH}`);

const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
const tokenizer = JSON.parse(fs.readFileSync(TOKENIZER_PATH, 'utf8'));
const weightsBuf = fs.readFileSync(WEIGHTS_PATH);

const wteTensor = manifest.tensors.find((t) => t.name === 'wte');
if (!wteTensor) fail('Could not find the "wte" tensor in manifest.json — checkpoint looks unusual.');

const [oldVocab, nEmbd] = wteTensor.shape;
if (oldVocab !== manifest.config.vocabSize) {
  fail(`manifest.json is inconsistent: config.vocabSize (${manifest.config.vocabSize}) ` +
       `does not match wte shape (${oldVocab}). Fix that first — don't proceed.`);
}
if (oldVocab !== tokenizer.idToToken.length) {
  fail(`tokenizer.json vocab (${tokenizer.idToToken.length}) does not match ` +
       `manifest.json's wte shape (${oldVocab}). These must match before adding tokens.`);
}

const expectedBytes = weightsBuf.length;
const totalFloats = expectedBytes / 4;
if (!Number.isInteger(totalFloats)) fail('weights.bin size is not a whole number of float32 values — file may be corrupt.');

console.log(`Current checkpoint: vocabSize=${oldVocab}, nEmbd=${nEmbd}, weights.bin=${expectedBytes} bytes`);

const existing = new Set(tokenizer.idToToken);
const toAdd = [];

function queueIfNew(t) {
  if (!existing.has(t) && !toAdd.includes(t)) toAdd.push(t);
}

for (const t of NEW_TOKENS) {
  queueIfNew(t);
  if (ADD_BOTH_FORMS) queueIfNew(t + '</w>');
}

if (toAdd.length === 0) {
  console.log('\nNothing to add — every token (and its forms) is already in the vocab. No changes made.');
  process.exit(0);
}

console.log(`\nTokens that will be added (${toAdd.length}):`);
toAdd.forEach((t, i) => console.log(`  [${oldVocab + i}] ${JSON.stringify(t)}`));

const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const backupDir = path.join(CHECKPOINT_DIR, `backup_${stamp}`);
fs.mkdirSync(backupDir, { recursive: true });
fs.copyFileSync(MANIFEST_PATH, path.join(backupDir, 'manifest.json'));
fs.copyFileSync(TOKENIZER_PATH, path.join(backupDir, 'tokenizer.json'));
fs.copyFileSync(WEIGHTS_PATH, path.join(backupDir, 'weights.bin'));
console.log(`\nBackup saved to: ${backupDir}`);

tokenizer.idToToken = tokenizer.idToToken.concat(toAdd);
fs.writeFileSync(TOKENIZER_PATH, JSON.stringify(tokenizer, null, 2));

const wteBytes = oldVocab * nEmbd * 4;
if (wteBytes > weightsBuf.length) fail('Computed wte size is bigger than the whole file — checkpoint looks corrupt already.');

const wtePart = weightsBuf.subarray(0, wteBytes);
const restPart = weightsBuf.subarray(wteBytes);

const STD = 0.02;
const newRows = Buffer.alloc(toAdd.length * nEmbd * 4);
let offset = 0;
for (let i = 0; i < toAdd.length * nEmbd; i++) {

  const u1 = Math.random() || 1e-9;
  const u2 = Math.random();
  const val = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * STD;
  newRows.writeFloatLE(val, offset);
  offset += 4;
}

const newWeights = Buffer.concat([wtePart, newRows, restPart]);
const expectedNewSize = weightsBuf.length + toAdd.length * nEmbd * 4;
if (newWeights.length !== expectedNewSize) {
  fail('Size check failed after building new weights.bin — aborting without saving.');
}
fs.writeFileSync(WEIGHTS_PATH, newWeights);

const newVocab = oldVocab + toAdd.length;
manifest.config.vocabSize = newVocab;
wteTensor.shape = [newVocab, nEmbd];
fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));

const checkManifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
const checkTokenizer = JSON.parse(fs.readFileSync(TOKENIZER_PATH, 'utf8'));
const checkWeights = fs.readFileSync(WEIGHTS_PATH);

const ok =
  checkManifest.config.vocabSize === newVocab &&
  checkTokenizer.idToToken.length === newVocab &&
  checkWeights.length === expectedNewSize;

if (!ok) {
  fail('Post-write verification failed! Restore from the backup folder printed above immediately.');
}

console.log(`\n✅ Done. Vocab size: ${oldVocab} → ${newVocab}`);
console.log(`✅ weights.bin: ${weightsBuf.length} → ${checkWeights.length} bytes`);
console.log(`✅ All 3 files verified consistent.`);
console.log(`\nIMPORTANT: update VOCAB_SIZE=${newVocab} in your .env file, then resume training.`);
