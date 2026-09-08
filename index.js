const path = require('path');
// Anchor .env to this file's own folder, not the process's working
// directory. If the hosting panel launches the app from a different CWD
// (common — panels often cd elsewhere before running the start command),
// plain require('dotenv').config() silently finds no .env and every
// setting (START_MODE, PORT, HOST...) falls back to code defaults with
// zero warning. This guarantees .env is found regardless of CWD.
require('dotenv').config({ path: path.join(__dirname, '.env') });

// Hosting panels (HideNCloud etc.) typically just run the main file directly
// (`node index.js`, no CLI args) and expect the process to stay alive as a
// server. Previously, no args meant we hit the default branch, printed a
// usage message, and exited cleanly (code 0) — the panel's supervisor saw
// that as an instant crash and gave up restarting. START_MODE in .env lets
// you control the default without touching code; it defaults to 'serve' so
// a bare `node index.js` (or `npm start`) boots the web server.
const command = process.argv[2] || process.env.START_MODE || 'serve';

async function chatOnce() {
  const readline = require('readline');
  const tf = require('@tensorflow/tfjs');
  const config = require('./model/config');
  const { BPETokenizer } = require('./model/tokenizer');
  const { TinyGPT } = require('./model/transformer');
  const { loadCheckpointInto, checkpointExists } = require('./model/checkpoint');
  const { generateReply } = require('./model/generate');

  if (!checkpointExists(config.checkpointDir)) {
    console.error('No checkpoint found. Run `npm run train` first.');
    process.exit(1);
  }

  const tokenizer = BPETokenizer.load(config.checkpointDir);
  config.vocabSize = tokenizer.vocabSize;
  const model = new TinyGPT(config);
  loadCheckpointInto(model, config.checkpointDir);

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log('Tiny LLM chat — type "exit" to quit.\n');

  const ask = () =>
    rl.question('You: ', (msg) => {
      if (msg.trim().toLowerCase() === 'exit') {
        rl.close();
        return;
      }
      const reply = generateReply(model, tokenizer, config, msg);
      console.log('Bot:', reply, '\n');
      ask();
    });
  ask();
}

// A small fixed set of prompts used by `node index.js eval` to compare
// checkpoints apples-to-apples over time. Kept deliberately short/varied —
// the same prompts, greedy-decoded, run against successive checkpoints so
// changes in output reflect the model actually improving, not sampling
// randomness (temperature=0.8 top-k sampling gives a different answer to
// the same prompt every time, which makes two checkpoints impossible to
// compare by eye).
const EVAL_PROMPTS = [
  'hi',
  'kaise ho',
  'aaj kya kar rahe ho',
  'tumhara naam kya hai',
  'mujhe ek joke sunao',
  '5 + 7 kitna hota hai',
  'ganga nadi kaha se nikalti hai',
  'bye',
];

async function evalOnce() {
  const config = require('./model/config');
  const { BPETokenizer } = require('./model/tokenizer');
  const { TinyGPT } = require('./model/transformer');
  const { loadCheckpointInto, checkpointExists } = require('./model/checkpoint');
  const { generateReply } = require('./model/generate');

  if (!checkpointExists(config.checkpointDir)) {
    console.error('No checkpoint found. Run `npm run train` first.');
    process.exit(1);
  }

  const tokenizer = BPETokenizer.load(config.checkpointDir);
  config.vocabSize = tokenizer.vocabSize;
  const model = new TinyGPT(config);
  loadCheckpointInto(model, config.checkpointDir);

  console.log(`--- Fixed-prompt eval (greedy decoding, temperature≈0) — ${config.checkpointDir} ---\n`);
  for (const prompt of EVAL_PROMPTS) {
    // temperature is clamped to a minimum of 1e-6 in sampleNextToken, which
    // makes softmax so sharp it's effectively argmax — i.e. greedy decoding.
    // Same prompt -> same output every run, so successive checkpoints can
    // be compared side by side instead of fighting sampling noise.
    const reply = generateReply(model, tokenizer, config, prompt, { temperature: 0, topK: 1 });
    console.log(`> ${prompt}\n${reply}\n`);
  }
}

switch (command) {
  case 'train':
    require('./model/train').main().catch((e) => {
      console.error(e);
      process.exit(1);
    });
    break;
  case 'serve':
    require('./ui/server');
    break;
  case 'chat':
    chatOnce();
    break;
  case 'eval':
    evalOnce().catch((e) => {
      console.error(e);
      process.exit(1);
    });
    break;
  default:
    // Fail-safe: an unrecognized/empty command used to print this usage
    // text and let the script run off the end, exiting with code 0. On a
    // hosting panel that expects a long-running server, that reads as an
    // instant crash and triggers a restart loop. Now we still print the
    // usage for humans running it by hand, but we boot the server anyway
    // so the process never just exits.
    console.log(`Usage:
  node index.js train   # train the model on data/*.jsonl
  node index.js serve   # start the web chat UI
  node index.js chat    # chat with the model in the terminal
  node index.js eval    # run fixed prompts with greedy decoding to compare checkpoints
No/unrecognized command given — defaulting to "serve" so the process stays alive.
`);
    require('./ui/server');
}
