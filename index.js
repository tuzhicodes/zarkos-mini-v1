const path = require('path');

require('dotenv').config({ path: path.join(__dirname, '.env') });

const command = process.argv[2] || process.env.START_MODE || 'serve';

(function loadTfQuietly() {
  const originalLog = console.log;
  console.log = (...args) => {
    const text = args.join(' ');
    if (text.includes('tfjs-node') || /^=+$/.test(text.trim())) return;
    originalLog(...args);
  };
  require('./model/tf');
  console.log = originalLog;
})();

async function chatOnce() {
  const readline = require('readline');
  const tf = require('./model/tf');
  const config = require('./model/config');
  const banner = require('./model/banner');
  const { BPETokenizer } = require('./model/tokenizer');
  const { ZarKosLM } = require('./model/transformer');
  const { loadCheckpointInto, checkpointExists } = require('./model/checkpoint');
  const { generateReply } = require('./model/generate');

  banner.printPreamble('chat');

  if (!checkpointExists(config.checkpointDir)) {
    console.error('No checkpoint found. Run `npm run train` first.');
    process.exit(1);
  }

  const tokenizer = BPETokenizer.load(config.checkpointDir);
  config.vocabSize = tokenizer.vocabSize;
  const model = new ZarKosLM(config);

  loadCheckpointInto(model, config.checkpointDir);
  banner.printBanner('Chat — type "/exit" to quit');

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = () =>
    rl.question(banner.boldGreen('You: '), async (msg) => {
      const cmd = msg.trim().toLowerCase();
      if (cmd === '/exit' || cmd === 'exit') {
        rl.close();
        return;
      }
      process.stdout.write(banner.boldGreen('Bot: '));
      const maxTokens = Number(process.env.MAX_NEW_TOKENS) || 30;
      await generateReply(model, tokenizer, config, msg, {
        maxNewTokens: maxTokens,
        onToken: (tok) => process.stdout.write(banner.white(tok)),
      });
      console.log('\n');
      ask();
    });
  ask();
}

const EVAL_PROMPTS = [
  'hello',
  'how are you?',
  'what is your name?',
  'tell me a short joke',
  'what is 5 + 7?',
  'what is the capital of France?',
  'explain how plants make food',
  'goodbye',
];

async function evalOnce() {
  const config = require('./model/config');
  const banner = require('./model/banner');
  const { BPETokenizer } = require('./model/tokenizer');
  const { ZarKosLM } = require('./model/transformer');
  const { loadCheckpointInto, checkpointExists } = require('./model/checkpoint');
  const { generateReply } = require('./model/generate');

  banner.printPreamble('eval');

  if (!checkpointExists(config.checkpointDir)) {
    console.error('No checkpoint found. Run `npm run train` first.');
    process.exit(1);
  }

  const tokenizer = BPETokenizer.load(config.checkpointDir);
  config.vocabSize = tokenizer.vocabSize;
  const model = new ZarKosLM(config);
  loadCheckpointInto(model, config.checkpointDir);
  banner.printBanner('Eval — running fixed prompts');

  console.log('--- Fixed-prompt eval (greedy decoding, temperature≈0) ---\n');
  for (const prompt of EVAL_PROMPTS) {
    process.stdout.write(`> ${prompt}\n`);
    await generateReply(model, tokenizer, config, prompt, {
      maxNewTokens: 25,
      temperature: 0,
      topK: 1,
      onToken: (tok) => process.stdout.write(tok),
    });
    process.stdout.write('\n\n');
  }
}

switch (command) {
  case 'train': {
    const banner = require('./model/banner');
    banner.printPreamble('train');
    banner.printBanner('Training — starting');
    require('./model/train').main().catch((e) => {
      console.error(e);
      process.exit(1);
    });
    break;
  }
  case 'serve': {
    const banner = require('./model/banner');
    banner.printPreamble('serve');
    banner.printBanner('Dashboard — starting');
    require('./ui/server');
    break;
  }
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

    console.log(`Usage:
  node index.js train   # train the model on data/*.jsonl
  node index.js serve   # start the web chat UI
  node index.js chat    # chat with the model in the terminal
  node index.js eval    # run fixed prompts with greedy decoding to compare checkpoints
No/unrecognized command given — defaulting to "serve" so the process stays alive.
`);
    require('./ui/server');
}
