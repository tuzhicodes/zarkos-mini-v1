const path = require('path');
const fs = require('fs');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const express = require('express');
const { spawn } = require('child_process');

const config = require('../model/config');
const { BPETokenizer } = require('../model/tokenizer');
const { TinyGPT } = require('../model/transformer');
const { loadCheckpointInto, checkpointExists } = require('../model/checkpoint');
const { generateReply } = require('../model/generate');

const projectRoot = path.join(__dirname, '..');

const app = express();
app.use(express.json());
// BUG FIX: this used to be `express.static(__dirname)`, which serves the
// ENTIRE ui/ folder — including server.js itself — as downloadable static
// files (anyone could open http://yourdomain/server.js and read your
// server source). Now only ui/public/ (which contains just main.html) is
// exposed publicly.
app.use(express.static(path.join(__dirname, 'public')));

let model, tokenizer, ready = false;
let lastLoadedMtime = 0;

function loadModel() {
  // Always reset first so a stale/partial previous state is never reported
  // as ready if loading fails partway through.
  ready = false;
  if (!checkpointExists(config.checkpointDir)) {
    console.log(`No checkpoint found in ${config.checkpointDir} yet.`);
    return false;
  }
  try {
    const oldModel = model;
    tokenizer = BPETokenizer.load(config.checkpointDir);
    config.vocabSize = tokenizer.vocabSize;
    model = new TinyGPT(config);
    loadCheckpointInto(model, config.checkpointDir);
    // BUG FIX: tf.js Variables aren't garbage-collected — replacing `model`
    // without disposing the previous one's weight tensors leaks memory on
    // every reload. Only matters now that loadModel() can run repeatedly
    // (see the hot-reload polling below), not just once at startup.
    if (oldModel) oldModel.dispose();
    ready = true;
    try {
      lastLoadedMtime = fs.statSync(path.join(config.checkpointDir, 'weights.bin')).mtimeMs;
    } catch (e) { /* fine — next poll will just pick it up */ }
    console.log(`Model loaded (${(model.countParams() / 1e6).toFixed(2)}M params, vocab ${tokenizer.vocabSize}).`);
    return true;
  } catch (err) {
    console.error('Failed to load model:', err.message);
    ready = false;
    return false;
  }
}

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

// Lightweight, dependency-free endpoint many hosting panels ping to decide
// whether the app is "up". Plain text on purpose (fast, no JSON parsing
// needed on the panel's side).
app.get('/health', (req, res) => res.type('text').send('ok'));

// Branding shown in the web UI header — pulled from .env so it can be
// changed without touching any code. Sensible fallbacks if unset.
app.get('/api/branding', (req, res) => {
  res.json({
    modelName: process.env.MODEL_NAME || 'Tiny LLM',
    modelId: process.env.MODEL_ID || '',
    owner: process.env.MODEL_OWNER || '',
    brand: process.env.MODEL_BRAND || '',
  });
});

app.get('/api/status', (req, res) => {
  res.json({ ready, params: ready ? model.countParams() : 0, vocabSize: ready ? tokenizer.vocabSize : 0 });
});

app.post('/api/chat', (req, res) => {
  if (!ready) return res.status(503).json({ error: 'Model not loaded. Train it first.' });
  const { message, temperature, maxNewTokens, history } = req.body || {};
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Missing "message" string in request body.' });
  }
  // BUG FIX: history used to be silently ignored — every reply was
  // generated with zero knowledge of earlier turns even though the UI
  // showed a running conversation. Now the client-supplied history (if
  // any) is validated and passed through to generateReply.
  let safeHistory = [];
  if (Array.isArray(history)) {
    safeHistory = history.filter(
      (t) => t && (t.role === 'user' || t.role === 'assistant') && typeof t.content === 'string'
    );
  }
  try {
    const reply = generateReply(model, tokenizer, config, message, {
      temperature: temperature ?? 0.8,
      maxNewTokens: maxNewTokens ?? 100,
      history: safeHistory,
    });
    res.json({ reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Generation failed.', details: err.message });
  }
});

// ---------------- Training controls ----------------
// Training (model/train.js) is a synchronous, CPU-bound loop. Running it
// inline inside this Express process would block the event loop for the
// entire run — the chat API and even the "Stop" button click would never
// get a chance to be handled until training finished on its own. Instead
// we spawn it as a separate child process:
//   - the server stays responsive (status polling, chat, stop button) the
//     whole time training runs
//   - "Stop" is a real interrupt: killing the child process actually halts
//     training immediately, instead of needing cooperative cancellation
//     checks sprinkled through the training loop
//   - train.js already checkpoints after every epoch, so stopping early
//     just means "resume from the last completed epoch" next time

let trainProc = null;
let trainState = 'idle'; // idle | running | stopped | done | error
let trainLogs = [];
const MAX_LOG_LINES = 500;

function pushLog(chunk) {
  for (const line of chunk.toString().split('\n')) {
    if (line.length) trainLogs.push(line);
  }
  if (trainLogs.length > MAX_LOG_LINES) {
    trainLogs = trainLogs.slice(trainLogs.length - MAX_LOG_LINES);
  }
}

// ---------------- Out-of-memory auto-recovery ----------------
// A training process that gets killed with signal SIGKILL (as opposed to
// SIGTERM, which is what our own Stop button sends) was almost certainly
// killed by the HOST's out-of-memory guard, not by the user. This is very
// common on memory-limited hosting: the default model/batch/context size
// assumes a few GB of RAM, but small hosting plans (common for Discord-bot
// panels being reused to also host this) may only give the container
// 256MB-1GB. Rather than making the user manually figure out which
// .env number to lower and retry by hand, we do it automatically: each
// SIGKILL before the first checkpoint is saved halves BATCH_SIZE (and once
// that hits 1, halves CONTEXT_LENGTH instead) and restarts training with
// that override, up to a few attempts. Once a checkpoint has been saved
// successfully we stop adjusting — changing CONTEXT_LENGTH after weights
// exist would make them incompatible (positional embedding shape changes).
const MAX_OOM_RETRIES = 4;
let oomRetries = 0;
let overrideBatchSize = null;
let overrideContextLength = null;

function launchTraining() {
  trainState = 'running';
  const env = { ...process.env };
  if (overrideBatchSize !== null) env.BATCH_SIZE = String(overrideBatchSize);
  if (overrideContextLength !== null) env.CONTEXT_LENGTH = String(overrideContextLength);

  trainProc = spawn(process.execPath, [path.join(projectRoot, 'index.js'), 'train'], {
    cwd: projectRoot,
    env,
  });

  trainProc.stdout.on('data', pushLog);
  trainProc.stderr.on('data', pushLog);

  trainProc.on('close', (code, signal) => {
    trainProc = null;

    const hasCheckpoint = checkpointExists(config.checkpointDir);

    if (signal === 'SIGKILL' && !hasCheckpoint && oomRetries < MAX_OOM_RETRIES) {
      oomRetries++;
      const prevBatch = overrideBatchSize ?? config.batchSize;
      const prevCtx = overrideContextLength ?? config.contextLength;
      if (prevBatch > 1) {
        overrideBatchSize = Math.max(1, Math.floor(prevBatch / 2));
        pushLog(`Training was killed (likely out-of-memory on this host). Auto-retrying with a smaller BATCH_SIZE: ${prevBatch} -> ${overrideBatchSize} (attempt ${oomRetries}/${MAX_OOM_RETRIES}).`);
      } else {
        overrideContextLength = Math.max(16, Math.floor(prevCtx / 2));
        pushLog(`Training was killed (likely out-of-memory on this host). BATCH_SIZE is already at minimum, lowering CONTEXT_LENGTH instead: ${prevCtx} -> ${overrideContextLength} (attempt ${oomRetries}/${MAX_OOM_RETRIES}).`);
      }
      launchTraining();
      return;
    }

    if (signal === 'SIGKILL') {
      trainState = 'error';
      pushLog(
        `Training was killed by the host (out-of-memory), even after ${oomRetries} automatic retries with a ` +
        `smaller batch/context size. This host likely doesn't have enough RAM for this dataset size. Try: ` +
        `lowering N_EMBD/N_LAYER/FFN_HIDDEN in .env for a smaller model, splitting your data into a smaller ` +
        `subset, or moving to a host with more memory.`
      );
    } else if (signal) {
      // Any other signal (SIGTERM etc.) means the user pressed Stop, or the
      // server itself was told to shut down — not an OOM kill.
      trainState = 'stopped';
      pushLog(`Training stopped (signal ${signal}).`);
    } else if (code === 0) {
      trainState = 'done';
      oomRetries = 0; // reset backoff state now that a full run has succeeded
      pushLog('Training finished successfully.');
    } else {
      trainState = 'error';
      pushLog(`Training process exited with code ${code}.`);
    }
    // Reload whatever checkpoint exists now so the Chat tab immediately
    // reflects the latest trained weights without restarting the server.
    loadModel();
  });

  trainProc.on('error', (err) => {
    trainState = 'error';
    pushLog(`Failed to start training process: ${err.message}`);
    trainProc = null;
  });
}

app.post('/api/train/start', (req, res) => {
  if (trainProc) {
    return res.status(409).json({ error: 'Training is already running.' });
  }
  trainLogs = [];
  oomRetries = 0; // fresh manual start always gets a clean slate of retries
  pushLog(`Starting training (data: ${config.dataDir}, checkpoint: ${config.checkpointDir})...`);
  launchTraining();
  res.json({ started: true });
});

app.post('/api/train/stop', (req, res) => {
  if (!trainProc) {
    return res.status(409).json({ error: 'No training run in progress.' });
  }
  trainProc.kill('SIGTERM');
  res.json({ stopping: true });
});

app.get('/api/train/status', (req, res) => {
  res.json({
    state: trainState,
    running: !!trainProc,
    logs: trainLogs,
    modelReady: ready,
  });
});

// Don't leave an orphaned training process behind if the server itself
// is stopped/restarted while a run is in progress.
process.on('exit', () => {
  if (trainProc) trainProc.kill('SIGTERM');
});

loadModel();

// ---------------- Automatic hot-reload of the latest checkpoint ----------------
// BUG FIX: previously the ONLY way to chat with a checkpoint saved mid-
// training was to press Stop first — the running chat server kept using
// whatever model it loaded at startup (or after the previous training run
// finished) until loadModel() was explicitly called again. Since training
// now saves a checkpoint periodically (CHECKPOINT_EVERY) as well as at
// every epoch end, this polls the checkpoint's weights.bin mtime and
// hot-reloads automatically whenever a newer one appears — no need to stop
// training to test progress. saveCheckpoint()'s atomic rename (see
// model/checkpoint.js) guarantees this never reads a half-written file.
function checkForNewCheckpoint() {
  const weightsPath = path.join(config.checkpointDir, 'weights.bin');
  fs.stat(weightsPath, (err, stats) => {
    if (err) return; // no checkpoint yet, or transient race — just try again next tick
    if (stats.mtimeMs > lastLoadedMtime) {
      lastLoadedMtime = stats.mtimeMs;
      console.log('Detected an updated checkpoint on disk — hot-reloading into the chat server...');
      loadModel();
    }
  });
}
setInterval(checkForNewCheckpoint, 10000);

// BUG FIX: previously nothing caught unexpected async errors — a single
// uncaught exception or rejected promise anywhere (e.g. a bad request body,
// a transient fs error) would silently kill the whole Node process. On a
// hosting panel that looks exactly like "the website doesn't open": the
// domain/port were configured correctly, but the process had already
// crashed and nothing was listening anymore. These handlers make sure the
// real error always gets printed to the panel's logs instead of vanishing.
process.on('uncaughtException', (err) => {
  console.error('[uncaughtException]', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[unhandledRejection]', reason);
});

const PORT = process.env.PORT || 3000;
// Bind address: on a hosting panel "localhost" only accepts connections
// from inside the same container, so the panel's reverse proxy/router
// (coming from a different address) can't reach it — this is the
// "address is different on the server" problem. 0.0.0.0 means "listen on
// every network interface", which works both locally and on hosting.
// Override via HOST (or IP, some panels expose that name instead) in .env
// if a specific address is ever required.
const HOST_RAW = process.env.HOST || process.env.IP || '0.0.0.0';
// Guard against the exact mistake this session hit: HOST set to a domain
// name (e.g. "yourapp.example.net") instead of a bindable local address.
// A quick IPv4/"localhost" shape check catches it before even attempting
// the listen() call, avoiding an extra crash-and-retry cycle on panels that
// count crashes (like HidenCloud's "aborting automatic restart" guard).
const looksLikeBindableAddress = /^(0\.0\.0\.0|127\.0\.0\.1|localhost|(\d{1,3}\.){3}\d{1,3})$/.test(HOST_RAW);
if (!looksLikeBindableAddress) {
  console.warn(
    `\nHOST="${HOST_RAW}" looks like a domain name, not a bindable address.\n` +
    `Using 0.0.0.0 instead — your panel's domain/reverse-proxy will still\n` +
    `route traffic to this container on the port below; the app itself\n` +
    `never needs to bind to the domain directly. Remove HOST from .env (or\n` +
    `your panel's environment variables) to silence this warning.\n`
  );
}
const HOST = looksLikeBindableAddress ? HOST_RAW : '0.0.0.0';

// Diagnostic banner: most "website won't open" hosting issues come down to
// the app listening on a different PORT/HOST than what the panel's
// domain/proxy actually forwards to. Printing every candidate env var the
// process saw makes a mismatch obvious in the panel's log viewer instead of
// requiring guesswork.
console.log('--- Tiny LLM server starting ---');
console.log('  PORT env:', process.env.PORT, '| HOST env:', process.env.HOST, '| IP env:', process.env.IP);
console.log('  -> binding to', `${HOST}:${PORT}`);
console.log('  If your hosting panel shows you a specific port/domain to use,');
console.log('  make sure it matches the values above — panels usually inject');
console.log('  their own PORT automatically, which overrides .env\'s PORT.');

function startServer(port, host, isRetry = false) {
  const srv = app.listen(port, host, () => {
    console.log(`Tiny LLM chat server running at http://${host}:${port} (local: http://localhost:${port})`);
  });

  // BUG FIX: previously a listen failure (e.g. port already in use, no
  // permission to bind, or an unbindable HOST) threw an unhandled 'error'
  // event and crashed the process with a raw stack trace — no clue what to
  // actually do about it.
  srv.on('error', (err) => {
    if (err.code === 'EADDRNOTAVAIL' && !isRetry) {
      // This is almost always caused by HOST being set to a domain name
      // (e.g. "yourapp.yourhost.net") instead of a bindable address. A
      // domain is something a DNS/reverse-proxy layer routes TO your
      // container from the outside — the process inside the container can
      // never actually listen "as" that domain, only on its own network
      // interfaces. Auto-retrying on 0.0.0.0 keeps the app alive instead of
      // crash-looping every time, while the message below tells the user
      // the real fix (remove HOST from .env / the panel's env settings).
      console.error(
        `\nCould not bind to HOST="${host}" (${err.message}).\n` +
        `HOST should be an IP like 0.0.0.0, not a domain name — your panel's\n` +
        `domain/reverse-proxy forwards traffic to this port from the outside,\n` +
        `the app itself never binds to the domain directly.\n` +
        `Retrying on 0.0.0.0 now. To fix this permanently, remove/clear the\n` +
        `HOST value in .env (or your panel's environment variables) so it\n` +
        `defaults to 0.0.0.0.\n`
      );
      startServer(port, '0.0.0.0', true);
      return;
    }
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${port} is already in use. Another process (maybe a previous run of this same app) is still running. Stop it, or set a different PORT in .env.`);
    } else if (err.code === 'EACCES') {
      console.error(`No permission to bind to port ${port}/host ${host}. Ports below 1024 usually need elevated permissions — try a port like 3000+ instead.`);
    } else {
      console.error('Server failed to start:', err);
    }
    process.exit(1);
  });

  return srv;
}

startServer(PORT, HOST);
