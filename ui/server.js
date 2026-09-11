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
const {
  login,
  getSession,
  clearSessionCookie,
  destroySession,
  requireLogin,
  requireApiKey,
  getCredential,
} = require('../auth');

const projectRoot = path.join(__dirname, '..');
const app = express();

app.use(express.json({ limit: '256kb' }));
app.use((req, res, next) => {
  const origin = process.env.API_CORS_ORIGIN || '*';
  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, Authorization');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
app.use(express.static(path.join(__dirname, 'public')));

// Keep the installable PWA manifest truthful to the same .env branding used
// by the dashboard/API. This avoids hard-coded model names in the UI shell.
app.get('/manifest.webmanifest', (req, res) => {
  const name = process.env.MODEL_NAME || 'ZarKos';
  const shortName = name.replace(/\s+(Training|Model)$/i, '').slice(0, 20) || 'ZarKos';
  res.type('application/manifest+json').json({
    name: `${name} Training`,
    short_name: shortName,
    description: `Training dashboard for ${name}.`,
    start_url: '/',
    scope: '/',
    display: 'standalone',
    background_color: '#0c0d10',
    theme_color: '#0c0d10',
    orientation: 'portrait-primary',
    icons: [{ src: '/icon/app.png', sizes: '192x192', type: 'image/png', purpose: 'any maskable' }],
  });
});

let model = null;
let tokenizer = null;
let ready = false;
let lastLoadedMtime = 0;

function loadModel() {
  if (!checkpointExists(config.checkpointDir)) {
    ready = false;
    console.log(`No checkpoint found in ${config.checkpointDir} yet.`);
    return false;
  }

  try {
    const candidateTokenizer = BPETokenizer.load(config.checkpointDir);
    const candidateConfig = { ...config, vocabSize: candidateTokenizer.vocabSize };
    const candidateModel = new TinyGPT(candidateConfig);
    loadCheckpointInto(candidateModel, config.checkpointDir);

    const oldModel = model;
    tokenizer = candidateTokenizer;
    model = candidateModel;
    config.vocabSize = candidateTokenizer.vocabSize;
    ready = true;
    if (oldModel) oldModel.dispose();

    try {
      lastLoadedMtime = fs.statSync(path.join(config.checkpointDir, 'weights.bin')).mtimeMs;
    } catch (_) {}

    console.log(`Model loaded (${(model.countParams() / 1e6).toFixed(2)}M params, vocab ${tokenizer.vocabSize}).`);
    return true;
  } catch (err) {
    console.error('Failed to load model:', err.message);
    ready = false;
    return false;
  }
}

function requireInferenceConfig() {
  const apiKey = getCredential('MODEL_API_KEY');
  if (!apiKey || apiKey === 'change-this-api-key') {
    console.warn('WARNING: MODEL_API_KEY is still using the default placeholder. Set it in .env before external use.');
  }
}

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

app.get('/health', (req, res) => res.type('text').send('ok'));

app.get('/api/branding', (req, res) => {
  res.json({
    modelName: process.env.MODEL_NAME || 'Tiny LLM',
    modelId: process.env.MODEL_ID || '',
    owner: process.env.MODEL_OWNER || '',
    brand: process.env.MODEL_BRAND || '',
  });
});

// ---------------- Admin authentication ----------------
app.post('/api/auth/login', (req, res) => login(req, res));
app.get('/api/auth/me', (req, res) => {
  const session = getSession(req);
  if (!session) return res.json({ authenticated: false });
  res.json({ authenticated: true, username: session.username, expiresAt: session.expiresAt });
});
app.post('/api/auth/logout', (req, res) => {
  destroySession(req);
  clearSessionCookie(res);
  res.json({ ok: true });
});

// ---------------- Protected dashboard status ----------------
app.get('/api/status', requireLogin, (req, res) => {
  res.json({
    ready,
    params: ready ? model.countParams() : 0,
    vocabSize: ready ? tokenizer.vocabSize : 0,
    modelName: process.env.MODEL_NAME || 'Tiny LLM',
    modelId: process.env.MODEL_ID || '',
  });
});

// ---------------- External inference API ----------------
function cleanHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter((t) => t && (t.role === 'user' || t.role === 'assistant') && typeof t.content === 'string')
    .slice(-16)
    .map((t) => ({ role: t.role, content: t.content.slice(0, 4000) }));
}

function buildGenerationOptions(body = {}) {
  const tempRaw = Number(body.temperature);
  const topKRaw = Number(body.topK);
  const repRaw = Number(body.repetitionPenalty);
  const freqRaw = Number(body.frequencyPenalty);
  const presRaw = Number(body.presencePenalty);
  const ngramRaw = Number(body.noRepeatNgramSize);
  const maxRaw = Number(body.maxNewTokens);

  return {
    temperature: Number.isFinite(tempRaw) ? Math.min(Math.max(tempRaw, 0), 2) : 0.8,
    topK: Number.isFinite(topKRaw) ? Math.min(Math.max(Math.round(topKRaw), 1), 200) : 40,
    repetitionPenalty: Number.isFinite(repRaw) ? Math.min(Math.max(repRaw, 1), 3) : 1.2,
    frequencyPenalty: Number.isFinite(freqRaw) ? Math.min(Math.max(freqRaw, 0), 3) : 0.25,
    presencePenalty: Number.isFinite(presRaw) ? Math.min(Math.max(presRaw, 0), 3) : 0.15,
    noRepeatNgramSize: Number.isFinite(ngramRaw) ? Math.min(Math.max(Math.round(ngramRaw), 0), 5) : 3,
    maxNewTokens: Number.isFinite(maxRaw) ? Math.min(Math.max(Math.round(maxRaw), 1), 256) : 128,
  };
}

function generateFromRequest(body) {
  if (!ready) throw new Error('Model not loaded. Train it first.');
  const message = typeof body?.message === 'string' ? body.message.trim() : '';
  if (!message) throw Object.assign(new Error('Missing "message" string in request body.'), { statusCode: 400 });
  if (message.length > 8000) throw Object.assign(new Error('Message is too long. Maximum length is 8000 characters.'), { statusCode: 400 });

  return generateReply(model, tokenizer, config, message, {
    ...buildGenerationOptions(body),
    history: cleanHistory(body.history),
  });
}

app.post('/api/chat', requireApiKey, (req, res) => {
  try {
    res.json({ reply: generateFromRequest(req.body || {}) });
  } catch (err) {
    console.error('[api/chat]', err);
    res.status(err.statusCode || 500).json({ error: err.message || 'Generation failed.' });
  }
});

// OpenAI-style compatibility endpoint for local apps that already speak the
// common chat-completions request/response shape.
app.get('/v1/models', requireApiKey, (req, res) => {
  res.json({
    object: 'list',
    data: [{
      id: process.env.MODEL_ID || 'tiny-llm',
      object: 'model',
      owned_by: process.env.MODEL_OWNER || 'local',
    }],
  });
});

app.post('/v1/chat/completions', requireApiKey, (req, res) => {
  try {
    const body = req.body || {};
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const lastUser = [...messages].reverse().find((m) => m && m.role === 'user' && typeof m.content === 'string');
    if (!lastUser) throw Object.assign(new Error('messages must contain at least one user message.'), { statusCode: 400 });

    const history = messages
      .slice(0, -1)
      .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .map((m) => ({ role: m.role, content: m.content }));

    const reply = generateFromRequest({
      ...body,
      message: lastUser.content,
      history,
      topK: body.top_k,
      maxNewTokens: body.max_tokens,
      repetitionPenalty: body.repetition_penalty,
    });

    res.json({
      id: `chatcmpl-${Date.now().toString(36)}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: process.env.MODEL_ID || 'tiny-llm',
      choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }],
    });
  } catch (err) {
    console.error('[v1/chat/completions]', err);
    res.status(err.statusCode || 500).json({ error: { message: err.message || 'Generation failed.' } });
  }
});

// ---------------- Training controls ----------------
let trainProc = null;
let trainState = 'idle';
let trainLogs = [];
const MAX_LOG_LINES = 500;
const MAX_OOM_RETRIES = 4;
let oomRetries = 0;
let overrideBatchSize = null;
let overrideContextLength = null;

function pushLog(chunk) {
  for (const line of chunk.toString().split('\n')) if (line.length) trainLogs.push(line);
  if (trainLogs.length > MAX_LOG_LINES) trainLogs = trainLogs.slice(-MAX_LOG_LINES);
}

function launchTraining() {
  trainState = 'running';
  const env = { ...process.env };
  if (overrideBatchSize !== null) env.BATCH_SIZE = String(overrideBatchSize);
  if (overrideContextLength !== null) env.CONTEXT_LENGTH = String(overrideContextLength);

  trainProc = spawn(process.execPath, [path.join(projectRoot, 'index.js'), 'train'], {
    cwd: projectRoot,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
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
        pushLog(`Host killed training. Auto-retrying with BATCH_SIZE ${prevBatch} -> ${overrideBatchSize} (attempt ${oomRetries}/${MAX_OOM_RETRIES}).`);
      } else if (prevCtx > 16) {
        overrideContextLength = Math.max(16, Math.floor(prevCtx / 2));
        pushLog(`Host killed training. Auto-retrying with CONTEXT_LENGTH ${prevCtx} -> ${overrideContextLength} (attempt ${oomRetries}/${MAX_OOM_RETRIES}).`);
      }
      launchTraining();
      return;
    }

    if (signal === 'SIGKILL') {
      trainState = 'error';
      pushLog('Training was killed by the host after automatic memory retries. Reduce N_EMBD/N_LAYER/FFN_HIDDEN or use a larger-RAM host.');
    } else if (signal) {
      trainState = 'stopped';
      pushLog(`Training stopped (signal ${signal}).`);
    } else if (code === 0) {
      trainState = 'done';
      oomRetries = 0;
      pushLog('Training finished successfully.');
    } else {
      trainState = 'error';
      pushLog(`Training process exited with code ${code}.`);
    }
    loadModel();
  });
  trainProc.on('error', (err) => {
    trainState = 'error';
    pushLog(`Failed to start training process: ${err.message}`);
    trainProc = null;
  });
}

app.post('/api/train/start', requireLogin, (req, res) => {
  if (trainProc) return res.status(409).json({ error: 'Training is already running.' });
  trainLogs = [];
  oomRetries = 0;
  overrideBatchSize = null;
  overrideContextLength = null;
  pushLog(`Starting training (data: ${config.dataDir}, checkpoint: ${config.checkpointDir})...`);
  launchTraining();
  res.json({ started: true });
});

app.post('/api/train/stop', requireLogin, (req, res) => {
  if (!trainProc) return res.status(409).json({ error: 'No training run in progress.' });
  trainProc.kill('SIGTERM');
  res.json({ stopping: true });
});

app.get('/api/train/status', requireLogin, (req, res) => {
  res.json({ state: trainState, running: !!trainProc, logs: trainLogs, modelReady: ready });
});

process.on('exit', () => {
  if (trainProc) trainProc.kill('SIGTERM');
});

loadModel();
requireInferenceConfig();

function checkForNewCheckpoint() {
  const weightsPath = path.join(config.checkpointDir, 'weights.bin');
  fs.stat(weightsPath, (err, stats) => {
    if (err || stats.mtimeMs <= lastLoadedMtime) return;
    console.log('Detected newer checkpoint; reloading model...');
    if (!loadModel()) console.warn('Checkpoint reload failed; keeping the previous model state unavailable until a valid checkpoint is present.');
  });
}
const checkpointPoller = setInterval(checkForNewCheckpoint, 10000);
checkpointPoller.unref();

process.on('uncaughtException', (err) => console.error('[uncaughtException]', err));
process.on('unhandledRejection', (reason) => console.error('[unhandledRejection]', reason));

const PORT = Number(process.env.PORT || 3000);
const HOST_RAW = process.env.HOST || process.env.IP || '0.0.0.0';
const looksLikeBindableAddress = /^(0\.0\.0\.0|127\.0\.0\.1|localhost|(\d{1,3}\.){3}\d{1,3})$/.test(HOST_RAW);
const HOST = looksLikeBindableAddress ? HOST_RAW : '0.0.0.0';

if (!looksLikeBindableAddress) console.warn(`HOST="${HOST_RAW}" is not a local bind address; using 0.0.0.0.`);
console.log('--- ZarKos server starting ---');
console.log(`Binding: ${HOST}:${PORT}`);

function startServer(port, host, retry = false) {
  const srv = app.listen(port, host, () => {
    console.log(`ZarKos server running on http://${host}:${port}`);
  });
  srv.on('error', (err) => {
    if (err.code === 'EADDRNOTAVAIL' && !retry) return startServer(port, '0.0.0.0', true);
    if (err.code === 'EADDRINUSE') console.error(`Port ${port} is already in use.`);
    else if (err.code === 'EACCES') console.error(`No permission to bind ${host}:${port}.`);
    else console.error('Server failed to start:', err);
    process.exit(1);
  });
}

startServer(PORT, HOST);
