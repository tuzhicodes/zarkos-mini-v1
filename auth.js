const crypto = require('crypto');

const SESSION_COOKIE = 'zarkos_session';
const SESSION_TTL_MS = 12 * 60 * 60 * 1000;
const sessions = new Map();
const failedLogins = new Map();
const MAX_FAILED = 8;
const FAIL_WINDOW_MS = 10 * 60 * 1000;

function getCredential(name, fallback = '') {
  return String(process.env[name] ?? fallback).trim();
}

function safeEqual(a, b) {
  const aa = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (aa.length !== bb.length) return false;
  return crypto.timingSafeEqual(aa, bb);
}

function parseCookies(req) {
  const header = req.headers.cookie || '';
  const out = {};
  for (const part of header.split(';')) {
    const i = part.indexOf('=');
    if (i <= 0) continue;
    out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

function clientKey(req) {
  return req.socket.remoteAddress || 'unknown';
}

function canAttemptLogin(req) {
  const key = clientKey(req);
  const now = Date.now();
  const entry = failedLogins.get(key);
  if (!entry || now - entry.firstAt >= FAIL_WINDOW_MS) {
    failedLogins.set(key, { firstAt: now, count: 0 });
    return true;
  }
  return entry.count < MAX_FAILED;
}

function recordFailedLogin(req) {
  const key = clientKey(req);
  const now = Date.now();
  const entry = failedLogins.get(key);
  if (!entry || now - entry.firstAt >= FAIL_WINDOW_MS) {
    failedLogins.set(key, { firstAt: now, count: 1 });
  } else {
    entry.count += 1;
  }
}

function clearLoginFailures(req) {
  failedLogins.delete(clientKey(req));
}

function createSession(username) {
  const token = crypto.randomBytes(32).toString('hex');
  sessions.set(token, { username, expiresAt: Date.now() + SESSION_TTL_MS });
  return token;
}

function getSession(req) {
  const token = parseCookies(req)[SESSION_COOKIE];
  if (!token) return null;
  const session = sessions.get(token);
  if (!session) return null;
  if (session.expiresAt <= Date.now()) {
    sessions.delete(token);
    return null;
  }
  return { token, ...session };
}

function setSessionCookie(res, token) {
  const secure = process.env.NODE_ENV === 'production' ? '; Secure' : '';
  res.setHeader('Set-Cookie', `${SESSION_COOKIE}=${encodeURIComponent(token)}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}${secure}`);
}

function clearSessionCookie(res) {
  res.setHeader('Set-Cookie', `${SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0`);
}

function destroySession(req) {
  const session = getSession(req);
  if (session) sessions.delete(session.token);
}

function requireLogin(req, res, next) {
  if (!getSession(req)) return res.status(401).json({ error: 'Login required.' });
  next();
}

function isValidApiKey(req) {
  const expected = getCredential('MODEL_API_KEY');
  if (!expected || expected === 'change-this-api-key') return false;
  const fromHeader = req.get('x-api-key');
  const auth = req.get('authorization') || '';
  const bearer = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  return (fromHeader && safeEqual(fromHeader, expected)) || (bearer && safeEqual(bearer, expected));
}

function requireApiKey(req, res, next) {
  if (isValidApiKey(req)) return next();
  return res.status(401).json({ error: 'Valid API key required. Use x-api-key or Authorization: Bearer <key>.' });
}

function login(req, res) {
  if (!canAttemptLogin(req)) {
    return res.status(429).json({ error: 'Too many failed login attempts. Try again later.' });
  }
  const expectedUser = getCredential('ADMIN_USERNAME', 'admin');
  const expectedPass = getCredential('ADMIN_PASSWORD');
  if (!expectedPass) return res.status(500).json({ error: 'ADMIN_PASSWORD is not configured in .env.' });
  const username = typeof req.body?.username === 'string' ? req.body.username.trim() : '';
  const password = typeof req.body?.password === 'string' ? req.body.password : '';
  if (!safeEqual(username, expectedUser) || !safeEqual(password, expectedPass)) {
    recordFailedLogin(req);
    return res.status(401).json({ error: 'Invalid username or password.' });
  }
  clearLoginFailures(req);
  const token = createSession(username);
  setSessionCookie(res, token);
  res.json({ ok: true, username });
}

setInterval(() => {
  const now = Date.now();
  for (const [token, session] of sessions) {
    if (session.expiresAt <= now) sessions.delete(token);
  }
  for (const [key, entry] of failedLogins) {
    if (now - entry.firstAt >= FAIL_WINDOW_MS) failedLogins.delete(key);
  }
}, 60 * 60 * 1000).unref();

module.exports = {
  SESSION_COOKIE,
  login,
  getSession,
  setSessionCookie,
  clearSessionCookie,
  destroySession,
  requireLogin,
  requireApiKey,
  isValidApiKey,
  getCredential,
};
