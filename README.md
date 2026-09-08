# ZarKos Mini v1

A real, from-scratch decoder-only transformer (GPT-style), built with **pure `@tensorflow/tfjs` tensor ops** — not the Layers API. Every weight is a hand-defined `tf.Variable`, so this is an actual small language model, not a wrapper around a pre-built one.

By **TuZhi Codes** (TuZhi Studio).

---

## Features

- From-scratch decoder-only transformer (attention, feed-forward, layernorm — all raw ops)
- From-scratch BPE tokenizer, trained on your own data
- Multi-file `.jsonl` dataset loader with chat templating
- Checkpointing with automatic resume
- Terminal chat mode and a web UI (chat + live training control)
- OOM-safe training with automatic batch-size recovery
- Fully configurable via `.env` (model size, context length, training hyperparameters, display name)

## Setup

```bash
npm install
```

## 1. Add your data

Drop as many `.jsonl` files as you like into the `data/` folder. Each line is one conversation:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]}
```

An example file (`data/example.jsonl`) is included to show the format. Before real training, replace it with your own, larger dataset — more data generally means a better chatbot.

## 2. Train

```bash
npm run train
```

The first run will:

1. Train a BPE tokenizer from all `.jsonl` files and save it to `checkpoint/tokenizer.json`.
2. Train the model on windows built from your `data/` files.
3. Save `checkpoint/weights.bin` + `checkpoint/manifest.json` after every epoch.

Running `npm run train` again resumes from the existing checkpoint.

## 3. Chat

**In the terminal:**
```bash
npm run chat
```

**In the web UI:**
```bash
npm run serve
```
Then open `http://localhost:3000` in your browser. The UI has two tabs:
- **Chat** — talk to the model live to test it.
- **Train** — start/stop training from here, with live logs. Training runs in a separate background process (the server won't freeze), and a checkpoint is saved after every epoch — so it's safe to stop mid-run. Once training finishes or is stopped, the Chat tab automatically reloads the latest checkpoint.

## Configuration

All configuration lives in `.env`. Set `N_LAYER`, `N_EMBD`, `N_HEAD`, `CONTEXT_LENGTH`, `VOCAB_SIZE`, etc. to change the model architecture. Defaults land in the ~5–10M parameter range, which trains in a reasonable time on CPU.

`MODEL_NAME` in `.env` controls the display name shown in the web UI and server logs — set it to brand this as your own model.

⚠️ **Changing the architecture invalidates any existing checkpoint** (the tensor shapes will no longer match) — clear the `checkpoint/` folder and retrain from scratch.

## Folder structure

```
index.js            - CLI entry point: train / serve / chat
model/
  config.js          - hyperparameters, loaded from .env
  tokenizer.js        - from-scratch BPE tokenizer
  dataset.js          - multi-jsonl loader + chat template + windowing
  transformer.js       - the transformer itself: decoder-only, raw tfjs ops
  train.js            - training loop
  generate.js          - sampling / inference
  checkpoint.js        - binary weight save/load
checkpoint/           - tokenizer.json, manifest.json, weights.bin (created by training)
data/                 - your *.jsonl training files
ui/
  server.js            - Express API + static server + training start/stop endpoints
  public/main.html      - dark, mobile-app-style UI: Chat tab + Train tab (start/stop + live logs)
```

## Reliability notes

A few things this project handles so it stays usable on constrained hosting (e.g. small Discord-bot-style panels):

- **Path resolution is CWD-independent.** `.env` loading and the `data/`/`checkpoint/` directories are resolved relative to the project root rather than wherever the process happens to be launched from — so `node index.js`, `npm start`, a cron job, or a hosting panel's start command all behave the same.
- **A bare `node index.js` boots the web server.** `START_MODE` in `.env` controls what happens with no CLI arguments (default: `serve`), so process supervisors that expect a long-running server don't see an immediate clean exit as a crash.
- **Training survives out-of-memory kills.** Defaults are tuned to be memory-safe (`BATCH_SIZE=4`, `CONTEXT_LENGTH=64`), and if training is still killed by the OS, the server automatically retries with a smaller batch size (and context length if needed), up to 4 attempts — visible in the Train tab's logs. Once a checkpoint has been saved, auto-adjustment stops, so weight shapes always stay consistent with the checkpoint.
- **The server only serves what it should.** Only `ui/public/` is served statically, so server source code (like `server.js`) is never publicly reachable.
- **Crashes are visible, not silent.** `uncaughtException`/`unhandledRejection` are caught and logged instead of killing the process outright.
- **Startup diagnostics.** The server prints the `PORT`/`HOST` it's actually bound to, and exposes `GET /health` for hosting panels that poll for liveness.

### Hosting checklist

If it trains, serves, and chats fine locally but not on your host, check:

1. The `PORT`/`HOST` your hosting panel assigns match what's in `.env` — compare against the values printed at startup.
2. Don't upload `node_modules` — run `npm install` on the host itself.
3. Confirm your panel actually supports running a web server (some Discord-bot hosting panels only run bot processes and need a separate "website" feature enabled).
4. The start command should be `node index.js` or `npm start` — not `node index.js train`, which trains once and exits.
5. Check the panel's logs — crashes now print a clear error instead of failing silently.

## Realistic expectations

- Training a small model (~5–25M params) on a small dataset will teach you how a real transformer works and give you a genuine — but limited — chatbot.
- Output quality depends directly on your dataset's size and quality. `data/example.jsonl` only demonstrates the format; training on it alone won't produce a useful chatbot.
- Without a GPU, training will be slow — keep `CONTEXT_LENGTH`/`BATCH_SIZE` small.
