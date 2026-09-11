# ZarKos Mini v1

A real, from-scratch decoder-only Transformer language model, trained and served entirely in Node.js. It does not call, wrap, or proxy any external LLM — the tokenizer, the training loop, and the model itself are all implemented here.

| | |
|---|---|
| **Model** | ZarKos Mini v1 |
| **Model ID** | `tuzhi/zarkos-mini-v1` |
| **Owner** | TuZhi Codes |
| **Built by** | TuZhi Studio |
| **Repository** | https://github.com/tuzhicodes/zarkos-mini-v1 |
| **License** | MIT |

This project is open source. Fork it, fine-tune it on your own data, rebrand it, or deploy it as-is — no dataset or checkpoint from the original build is included, so you start from a clean slate.

## What's in here

- **Training** — a small BPE tokenizer and a decoder-only Transformer, trained on your own JSONL conversation data.
- **Web dashboard** — a password-protected page to start/stop training and watch live logs. See [`ui/README.md`](ui/README.md) for details.
- **Inference API** — an API-key-protected chat endpoint, plus an OpenAI-compatible `/v1/chat/completions` endpoint so it can be dropped into existing chat clients.

## Requirements

- Node.js 18+
- npm

Install dependencies:

```bash
npm install
```

## 1. Set up your environment

Copy the example file and edit it:

```bash
cp .env.example .env
```

At minimum, change these before running anything publicly:

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this-password
MODEL_API_KEY=change-this-api-key
```

`ADMIN_USERNAME` / `ADMIN_PASSWORD` control who can log into the training dashboard. `MODEL_API_KEY` controls who can call the inference API. Never commit a real `.env` file — it's already excluded via `.gitignore`.

The model's name, ID, owner, and brand are also read from `.env`, so you can rebrand the whole app without touching any code:

```env
MODEL_NAME=ZarKos Mini v1
MODEL_ID=tuzhi/zarkos-mini-v1
MODEL_OWNER=TuZhi Codes
MODEL_BRAND=TuZhi Studio
```

## 2. Add your training data

Drop your own conversation datasets into `data/` as `.jsonl` files. Every line should look like this:

```json
{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"Hi!"}]}
```

Every `.jsonl` file in `data/` is loaded automatically. The folder ships empty — this repo does not include any training data or pretrained checkpoint, so the model knows nothing until you train it.

Before building the tokenizer and training windows, the dataset is split into training and held-out validation conversations, so validation loss reflects data the model has never seen.

## 3. Train the model

From the project root, either run training directly:

```bash
npm run train
```

or start the server and use the dashboard:

```bash
npm start
```

Then open the server address in a browser, log in with your `.env` credentials, and click **Start Training**. The dashboard only handles authentication and training controls — it has no chat interface.

## 4. Talk to the model

**Terminal chat**, once you have a trained checkpoint:

```bash
npm run chat
```

**Inference API**, protected by `MODEL_API_KEY`:

```bash
curl -X POST http://localhost:24705/api/chat \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: your-api-key' \
  -d '{"message":"hello","maxNewTokens":64}'
```

An `Authorization: Bearer your-api-key` header works the same way.

**OpenAI-compatible endpoint**, for clients that expect that shape:

```bash
curl http://localhost:24705/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-api-key' \
  -d '{"model":"tuzhi/zarkos-mini-v1","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

## How the model works

- Learned token embeddings, tied to the output projection
- Learned positional embeddings
- Causal multi-head self-attention with pre-layer normalization
- GELU feed-forward blocks
- Next-token cross-entropy training
- Temperature / top-k sampling with repetition control at inference
- A small custom BPE tokenizer trained from your own dataset

The architecture defaults to roughly **6.8M trainable parameters** (see `model/config.js`), but this scales with the size of your training data and how long you train. The exact parameter count is printed when the server starts. This is a genuine, working neural network — not a wrapper around a larger model — but its quality depends entirely on your dataset size and quality, the tokenizer, and training time. It should not be expected to match large pretrained foundation models.

## Changing the architecture

If you change any of `N_EMBD`, `N_HEAD`, `N_LAYER`, `FFN_HIDDEN`, `VOCAB_SIZE`, or `CONTEXT_LENGTH` in `.env`, any existing checkpoint becomes incompatible. Delete or move the `checkpoint/` folder and train again from scratch.

## API endpoints

| Endpoint | Auth |
|---|---|
| `/api/auth/login`, `/api/auth/logout`, `/api/auth/me` | Dashboard session cookie |
| `/api/training/*` (start/stop/status) | Dashboard session cookie |
| `/api/chat` | `MODEL_API_KEY` |
| `/v1/models`, `/v1/chat/completions` | `MODEL_API_KEY` |
| `/health` | Public (for hosting/uptime checks) |

The web page never renders the API key or admin password back to the browser.

## CLI commands

```bash
npm start          # start the web server (uses START_MODE from .env)
npm run serve      # start the web server directly
npm run train      # train or resume training
npm run chat       # chat with the current checkpoint in the terminal
npm run eval       # run a fixed set of prompts to compare checkpoints
```

## Hosting

This app is built to run anywhere Node.js runs — a VPS, a container, or a managed Node hosting panel. It has no native dependencies (TensorFlow.js runs on the pure JS CPU backend), so there's nothing extra to compile at deploy time.

A few things that make deployment easier:

- The server reads `.env` from the project's own folder, so it works even if your host launches the app from a different working directory.
- If no start command/argument is given, the app defaults to `serve` mode, so a plain `node index.js` boots the web server instead of exiting.
- The dashboard is installable as a PWA (see [`ui/README.md`](ui/README.md)) — useful if you want to check on training from your phone.

## Before going public

1. Set a strong, unique `ADMIN_PASSWORD`.
2. Set a strong, random `MODEL_API_KEY`.
3. Restrict `API_CORS_ORIGIN` to your actual domain if you don't need open browser access.
4. Watch validation loss, not just training loss, to judge whether the model is actually learning.
5. Keep `.env` out of version control (it already is, via `.gitignore`).

## Fine-tuning / forking this project

Everything needed to retrain from zero is in `model/`. To build your own version:

1. Fork the repo.
2. Replace the contents of `data/` with your own JSONL conversations.
3. Adjust the architecture settings in `.env` to fit your hardware and dataset size.
4. Run `npm run train`.
5. Update `MODEL_NAME` / `MODEL_ID` / `MODEL_OWNER` / `MODEL_BRAND` in `.env` to rebrand it as your own.

No code changes are required for any of the above.

## License

MIT — see the `package.json` for details. Use it, modify it, ship it.
