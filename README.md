# ZarKos Mini v1

**ZarKos Mini v1** is a small, from-scratch decoder-only Transformer language model by **TuZhi Codes**, developed under **TuZhi Studio**.

It is implemented in Node.js with `@tensorflow/tfjs` using raw tensor operations rather than a high-level model wrapper. The repository contains the model architecture, tokenizer, dataset pipeline, training loop, checkpointing, inference code, terminal interface, and a lightweight web UI.

> **Project status:** This is a small, experimental language model. It is a real trainable neural language model, but it is not intended to compete with large production LLMs. Output quality depends heavily on the training data, model configuration, and amount of training.

---

## Model information

| Field | Value |
|---|---|
| **Name** | ZarKos Mini v1 |
| **Model ID** | `tuzhi/zarkos-mini-v1` |
| **Owner** | TuZhi Codes |
| **Studio** | TuZhi Studio |
| **Architecture** | Decoder-only Transformer |
| **Framework** | Node.js + `@tensorflow/tfjs` |
| **Tokenizer** | From-scratch BPE tokenizer |
| **Training format** | JSONL conversations |
| **License** | MIT |

The model ID is a project identifier used by this repository. It does not imply that the model is automatically published to a model hub or hosted inference service.

---

## What it can be used for

ZarKos Mini v1 is suitable for:

- experimenting with small language models
- learning how decoder-only Transformers work
- training a custom chatbot on your own conversation data
- testing tokenizer and dataset designs
- running lightweight local inference
- experimenting with sampling parameters
- comparing checkpoints during training
- building small Node.js applications around a custom model
- learning about checkpoint-based training and model evaluation

It can also be adapted for other small language-model experiments by changing the configuration and dataset.

### What it is not

ZarKos Mini v1 should **not** be presented as a general-purpose frontier LLM. A small model has limited capacity and context, and it can produce incorrect, repetitive, nonsensical, or unsafe text.

Do not rely on its output for medical, legal, financial, security, or other high-stakes decisions.

---

## Features

### Model

- From-scratch decoder-only Transformer
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Hand-defined TensorFlow.js variables
- Configurable embedding size, layers, attention heads, context length, and FFN size

### Tokenizer

- From-scratch BPE tokenizer
- Vocabulary trained from the project's JSONL data
- Special chat tokens:
  - `<pad>`
  - `<unk>`
  - `<bos>`
  - `<eos>`
  - `<|user|>`
  - `<|assistant|>`
  - `<|end|>`

### Dataset and training

- Reads every `*.jsonl` file from the configured data directory
- Conversation-based training format
- Train/validation split before token windowing
- Per-conversation training windows
- 50% overlapping windows for longer conversations
- Padding for short conversations
- Configurable batch size, epochs, learning rate, context length, and validation fraction
- Checkpoint saving during training and at epoch completion
- Resume support from an existing checkpoint

### Inference

- Terminal chat mode
- Web chat interface
- Conversation history support in the web UI/API
- Temperature sampling
- Top-k sampling
- Configurable maximum generated tokens
- Automatic stop at the trained end-of-turn token

### Evaluation

The repository includes a fixed-prompt evaluation command:

```bash
npm run eval
```

It uses deterministic greedy-style decoding (`temperature=0`, `topK=1`) so outputs from different checkpoints can be compared more consistently.

### Web UI and server

- Lightweight Express server
- Mobile-friendly chat UI
- Training control from the web UI
- Live training logs
- Start/stop training controls
- Model status endpoint
- Hosting-panel-friendly `/health` endpoint
- Automatic checkpoint hot-reload
- Public static files limited to `ui/public/`
- Basic request validation and visible server errors

### Resource-conscious behavior

The project includes safeguards intended for smaller hosting environments:

- configurable model size
- configurable batch/context size
- automatic batch/context reduction after certain pre-checkpoint `SIGKILL`/OOM situations in the web training process
- checkpointing during long epochs
- model disposal during hot reload to reduce TensorFlow.js tensor leaks
- project-root-relative paths so launching from another working directory does not silently use the wrong data/checkpoint folders

These are safeguards, not a guarantee that every hosting plan can train the model.

---

# Quick start

## Requirements

You need:

- Node.js
- npm
- enough RAM for the model/configuration you choose
- a writable project directory

Install dependencies:

```bash
npm install
```

---

## Add training data

Put one or more `.jsonl` files inside `data/`.

Each line should contain one conversation object:

```json
{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello!"}]}
```

Multiple conversation files are supported.

For example:

```text
data/
├── chat.jsonl
├── knowledge.jsonl
└── examples.jsonl
```

The loader reads all files ending in `.jsonl` from the configured data directory.

### Data quality matters

The model learns statistical patterns from the data you provide. Clean, diverse, correctly formatted data is generally more useful than simply adding large amounts of noisy or duplicated data.

Do not put passwords, API keys, private conversations, personal secrets, or other sensitive information into a public training dataset.

---

# Configuration with `.env`

ZarKos Mini v1 reads configuration from a `.env` file in the project root.

Create:

```text
.env
```

Example configuration:

```env
# -----------------------------
# Model identity
# -----------------------------
MODEL_NAME=ZarKos Mini v1
MODEL_ID=tuzhi/zarkos-mini-v1
MODEL_OWNER=TuZhi Codes
MODEL_BRAND=TuZhi Studio

# -----------------------------
# Server
# -----------------------------
START_MODE=serve
HOST=0.0.0.0
PORT=3000

# -----------------------------
# Data / checkpoints
# -----------------------------
DATA_DIR=data
CHECKPOINT_DIR=checkpoint

# -----------------------------
# Tokenizer
# -----------------------------
VOCAB_SIZE=8000

# -----------------------------
# Model architecture
# -----------------------------
CONTEXT_LENGTH=128
N_EMBD=128
N_HEAD=4
N_LAYER=4
FFN_HIDDEN=512

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE=16
EPOCHS=5
LEARNING_RATE=0.0003
VAL_FRACTION=0.05

# Save a checkpoint every N steps
CHECKPOINT_EVERY=500
```

These are configuration examples based on the values currently defined by the repository's `model/config.js`. They are not a claim about the size or quality of every trained ZarKos Mini v1 checkpoint.

### Important

Some architecture settings determine tensor shapes. If you change:

```text
N_EMBD
N_HEAD
N_LAYER
FFN_HIDDEN
CONTEXT_LENGTH
```

an existing checkpoint may no longer be compatible.

For a changed architecture, remove the old `checkpoint/` directory and train a new checkpoint.

Also make sure:

```text
N_EMBD % N_HEAD == 0
```

because the embedding dimension must divide evenly across attention heads.

### Secrets

Do not place real API keys, passwords, tokens, or other secrets in `.env` if the file will be committed to a public repository.

If you later add secrets to the project, keep them outside source control and add `.env` to `.gitignore`.

---

# Train the model

Run:

```bash
npm run train
```

The training process:

1. loads the JSONL conversations
2. trains/loads the project tokenizer as required
3. splits conversations into training and validation sets
4. builds token windows
5. performs next-token prediction training
6. periodically saves checkpoints
7. saves the completed checkpoint for later inference

The checkpoint directory contains generated model artifacts such as:

```text
checkpoint/
├── tokenizer.json
├── manifest.json
└── weights.bin
```

Do not manually edit generated checkpoint files unless you know the expected format.

---

# Chat with ZarKos Mini v1

## Terminal

```bash
npm run chat
```

The terminal chat loads the latest available checkpoint.

If there is no checkpoint yet, train the model first:

```bash
npm run train
```

## Web UI

Start the server:

```bash
npm run serve
```

Then open:

```text
http://localhost:3000
```

The UI provides:

- **Chat** — interact with the loaded model
- **Train** — start/stop training and inspect live logs

The server can also be started with:

```bash
npm start
```

With the default configuration, a bare:

```bash
node index.js
```

starts the web server.

---

# Evaluation

Run the fixed evaluation prompts:

```bash
npm run eval
```

This is useful when comparing checkpoints.

It does **not** provide a standardized benchmark score. It is simply a small, repeatable project-level check intended to make checkpoint-to-checkpoint comparisons easier.

For meaningful model evaluation, use a larger held-out test set and task-specific metrics rather than relying only on a few example prompts.

---

# HTTP API

The web server exposes a small API.

## Health

```http
GET /health
```

Expected response:

```text
ok
```

## Model status

```http
GET /api/status
```

Returns information such as whether a checkpoint is loaded and the currently loaded parameter/vocabulary counts.

## Branding

```http
GET /api/branding
```

Returns the configured model identity.

## Chat

```http
POST /api/chat
Content-Type: application/json
```

Example:

```json
{
  "message": "Hello",
  "history": []
}
```

Optional generation fields include:

```json
{
  "message": "Tell me a short joke",
  "temperature": 0.8,
  "maxNewTokens": 100,
  "history": [
    {
      "role": "user",
      "content": "Hi"
    },
    {
      "role": "assistant",
      "content": "Hello!"
    }
  ]
}
```

The API validates the message and supported history roles before generation.

---

# Project structure

```text
zarkos-mini-v1/
├── data/
│   └── example.jsonl
│
├── model/
│   ├── checkpoint.js
│   ├── config.js
│   ├── dataset.js
│   ├── generate.js
│   ├── tokenizer.js
│   ├── train.js
│   └── transformer.js
│
├── ui/
│   ├── public/
│   │   └── main.html
│   └── server.js
│
├── checkpoint/          # generated after training
├── index.js
├── package.json
├── package-lock.json
├── LICENSE
└── README.md
```

### Main components

| File | Purpose |
|---|---|
| `index.js` | CLI entry point for train/serve/chat/eval |
| `model/transformer.js` | Transformer model implementation |
| `model/tokenizer.js` | BPE tokenizer |
| `model/dataset.js` | JSONL loading, chat formatting, splitting, and windows |
| `model/train.js` | Training loop |
| `model/generate.js` | Text generation and sampling |
| `model/checkpoint.js` | Checkpoint save/load |
| `model/config.js` | Environment-based configuration |
| `ui/server.js` | Express server and model/training API |
| `ui/public/main.html` | Web interface |

---

# Training data format

The expected format is:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"2 + 2 = 4."}]}
```

Supported conversation roles are primarily:

```text
user
assistant
```

Other roles are handled as plain context by the current dataset formatter and should be used deliberately.

Keep each JSONL record valid JSON and keep the conversation structure consistent.

Malformed JSONL lines are skipped with a warning rather than being silently treated as valid training examples.

---

# Checkpoints and resuming

Training creates a checkpoint that can be loaded for inference.

The training system also saves checkpoints during long runs according to:

```env
CHECKPOINT_EVERY=500
```

If training is stopped after a checkpoint has been written, the project can use the available checkpoint when training is run again.

Changing architecture-related settings after a checkpoint exists can make the old weights incompatible. Treat architecture changes as a new training run.

---

# Responsible use

ZarKos Mini v1 is experimental software and an experimental language model.

The model can:

- hallucinate information
- repeat training patterns
- generate incorrect statements
- produce low-quality or incomplete answers
- behave differently depending on sampling settings
- fail on tasks outside its training distribution

The repository does not claim that the model is factual, safe, unbiased, or suitable for high-stakes use.

If you deploy the model publicly, add the application-level controls appropriate for your environment, such as authentication, rate limiting, request-size limits, logging/privacy controls, and abuse protection.

Never expose private training data or server secrets through a public deployment.

---

# Development notes

This project is intentionally small and understandable rather than optimized for frontier-scale performance.

Its main purpose is to provide a practical, inspectable implementation of:

```text
JSONL data
   ↓
conversation formatting
   ↓
BPE tokenizer
   ↓
token windows
   ↓
decoder-only Transformer
   ↓
next-token training
   ↓
checkpoint
   ↓
text generation
```

The implementation can be modified for experiments with architecture, datasets, training schedules, tokenization, inference, and deployment.

---

# Limitations

ZarKos Mini v1 has deliberately modest resource requirements compared with large language models, but that also limits its capabilities.

Performance depends on:

- parameter count
- training data quality and size
- number of training steps/epochs
- context length
- tokenizer vocabulary
- optimization settings
- available compute
- generation parameters

A lower training loss does not automatically mean that a model is more useful for every task. Always evaluate the actual behavior you care about.

---

# License

ZarKos Mini v1 is released under the MIT License.

Copyright © 2026 TuZhi Codes.

See [`LICENSE`](LICENSE) for the full license text.

---

# Credits

**ZarKos Mini v1**  
Model ID: `tuzhi/zarkos-mini-v1`

Created by **TuZhi Codes**  
**TuZhi Studio**
