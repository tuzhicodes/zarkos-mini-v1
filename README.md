# ZarKos Mini v1

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org/)
[![TensorFlow.js](https://img.shields.io/badge/Engine-TensorFlow.js-orange.svg)](https://js.tensorflow.org/)
[![Parameters](https://img.shields.io/badge/Parameters-18.10M-purple.svg)](#architecture-specifications)
[![Backend](https://img.shields.io/badge/Backend-CPU%20%7C%20CUDA%20GPU-darkblue.svg)](#compute-backend)

**ZarKos Mini v1** is a decoder-only Transformer language model built from scratch in Node.js using TensorFlow.js tensor primitives.

Most JavaScript machine learning projects are either API wrappers or Python child processes. ZarKos Mini v1 is a ground-up neural network implementation written in JavaScript: multi-head causal self-attention, layer normalization, BPE tokenization, loss masking, gradient updates, and token sampling run natively in Node.js.

The repository includes pre-trained weights in `checkpoint/` so you can clone and test the model immediately without training from scratch.

---

## Transformer Architecture

```
Input Tokens: [ <|user|> ... <|end|><|assistant|> ]
       │
       ▼
┌──────────────────────────────────────────────┐
│  Token Embedding (wte: 9000 × 352)           │
│  + Positional Embedding (wpe: 56 × 352)      │
└──────────────────────────────────────────────┘
       │
       ▼  [Residual Stream: 352-dim]
┌──────────────────────────────────────────────┐ ──┐
│  1. Pre-Layer Normalization (ln1)            │   │
│  2. Causal Multi-Head Attention (8 heads)    │   │
│     - Scaled Dot-Product: Q × K^T / √d_k     │   │  Repeated
│     - Lower-triangular Causal Attention Mask │   │  across
│     - Head projection & Output Linear (wo)   │   │  10 Layers
│  3. Residual Addition (x + attn)             │   │  (nLayer = 10)
│  4. Pre-Layer Normalization (ln2)            │   │
│  5. Feed-Forward Network (FFN: 352 → 1408)   │   │
│     - GELU Activation Function               │   │
│     - Linear Projection (1408 → 352)         │   │
│  6. Residual Addition (x + ffn)              │   │
└──────────────────────────────────────────────┘ ──┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Final Layer Normalization (lnF)             │
│  Weight-Tied Output Projection (wte^T)       │
└──────────────────────────────────────────────┘
       │
       ▼
Logits (Batch × SeqLen × 9000) ──► Sampling Engine (Top-K / Temp / Repetition Penalty)
```

---

## Technical Specifications

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **Model Type** | Decoder-only Transformer | Autoregressive language modeling |
| **Total Parameters** | **18,103,136** (~18.10M) | All trainable variables (`tf.Variable`) |
| **Number of Layers** | 10 | Sequential Transformer blocks |
| **Attention Heads** | 8 | Dimension per head: `352 / 8 = 44` |
| **Embedding Dimension** | 352 | Hidden representation width |
| **Feed-Forward Dimension** | 1408 | 4× expansion factor |
| **Context Length** | 56 tokens | Bounded sliding window |
| **Vocabulary Size** | 9,000 tokens | Custom Byte-Pair Encoding (BPE) |
| **Weight Tying** | Enabled | Input embedding (`wte`) shared with LM head |
| **Activation** | GELU | Gaussian Error Linear Unit |
| **Normalization** | Pre-LN | Pre-Layer Normalization for stable gradients |
| **Inference Weight Size** | ~70 MB | Float32 raw binary buffer (`weights.bin`) |

---

## Quickstart

### Prerequisites
* [Node.js](https://nodejs.org/) v18.0.0 or higher
* npm or pnpm

### 1. Clone & Setup
```bash
git clone https://github.com/tuzhicodes/zarkos-mini-v1.git
cd zarkos-mini-v1
npm install
cp .env.example .env
```

### 2. Run Interactive Terminal Chat
```bash
npm run chat
```
The model loads `checkpoint/` into memory and begins streaming responses with live token-by-token output.

### 3. Start Web UI & API Server
```bash
npm start
```
* **Dashboard URL:** `http://localhost:24705`
* **Default Username:** `admin`
* **Default Password:** `change-this-password` *(configured in `.env`)*

---

## CLI Reference

| Command | Action | Description |
| :--- | :--- | :--- |
| `npm start` | **Serve** | Starts the Express web server and OpenAI API |
| `npm run chat` | **Chat** | Real-time terminal chat with streaming text |
| `node index.js eval` | **Eval** | Runs fixed greedy-decoding prompts to benchmark checkpoints |
| `npm run train` | **Train** | Starts or resumes training on `data/*.jsonl` |
| `node verify_device.js` | **Verify** | Tests TensorFlow backend and GPU/CPU tensor execution |

---

## Inference API

ZarKos Mini v1 provides an OpenAI-compatible interface alongside a lightweight JSON endpoint.

All API requests require authentication via the `MODEL_API_KEY` environment variable.

### OpenAI Compatibility Endpoint (`/v1/chat/completions`)

#### cURL
```bash
curl -X POST http://localhost:24705/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-this-api-key" \
  -d '{
    "model": "tuzhi/zarkos-mini-v1",
    "messages": [
      {"role": "user", "content": "What is 5 + 7?"}
    ],
    "temperature": 0.7,
    "max_tokens": 40
  }'
```

#### Python (`openai` client)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:24705/v1",
    api_key="change-this-api-key"
)

response = client.chat.completions.create(
    model="tuzhi/zarkos-mini-v1",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=30
)

print(response.choices[0].message.content)
```

#### Node.js (`fetch`)
```javascript
const res = await fetch('http://localhost:24705/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer change-this-api-key'
  },
  body: JSON.stringify({
    model: 'tuzhi/zarkos-mini-v1',
    messages: [{ role: 'user', content: 'What is photosynthesis?' }]
  })
});
const data = await res.json();
console.log(data.choices[0].message.content);
```

---

## Generation & Anti-Repetition Engine

Small language models are structurally vulnerable to repeating n-grams or falling into token loops. ZarKos Mini v1 implements a three-tier defense pipeline in `model/generate.js`:

1. **HuggingFace-style N-Gram Blocking (`noRepeatNgramSize = 3`)**: Continuously monitors generated tokens. If a candidate token would complete an already seen 3-gram, its logit is clamped to `-Infinity`.
2. **Multiplicative Repetition Penalty**: CTRL-style penalty dividing positive logits and multiplying negative logits for tokens present in context.
3. **Frequency & Presence Penalty**: OpenAI-style linear penalty scaling directly with token occurrence count.
4. **Token Streaming**: Real-time asynchronous token decoding (`decodeToken()`) without memory buffering.

---

## Training Pipeline

### Dataset Format
Conversations are read from `data/*.jsonl`. Every entry is a JSON object with a standard message history:

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]}
{"messages": [{"role": "user", "content": "What is 5 + 7?"}, {"role": "assistant", "content": "5 + 7 equals 12."}]}
```

The loader automatically splits 5% of conversations into a held-out validation set.

### Tokenizer
A pure JavaScript Byte-Pair Encoding (BPE) tokenizer merges character pairs greedily until reaching the target vocabulary (`VOCAB_SIZE=9000`). Merges and vocabulary are serialized to `tokenizer.json`.

---

## Compute Backend

Hardware acceleration is configured via `DEVICE` in `.env`:

```env
# Pure JS CPU execution (Termux, macOS, generic Linux servers)
DEVICE=cpu

# Native NVIDIA CUDA acceleration (Linux x86_64 server with CUDA/cuDNN)
DEVICE=gpu
```

* **GPU Performance (Tesla T4 / RTX 3060+)**: Runs at ~15–20 steps/second with `@tensorflow/tfjs-node-gpu`.
* **CPU Portability**: Runs with zero native dependencies on Android (Termux), Apple Silicon, and lightweight VPS instances.
* **Format Portability**: Checkpoint files (`weights.bin`, `manifest.json`) are binary-identical across platforms. A model trained on a GPU cluster can be directly executed on a mobile phone without conversion.

---

## Checkpoint Format

Checkpoints are stored in `checkpoint/`:

```
checkpoint/
├── manifest.json      # Tensor shapes, dtypes, and byte offsets
├── tokenizer.json     # BPE vocabulary and merge priority table
├── meta.json          # Training epoch, global step, and validation loss
└── weights.bin        # Contiguous Float32 buffer of model weights (~70 MB)
```

*(Note: `optimizer.bin` contains Adam momentum accumulators for resuming training; it is excluded from git commits to keep the repository clone small.)*

---

## Directory Layout

```
├── .env.example         # Environment template
├── .gitignore            # Git exclusion rules
├── package.json          # Node.js project configuration
├── index.js              # Command-line entry point (train, serve, chat, eval)
├── verify_device.js      # Hardware verification script
├── add_tokens.js         # Checkpoint vocabulary expansion utility
├── auth.js               # Web UI session authentication
├── test.js               # CLI test client
├── checkpoint/           # Model weights and BPE dictionary
├── data/                 # Training JSONL files (example.jsonl included)
├── model/
│   ├── banner.js         # Terminal diagnostics & banners
│   ├── checkpoint.js     # Binary weight reader and writer
│   ├── config.js         # Hyperparameter validator
│   ├── dataset.js        # Windowing, padding, and loss masking
│   ├── generate.js       # Sampling and generation engine
│   ├── tf.js             # Device backend dispatcher
│   ├── tokenizer.js      # Byte-Pair Encoding implementation
│   ├── train.js          # Optimization loop and validation
│   └── transformer.js    # Multi-head attention & Transformer core
└── ui/
    ├── server.js         # HTTP server, dashboard, and OpenAI API
    └── public/           # PWA assets (HTML, CSS, service worker)
```

---

## Author & License

* **Author:** TuZhi Codes ([@tuzhicodes](https://github.com/tuzhicodes))
* **License:** [MIT License](LICENSE)
