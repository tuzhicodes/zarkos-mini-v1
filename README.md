# ZarKos Mini v1

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org/)
[![TensorFlow.js](https://img.shields.io/badge/Engine-TensorFlow.js-orange.svg)](https://js.tensorflow.org/)
[![Parameters](https://img.shields.io/badge/Parameters-18.10M-purple.svg)](#technical-specifications)
[![Backend](https://img.shields.io/badge/Backend-CPU%20%7C%20CUDA%20GPU-darkblue.svg)](#hardware-and-device-modes)

**ZarKos Mini v1** is a self-contained, decoder-only Transformer language model built from scratch in Node.js using TensorFlow.js tensor primitives.

It is designed to run everywhere with zero native build hassles:
- **Terminal Chat**: Test responses directly on your command line.
- **Web Dashboard**: Interactive training monitor and status overview.
- **OpenAI-Compatible API**: Plug the model directly into existing AI apps, chat frontends, or automation scripts.
- **Multi-User Ready**: Built-in non-blocking token generation and serialized inference queue to prevent crashes under concurrent requests.

---

## Table of Contents
1. [Beginner Quickstart](#beginner-quickstart)
2. [Hosting & Deployment Guide](#hosting--deployment-guide)
3. [Connecting from External AI Apps](#connecting-from-external-ai-apps)
4. [CLI Commands](#cli-commands)
5. [Technical Specifications](#technical-specifications)
6. [Hardware and Device Modes](#hardware-and-device-modes)
7. [Training Your Own Data](#training-your-own-data)
8. [Project Structure](#project-structure)
9. [Author & License](#author--license)

---

## Beginner Quickstart

Follow these steps to get ZarKos Mini running on your local computer or phone (Termux) in under 2 minutes:

### Step 1: Install Dependencies
Make sure you have [Node.js](https://nodejs.org/) (v18 or higher) installed, then run:

```bash
git clone https://github.com/tuzhi/zarkos-mini-v1.git
cd zarkos-mini-v1
npm install
```

### Step 2: Set Up Configuration
Copy the sample environment file to create your local `.env`:

```bash
cp .env.example .env
```

Open `.env` in any text editor and adjust your credentials:
```env
PORT=3000
ADMIN_USERNAME=admin
ADMIN_PASSWORD=create-a-secure-password
MODEL_API_KEY=create-your-secret-key
```

### Step 3: Start Chatting

#### Option A: Chat in your Terminal
```bash
npm run chat
```
Type any prompt and watch the model stream tokens live in your terminal. Type `/exit` to quit.

#### Option B: Run the Web Dashboard and API
```bash
npm start
```
Open your browser and navigate to:
```
http://localhost:3000
```
Log in using the `ADMIN_USERNAME` and `ADMIN_PASSWORD` you configured in your `.env`.

---

## Hosting & Deployment Guide

ZarKos Mini v1 can be hosted on cloud providers like Render, Railway, Fly.io, or any Ubuntu/Debian VPS.

### 1. Crucial Checkpoint Files
Make sure your `checkpoint/` directory includes all 4 essential files:
- `manifest.json`: Model architecture and tensor configurations.
- `tokenizer.json`: BPE token mappings.
- `meta.json`: Step counts and training progress.
- `weights.bin`: The trained model weights (~70 MB binary file).

> **Important**: When uploading to hosting platforms (like Render or Railway), ensure `checkpoint/weights.bin` is included in your deployment so the server can load the model into memory.

### 2. Environment Variables on Hosting
Set these in your hosting provider's dashboard:

| Variable | Recommended Value | Notes |
| :--- | :--- | :--- |
| `PORT` | Set automatically or `3000` | Port assigned by host |
| `HOST` | `0.0.0.0` | Allows incoming public connections |
| `ADMIN_USERNAME` | `your-admin-user` | For web dashboard login |
| `ADMIN_PASSWORD` | `your-strong-password` | For web dashboard login |
| `MODEL_API_KEY` | `your-secret-api-key` | Required to access `/api/chat` and `/v1/chat/completions` |
| `API_CORS_ORIGIN` | `*` | Enables cross-origin browser requests |
| `DEVICE` | `cpu` | Use `gpu` only if you have NVIDIA CUDA |

### 3. Reverse Proxies & HTTPS
The server includes automatic proxy trust and HTTPS cookie detection (`x-forwarded-proto`), ensuring dashboard sessions and API routes work out of the box behind cloud load balancers.

---

## Connecting from External AI Apps

You can use ZarKos Mini v1 as a drop-in replacement for any service expecting an OpenAI-compatible API.

* **Base URL**: `http://<your-host>:<port>/v1`
* **API Key Header**: `Authorization: Bearer <your-MODEL_API_KEY>` (or `x-api-key: <your-MODEL_API_KEY>`)

### 1. cURL Example
```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "model": "tuzhi/zarkos-mini-v1",
    "messages": [
      {"role": "user", "content": "Hello! Tell me a short story."}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

### 2. Python (`openai` SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="your-secret-api-key"
)

response = client.chat.completions.create(
    model="tuzhi/zarkos-mini-v1",
    messages=[{"role": "user", "content": "What is 5 + 7?"}],
    max_tokens=30,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 3. JavaScript / Node.js
```javascript
const response = await fetch("http://localhost:3000/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-api-key"
  },
  body: JSON.stringify({
    model: "tuzhi/zarkos-mini-v1",
    messages: [{ role: "user", content: "Explain how solar energy works." }],
    max_tokens: 60
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### 4. Direct Chat Endpoint (`/api/chat`)
For simple integrations without the full OpenAI schema:
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{"message": "Hello!", "maxNewTokens": 30}'
```

---

## CLI Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `npm start` | **Serve** | Starts the web UI dashboard and API server |
| `npm run chat` | **Chat** | Starts real-time interactive terminal chat with live streaming |
| `node index.js eval` | **Eval** | Runs fixed prompts with deterministic greedy decoding |
| `npm run train` | **Train** | Starts or resumes training on `data/*.jsonl` |
| `node verify_device.js` | **Verify** | Confirms active TensorFlow backend and hardware device |

---

## Technical Specifications

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **Model Type** | Decoder-only Transformer | Autoregressive language modeling |
| **Total Parameters** | **18,103,136** (~18.10M) | Trainable variables |
| **Number of Layers** | 10 | Sequential Transformer blocks |
| **Attention Heads** | 8 | Dimension per head: `352 / 8 = 44` |
| **Embedding Dimension** | 352 | Hidden representation width |
| **Feed-Forward Dimension** | 1408 | 4× expansion factor with GELU |
| **Context Length** | 56 tokens | Bounded sliding window |
| **Vocabulary Size** | 9,000 tokens | Custom Byte-Pair Encoding (BPE) |
| **Weight Tying** | Enabled | Input embedding (`wte`) shared with output projection |
| **Weights Buffer** | ~70 MB | Contiguous binary float buffer (`weights.bin`) |

---

## Hardware and Device Modes

Configure `DEVICE` in your `.env` file:

```env
# Pure JS CPU execution (Termux, macOS, generic cloud hosting, VPS)
DEVICE=cpu

# Native NVIDIA CUDA acceleration (Linux x86_64 server with CUDA installed)
DEVICE=gpu
```

- **CPU Mode**: Runs out of the box on Android (Termux), Apple Silicon, and cheap VPS servers without compiling C++ binaries.
- **GPU Mode**: Uses `@tensorflow/tfjs-node-gpu` for high-throughput training on desktop/server NVIDIA cards.
- **Checkpoint Compatibility**: Checkpoints are binary-identical across CPU and GPU backends.

---

## Training Your Own Data

### 1. Add Data
Place your JSONL files into the `data/` directory. Each line must be a JSON object containing a `messages` array:

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help?"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2 + 2 = 4."}]}
```

### 2. Run Training
```bash
npm run train
```
The trainer splits 5% for validation and automatically saves checkpoints to `checkpoint/` every 500 steps.

---

## Project Structure

```
├── .env.example         # Template for environment configuration
├── index.js              # Primary CLI entry point
├── auth.js               # Session auth and API key verification
├── package.json          # Node dependencies and scripts
├── verify_device.js      # Device diagnostics script
├── checkpoint/           # Model weights, tokenizer, and manifest
│   ├── manifest.json
│   ├── meta.json
│   ├── tokenizer.json
│   └── weights.bin
├── data/                 # Training dataset files (.jsonl)
├── model/
│   ├── checkpoint.js     # Binary weight reader and writer
│   ├── config.js         # Hyperparameters and settings
│   ├── dataset.js        # Data windowing and batching
│   ├── generate.js       # Asynchronous token generation engine
│   ├── tf.js             # Device backend loader
│   ├── tokenizer.js      # Byte-Pair Encoding tokenizer
│   ├── train.js          # Optimization loop and validation
│   └── transformer.js    # Self-attention and Transformer implementation
└── ui/
    ├── server.js         # API server, dashboard backend, and inference queue
    └── public/           # Dashboard static web files
```

---

## Author & License

* **Author:** TuZhi Codes ([@tuzhicodes](https://github.com/tuzhicodes))
* **License:** [MIT License](LICENSE)
