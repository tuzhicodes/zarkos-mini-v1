/**
 * TinyGPT: a decoder-only (GPT-style) transformer implemented directly with
 * @tensorflow/tfjs tensor ops (no Layers API), so every weight is a
 * plain tf.Variable we fully control — this is what makes it a "real"
 * from-scratch LLM rather than a wrapper around a pre-built layer.
 *
 * Uses the pure-JS @tensorflow/tfjs package (CPU backend, no native addon)
 * instead of @tensorflow/tfjs-node on purpose: tfjs-node needs node-gyp to
 * compile a native binding on install, which shared/limited hosting often
 * blocks or can't run at all. Pure-JS is slower per-op but for a model this
 * small it's fine, and it removes the native-build failure point entirely.
 */

const tf = require('@tensorflow/tfjs');

function initWeight(shape, std = 0.02) {
  return tf.variable(tf.randomNormal(shape, 0, std));
}
function initZeros(shape) {
  return tf.variable(tf.zeros(shape));
}
function initOnes(shape) {
  return tf.variable(tf.ones(shape));
}

/**
 * tf.matMul requires both operands to have the SAME rank (>= 2) — it does
 * NOT auto-broadcast a 3D [batch, seq, inDim] activation against a 2D
 * [inDim, outDim] weight matrix the way numpy/PyTorch would. Calling
 * x.matMul(w2d) directly throws:
 *   "Error in matMul: inputs must have the same rank of at least 2, got ranks 3 and 2."
 *
 * This helper flattens [batch, seq, inDim] -> [batch*seq, inDim], does a
 * plain rank-2 matmul against the rank-2 weight, then reshapes back to
 * [batch, seq, outDim]. Bias add is fine as-is since tf.add broadcasts
 * across ranks without restriction (only matMul has the equal-rank rule).
 */
function linear(x, w, b) {
  const [batch, seq, inDim] = x.shape;
  const outDim = w.shape[1];
  const flat = x.reshape([batch * seq, inDim]);
  let out = flat.matMul(w).reshape([batch, seq, outDim]);
  if (b) out = out.add(b);
  return out;
}

class TinyGPT {
  constructor(config) {
    this.config = config;
    const { vocabSize, contextLength, nEmbd, nLayer, ffnHidden } = config;

    this.wte = initWeight([vocabSize, nEmbd]); // token embedding (also used as output head, weight-tied)
    this.wpe = initWeight([contextLength, nEmbd]); // positional embedding

    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      this.layers.push({
        ln1g: initOnes([nEmbd]),
        ln1b: initZeros([nEmbd]),
        wq: initWeight([nEmbd, nEmbd]),
        bq: initZeros([nEmbd]),
        wk: initWeight([nEmbd, nEmbd]),
        bk: initZeros([nEmbd]),
        wv: initWeight([nEmbd, nEmbd]),
        bv: initZeros([nEmbd]),
        wo: initWeight([nEmbd, nEmbd]),
        bo: initZeros([nEmbd]),
        ln2g: initOnes([nEmbd]),
        ln2b: initZeros([nEmbd]),
        wfc1: initWeight([nEmbd, ffnHidden]),
        bfc1: initZeros([ffnHidden]),
        wfc2: initWeight([ffnHidden, nEmbd]),
        bfc2: initZeros([nEmbd]),
      });
    }
    this.lnFg = initOnes([nEmbd]);
    this.lnFb = initZeros([nEmbd]);
  }

  /** Ordered list of {name, variable} — order MUST match between save and load. */
  namedVariables() {
    const list = [
      { name: 'wte', variable: this.wte },
      { name: 'wpe', variable: this.wpe },
    ];
    this.layers.forEach((l, i) => {
      for (const key of Object.keys(l)) {
        list.push({ name: `layer${i}.${key}`, variable: l[key] });
      }
    });
    list.push({ name: 'lnFg', variable: this.lnFg });
    list.push({ name: 'lnFb', variable: this.lnFb });
    return list;
  }

  get trainableVariables() {
    return this.namedVariables().map((v) => v.variable);
  }

  /**
   * Dispose every tf.Variable this model owns. tf.js tensors/variables are
   * NOT garbage-collected by the JS engine — they live in the tf backend's
   * own memory until explicitly disposed. Needed whenever a model instance
   * is being replaced (e.g. hot-reloading a newer checkpoint into a running
   * chat server) so the old weights don't just leak memory forever.
   */
  dispose() {
    for (const { variable } of this.namedVariables()) {
      variable.dispose();
    }
  }

  _layerNorm(x, gain, bias, eps = 1e-5) {
    return tf.tidy(() => {
      const mean = x.mean(-1, true);
      const variance = x.sub(mean).square().mean(-1, true);
      const normed = x.sub(mean).div(variance.add(eps).sqrt());
      return normed.mul(gain).add(bias);
    });
  }

  _causalMask(seqLen) {
    return tf.tidy(() => {
      // 1s below/on diagonal (allowed), 0s above (future, disallowed)
      const ones = tf.ones([seqLen, seqLen]);
      const mask = tf.linalg.bandPart(ones, -1, 0); // lower-triangular incl diagonal
      return mask.sub(1).mul(1e9); // 0 where allowed, -1e9 where disallowed
    });
  }

  _selfAttention(x, layer, mask) {
    const { nHead, headDim } = this.config;
    return tf.tidy(() => {
      const [batch, seqLen] = [x.shape[0], x.shape[1]];

      const q = linear(x, layer.wq, layer.bq);
      const k = linear(x, layer.wk, layer.bk);
      const v = linear(x, layer.wv, layer.bv);

      const splitHeads = (t) =>
        t.reshape([batch, seqLen, nHead, headDim]).transpose([0, 2, 1, 3]); // [B,H,S,D]

      const qh = splitHeads(q);
      const kh = splitHeads(k);
      const vh = splitHeads(v);

      const scores = tf.matMul(qh, kh, false, true).div(Math.sqrt(headDim)); // [B,H,S,S]
      const maskedScores = scores.add(mask); // broadcast [S,S] over [B,H,S,S]
      const probs = tf.softmax(maskedScores, -1);

      const attnOut = tf.matMul(probs, vh); // [B,H,S,D]
      const merged = attnOut.transpose([0, 2, 1, 3]).reshape([batch, seqLen, nHead * headDim]);

      return linear(merged, layer.wo, layer.bo);
    });
  }

  _feedForward(x, layer) {
    return tf.tidy(() => {
      const hidden = linear(x, layer.wfc1, layer.bfc1).relu();
      return linear(hidden, layer.wfc2, layer.bfc2);
    });
  }

  /** Forward pass. idx: int32 tensor [batch, seqLen]. Returns logits [batch, seqLen, vocabSize]. */
  forward(idx) {
    return tf.tidy(() => {
      const seqLen = idx.shape[1];
      const tokEmb = tf.gather(this.wte, idx); // [B,S,E]
      const posEmb = this.wpe.slice([0, 0], [seqLen, this.config.nEmbd]); // [S,E]
      let x = tokEmb.add(posEmb);

      const mask = this._causalMask(seqLen);

      for (const layer of this.layers) {
        const ln1 = this._layerNorm(x, layer.ln1g, layer.ln1b);
        x = x.add(this._selfAttention(ln1, layer, mask));
        const ln2 = this._layerNorm(x, layer.ln2g, layer.ln2b);
        x = x.add(this._feedForward(ln2, layer));
      }

      x = this._layerNorm(x, this.lnFg, this.lnFb);
      // Weight-tied output head: logits = x @ wte^T
      // Same rank-mismatch rule applies here: x is [B,S,E] (rank 3) and wte
      // is [vocabSize,E] (rank 2) — flatten to rank 2, matmul, reshape back.
      const [batch, seq, embd] = x.shape;
      const vocabSize = this.wte.shape[0];
      const flat = x.reshape([batch * seq, embd]);
      const logitsFlat = tf.matMul(flat, this.wte, false, true); // [B*S, vocabSize]
      return logitsFlat.reshape([batch, seq, vocabSize]);
    });
  }

  /** Count total trainable scalar parameters. */
  countParams() {
    return this.trainableVariables.reduce((sum, v) => sum + v.size, 0);
  }
}

module.exports = { TinyGPT };
