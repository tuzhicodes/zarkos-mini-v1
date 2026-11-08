const tf = require('./tf');

function initWeight(shape, name, std = 0.02) {
  return tf.variable(tf.randomNormal(shape, 0, std), true, name);
}
function initZeros(shape, name) {
  return tf.variable(tf.zeros(shape), true, name);
}
function initOnes(shape, name) {
  return tf.variable(tf.ones(shape), true, name);
}

let nextInstanceId = 0;

function linear(x, w, b) {
  const [batch, seq, inDim] = x.shape;
  const outDim = w.shape[1];
  const flat = x.reshape([batch * seq, inDim]);
  let out = flat.matMul(w).reshape([batch, seq, outDim]);
  if (b) out = out.add(b);
  return out;
}

class ZarKosLM {
  constructor(config) {
    this.config = config;
    const { vocabSize, contextLength, nEmbd, nLayer, ffnHidden } = config;

    const uid = nextInstanceId++;
    const w = (shape, name, std) => initWeight(shape, `${name}#${uid}`, std);
    const z = (shape, name) => initZeros(shape, `${name}#${uid}`);
    const o = (shape, name) => initOnes(shape, `${name}#${uid}`);

    this.wte = w([vocabSize, nEmbd], 'wte');
    this.wpe = w([contextLength, nEmbd], 'wpe');

    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      this.layers.push({
        ln1g: o([nEmbd], `layer${i}.ln1g`),
        ln1b: z([nEmbd], `layer${i}.ln1b`),
        wq: w([nEmbd, nEmbd], `layer${i}.wq`),
        bq: z([nEmbd], `layer${i}.bq`),
        wk: w([nEmbd, nEmbd], `layer${i}.wk`),
        bk: z([nEmbd], `layer${i}.bk`),
        wv: w([nEmbd, nEmbd], `layer${i}.wv`),
        bv: z([nEmbd], `layer${i}.bv`),
        wo: w([nEmbd, nEmbd], `layer${i}.wo`),
        bo: z([nEmbd], `layer${i}.bo`),
        ln2g: o([nEmbd], `layer${i}.ln2g`),
        ln2b: z([nEmbd], `layer${i}.ln2b`),
        wfc1: w([nEmbd, ffnHidden], `layer${i}.wfc1`),
        bfc1: z([ffnHidden], `layer${i}.bfc1`),
        wfc2: w([ffnHidden, nEmbd], `layer${i}.wfc2`),
        bfc2: z([nEmbd], `layer${i}.bfc2`),
      });
    }
    this.lnFg = o([nEmbd], 'lnFg');
    this.lnFb = z([nEmbd], 'lnFb');
  }

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
      const rowIdx = tf.range(0, seqLen, 1, 'int32').reshape([seqLen, 1]);
      const colIdx = tf.range(0, seqLen, 1, 'int32').reshape([1, seqLen]);
      const allowed = colIdx.lessEqual(rowIdx);
      return allowed.logicalNot().cast('float32').mul(-1e9);
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
        t.reshape([batch, seqLen, nHead, headDim]).transpose([0, 2, 1, 3]);

      const qh = splitHeads(q);
      const kh = splitHeads(k);
      const vh = splitHeads(v);

      const scores = tf.matMul(qh, kh, false, true).div(Math.sqrt(headDim));
      const maskedScores = scores.add(mask);
      const probs = tf.softmax(maskedScores, -1);

      const attnOut = tf.matMul(probs, vh);
      const merged = attnOut.transpose([0, 2, 1, 3]).reshape([batch, seqLen, nHead * headDim]);

      return linear(merged, layer.wo, layer.bo);
    });
  }

  _feedForward(x, layer) {
    return tf.tidy(() => {

      const h = linear(x, layer.wfc1, layer.bfc1);
      const gelu = h.mul(0.5).mul(tf.add(1, tf.erf(h.div(Math.sqrt(2)))));
      return linear(gelu, layer.wfc2, layer.bfc2);
    });
  }

  forward(idx) {
    return tf.tidy(() => {
      const seqLen = idx.shape[1];
      const tokEmb = tf.gather(this.wte, idx);
      const posEmb = this.wpe.slice([0, 0], [seqLen, this.config.nEmbd]);
      let x = tokEmb.add(posEmb);

      const mask = this._causalMask(seqLen);

      for (const layer of this.layers) {
        const ln1 = this._layerNorm(x, layer.ln1g, layer.ln1b);
        x = x.add(this._selfAttention(ln1, layer, mask));
        const ln2 = this._layerNorm(x, layer.ln2g, layer.ln2b);
        x = x.add(this._feedForward(ln2, layer));
      }

      x = this._layerNorm(x, this.lnFg, this.lnFb);

      const [batch, seq, embd] = x.shape;
      const vocabSize = this.wte.shape[0];
      const flat = x.reshape([batch * seq, embd]);
      const logitsFlat = tf.matMul(flat, this.wte, false, true);
      return logitsFlat.reshape([batch, seq, vocabSize]);
    });
  }

  countParams() {
    return this.trainableVariables.reduce((sum, v) => sum + v.size, 0);
  }
}

module.exports = { ZarKosLM };
