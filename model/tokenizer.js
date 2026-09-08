/**
 * A small, dependency-free Byte-Pair Encoding (BPE) tokenizer.
 *
 * This is a simplified version of the algorithm used by GPT-2/GPT-3:
 *   1. Split raw text into words.
 *   2. Represent each word as a sequence of characters + an end-of-word marker.
 *   3. Repeatedly merge the most frequent adjacent pair of symbols into a new
 *      symbol, until we reach the target vocab size.
 *   4. Encoding text = greedily applying the learned merges.
 */

const fs = require('fs');
const path = require('path');

const EOW = '</w>'; // end-of-word marker

function wordToSymbols(word) {
  return word.split('').concat([EOW]);
}

function getPairCounts(corpusWords) {
  const counts = new Map();
  for (const { symbols, freq } of corpusWords) {
    for (let i = 0; i < symbols.length - 1; i++) {
      const pair = symbols[i] + '\u0000' + symbols[i + 1];
      counts.set(pair, (counts.get(pair) || 0) + freq);
    }
  }
  return counts;
}

function mergePairInWord(symbols, a, b) {
  const merged = [];
  let i = 0;
  while (i < symbols.length) {
    if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
      merged.push(a + b);
      i += 2;
    } else {
      merged.push(symbols[i]);
      i += 1;
    }
  }
  return merged;
}

class BPETokenizer {
  constructor(specialTokens = []) {
    this.specialTokens = specialTokens;
    this.merges = []; // ordered list of [a, b] pairs, highest priority first
    this.mergeRank = new Map(); // "a\0b" -> rank (lower = applied first)
    this.tokenToId = new Map();
    this.idToToken = [];
  }

  _addToken(tok) {
    if (!this.tokenToId.has(tok)) {
      this.tokenToId.set(tok, this.idToToken.length);
      this.idToToken.push(tok);
    }
  }

  /** Train BPE merges from an array of raw text strings. */
  train(texts, targetVocabSize) {
    // 1. Pre-tokenize into words (simple whitespace + punctuation split, Unicode-aware)
    const wordFreq = new Map();
    const wordRe = /[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu;
    for (const text of texts) {
      const words = text.match(wordRe) || [];
      for (const w of words) {
        wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      }
    }

    let corpusWords = Array.from(wordFreq.entries()).map(([word, freq]) => ({
      symbols: wordToSymbols(word),
      freq,
    }));

    // 2. Base vocab = all unique symbols (characters) seen + special tokens
    for (const st of this.specialTokens) this._addToken(st);
    const baseSymbols = new Set();
    for (const { symbols } of corpusWords) symbols.forEach((s) => baseSymbols.add(s));
    for (const s of baseSymbols) this._addToken(s);

    // 3. Iteratively merge most frequent pair until vocab is full
    const maxMerges = Math.max(0, targetVocabSize - this.idToToken.length);
    for (let m = 0; m < maxMerges; m++) {
      const pairCounts = getPairCounts(corpusWords);
      if (pairCounts.size === 0) break;

      let bestPair = null;
      let bestCount = -1;
      for (const [pair, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestPair = pair;
        }
      }
      if (bestCount < 2) break; // not worth merging singletons

      const [a, b] = bestPair.split('\u0000');
      this.merges.push([a, b]);
      this.mergeRank.set(bestPair, this.merges.length - 1);
      this._addToken(a + b);

      corpusWords = corpusWords.map(({ symbols, freq }) => ({
        symbols: mergePairInWord(symbols, a, b),
        freq,
      }));
    }

    return this;
  }

  /** Apply learned merges to a single word (array of chars + EOW). */
  _bpeWord(word) {
    let symbols = wordToSymbols(word);
    if (symbols.length === 1) return symbols;

    while (true) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + '\u0000' + symbols[i + 1];
        const rank = this.mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      symbols = mergePairInWord(symbols, symbols[bestIdx], symbols[bestIdx + 1]);
    }
    return symbols;
  }

  /** Encode raw text into an array of token ids. */
  encode(text) {
    const wordRe = /[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu;
    const unkId = this.tokenToId.get('<unk>');
    const ids = [];

    const pushWord = (w) => {
      const symbols = this._bpeWord(w);
      for (const s of symbols) {
        ids.push(this.tokenToId.has(s) ? this.tokenToId.get(s) : unkId);
      }
    };

    // Special tokens (e.g. <|user|>, <|end|>, <|assistant|>) must be
    // matched as a single atomic token BEFORE the generic word-level regex
    // runs. Without this, wordRe splits "<|user|>" into "<", "|", "user",
    // "|", ">" — five broken fragments instead of the one real token id —
    // even though the special token has its own dedicated vocab entry.
    // That mismatch is why generated replies were leaking raw fragments
    // like "< | user | >" instead of ever seeing/producing a clean
    // <|user|>/<|end|> boundary.
    const specials = this.specialTokens
      .filter((t) => this.tokenToId.has(t))
      .sort((a, b) => b.length - a.length); // longest first, avoids partial-overlap matches

    if (specials.length === 0) {
      const words = text.match(wordRe) || [];
      for (const w of words) pushWord(w);
      return ids;
    }

    const specialRe = new RegExp(
      specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
      'g'
    );

    let lastIndex = 0;
    let match;
    while ((match = specialRe.exec(text)) !== null) {
      const chunk = text.slice(lastIndex, match.index);
      const words = chunk.match(wordRe) || [];
      for (const w of words) pushWord(w);
      ids.push(this.tokenToId.get(match[0]));
      lastIndex = match.index + match[0].length;
    }
    const rest = text.slice(lastIndex).match(wordRe) || [];
    for (const w of rest) pushWord(w);

    return ids;
  }

  /** Decode an array of token ids back into text. */
  decode(ids) {
    let out = '';
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined || this.specialTokens.includes(tok)) continue;
      out += tok.endsWith(EOW) ? tok.slice(0, -EOW.length) + ' ' : tok;
    }
    return out.trim();
  }

  get vocabSize() {
    return this.idToToken.length;
  }

  tokenId(tok) {
    return this.tokenToId.get(tok);
  }

  save(dir) {
    const file = path.join(dir, 'tokenizer.json');
    fs.writeFileSync(
      file,
      JSON.stringify({
        specialTokens: this.specialTokens,
        merges: this.merges,
        idToToken: this.idToToken,
      })
    );
    return file;
  }

  static load(dir) {
    const file = path.join(dir, 'tokenizer.json');
    const raw = JSON.parse(fs.readFileSync(file, 'utf-8'));
    const tok = new BPETokenizer(raw.specialTokens);
    tok.merges = raw.merges;
    tok.idToToken = raw.idToToken;
    tok.tokenToId = new Map(raw.idToToken.map((t, i) => [t, i]));
    raw.merges.forEach(([a, b], i) => tok.mergeRank.set(a + '\u0000' + b, i));
    return tok;
  }
}

module.exports = { BPETokenizer, EOW };
