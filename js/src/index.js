/**
 * skill-extractor: extract skills from job postings and resumes.
 * 32K-skill gazetteer + MiniLM embeddings + MLP context classifier,
 * trained on 491K labeled samples (73% F1 held-out).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { KeywordMatcher } from './matcher.js';
import { MLP } from './mlp.js';

const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'data');
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const CONTEXT_BEFORE = 20;
const CONTEXT_AFTER = 21;
const DEFAULT_THRESHOLD = 0.5;

/** Python str.split() semantics: split on whitespace runs, drop empties. */
function pySplit(s) {
  const parts = s.split(/\s+/u);
  if (parts.length && parts[0] === '') parts.shift();
  if (parts.length && parts[parts.length - 1] === '') parts.pop();
  return parts;
}

export class SkillExtractor {
  /** @param {{quantized?: boolean}} [opts] fp32 (default) matches the reference bit-for-bit */
  constructor(opts = {}) {
    this.quantized = opts.quantized ?? false;
    this.matcher = new KeywordMatcher(
      JSON.parse(readFileSync(join(DATA_DIR, 'skills.json'), 'utf8'))
    );
    this.mlp = new MLP(
      JSON.parse(readFileSync(join(DATA_DIR, 'mlp.json'), 'utf8'))
    );
    this._pipe = null;
  }

  async _embedder() {
    if (!this._pipe) {
      const { pipeline } = await import('@huggingface/transformers');
      this._pipe = await pipeline('feature-extraction', MODEL, {
        dtype: this.quantized ? 'q8' : 'fp32',
      });
    }
    return this._pipe;
  }

  /** Gazetteer matches with their +/-20-word context windows. */
  candidates(text) {
    const matches = this.matcher.extract(text);
    if (!matches.length) return [];
    const words = pySplit(text);
    return matches.map(([skill, start]) => {
      const wordIdx = pySplit(text.slice(0, start)).length;
      const lo = Math.max(0, wordIdx - CONTEXT_BEFORE);
      const hi = Math.min(words.length, wordIdx + CONTEXT_AFTER);
      return { skill, context: words.slice(lo, hi).join(' ') };
    });
  }

  /** @returns {Promise<number[]>} P(skill) for each candidate input */
  async classify(inputs) {
    const pipe = await this._embedder();
    const out = await pipe(inputs, { pooling: 'mean', normalize: true });
    const [n, dim] = out.dims;
    const flat = out.data;
    const embs = [];
    for (let i = 0; i < n; i++) {
      embs.push(new Float32Array(flat.buffer, flat.byteOffset + i * dim * 4, dim));
    }
    return this.mlp.predictProba(embs);
  }

  /** Extract confirmed skills from a job posting or resume text. */
  async extract(text, threshold = DEFAULT_THRESHOLD) {
    const cands = this.candidates(text);
    if (!cands.length) return [];
    const probs = await this.classify(
      cands.map((c) => `${c.skill} : ${c.context}`)
    );
    const found = new Set();
    probs.forEach((p, i) => {
      if (p >= threshold) found.add(cands[i].skill);
    });
    return [...found].sort();
  }
}

let _default = null;

/** Module-level convenience wrapper around a shared SkillExtractor. */
export async function extractSkills(text, threshold = DEFAULT_THRESHOLD) {
  if (!_default) _default = new SkillExtractor();
  return _default.extract(text, threshold);
}

export { KeywordMatcher, MLP };
