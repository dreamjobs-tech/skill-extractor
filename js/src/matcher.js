/**
 * Keyword matcher — a faithful port of FlashText's extract_keywords.
 * Word boundaries: any char outside [A-Za-z0-9_]. Case-insensitive,
 * longest-match-wins, non-overlapping. Identical spans across all
 * skill-extractor language implementations.
 */

const KEYWORD = Symbol('keyword');

function isWordChar(ch) {
  const c = ch.charCodeAt(0);
  return (
    (c >= 48 && c <= 57) || // 0-9
    (c >= 65 && c <= 90) || // A-Z
    (c >= 97 && c <= 122) || // a-z
    c === 95 // _
  );
}

export class KeywordMatcher {
  constructor(keywords = []) {
    this.trie = new Map();
    for (const kw of keywords) this.add(kw);
  }

  add(keyword) {
    let node = this.trie;
    const kw = keyword.toLowerCase();
    for (let i = 0; i < kw.length; i++) {
      const ch = kw[i]; // code units, matching extract()'s indexing
      if (!node.has(ch)) node.set(ch, new Map());
      node = node.get(ch);
    }
    node.set(KEYWORD, keyword.toLowerCase());
  }

  /** @returns {Array<[string, number, number]>} [keyword, start, end] spans */
  extract(sentence) {
    const out = [];
    if (!sentence) return out;
    sentence = sentence.toLowerCase();
    let current = this.trie;
    let seqStart = 0;
    let seqEnd = 0;
    let reset = false;
    let idx = 0;
    const n = sentence.length;
    while (idx < n) {
      const ch = sentence[idx];
      if (!isWordChar(ch)) {
        if (current.has(KEYWORD) || current.has(ch)) {
          let longest = null;
          let longerFound = false;
          if (current.has(KEYWORD)) {
            longest = current.get(KEYWORD);
            seqEnd = idx;
          }
          if (current.has(ch)) {
            let cont = current.get(ch);
            let idy = idx + 1;
            let broke = false;
            while (idy < n) {
              const inner = sentence[idy];
              if (!isWordChar(inner) && cont.has(KEYWORD)) {
                longest = cont.get(KEYWORD);
                seqEnd = idy;
                longerFound = true;
              }
              if (cont.has(inner)) {
                cont = cont.get(inner);
              } else {
                broke = true;
                break;
              }
              idy += 1;
            }
            if (!broke && cont.has(KEYWORD)) {
              longest = cont.get(KEYWORD);
              seqEnd = idy;
              longerFound = true;
            }
            if (longerFound) idx = seqEnd;
          }
          current = this.trie;
          if (longest) out.push([longest, seqStart, idx]);
          reset = true;
        } else {
          current = this.trie;
          reset = true;
        }
      } else if (current.has(ch)) {
        current = current.get(ch);
      } else {
        current = this.trie;
        reset = true;
        let idy = idx + 1;
        while (idy < n) {
          if (!isWordChar(sentence[idy])) break;
          idy += 1;
        }
        idx = idy;
      }
      if (idx + 1 >= n && current.has(KEYWORD)) {
        out.push([current.get(KEYWORD), seqStart, n]);
      }
      idx += 1;
      if (reset) {
        reset = false;
        seqStart = idx;
      }
    }
    return out;
  }
}
