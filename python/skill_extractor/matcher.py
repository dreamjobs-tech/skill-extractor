"""Keyword matcher — a faithful port of FlashText's extract_keywords.

Word boundaries are any characters outside [A-Za-z0-9_]. Matching is
case-insensitive, longest-match-wins, non-overlapping. Ported 1:1 from
flashtext.KeywordProcessor so every language implementation of
skill-extractor produces identical spans.
"""

_WORD_CHARS = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
)
_KEYWORD = object()  # trie leaf sentinel


class KeywordMatcher:
    def __init__(self, keywords=None):
        self._trie = {}
        for kw in keywords or []:
            self.add(kw)

    def add(self, keyword: str) -> None:
        node = self._trie
        for ch in keyword.lower():
            node = node.setdefault(ch, {})
        node[_KEYWORD] = keyword.lower()

    def extract(self, sentence: str) -> list[tuple[str, int, int]]:
        """Return (keyword, start, end) spans, matching FlashText exactly."""
        out = []
        if not sentence:
            return out
        sentence = sentence.lower()
        current = self._trie
        seq_start = 0
        seq_end = 0
        reset = False
        idx = 0
        n = len(sentence)
        while idx < n:
            ch = sentence[idx]
            if ch not in _WORD_CHARS:
                if _KEYWORD in current or ch in current:
                    longest = None
                    longer_found = False
                    if _KEYWORD in current:
                        longest = current[_KEYWORD]
                        seq_end = idx
                    if ch in current:
                        cont = current[ch]
                        idy = idx + 1
                        broke = False
                        while idy < n:
                            inner = sentence[idy]
                            if inner not in _WORD_CHARS and _KEYWORD in cont:
                                longest = cont[_KEYWORD]
                                seq_end = idy
                                longer_found = True
                            if inner in cont:
                                cont = cont[inner]
                            else:
                                broke = True
                                break
                            idy += 1
                        if not broke:  # reached end of sentence
                            if _KEYWORD in cont:
                                longest = cont[_KEYWORD]
                                seq_end = idy
                                longer_found = True
                        if longer_found:
                            idx = seq_end
                    current = self._trie
                    if longest:
                        out.append((longest, seq_start, idx))
                    reset = True
                else:
                    current = self._trie
                    reset = True
            elif ch in current:
                current = current[ch]
            else:
                current = self._trie
                reset = True
                idy = idx + 1
                while idy < n:
                    if sentence[idy] not in _WORD_CHARS:
                        break
                    idy += 1
                idx = idy
            if idx + 1 >= n:
                if _KEYWORD in current:
                    out.append((current[_KEYWORD], seq_start, n))
            idx += 1
            if reset:
                reset = False
                seq_start = idx
        return out
