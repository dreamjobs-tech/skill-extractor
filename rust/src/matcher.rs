//! Keyword matcher — a faithful port of FlashText's `extract_keywords`.
//! Word boundaries: any char outside `[A-Za-z0-9_]`. Case-insensitive,
//! longest-match-wins, non-overlapping. Identical spans across all
//! skill-extractor language implementations (indices are code points,
//! like Python's).

use std::collections::HashMap;

#[derive(Default)]
struct Node {
    children: HashMap<char, Node>,
    keyword: Option<String>,
}

#[inline]
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[derive(Default)]
pub struct KeywordMatcher {
    root: Node,
}

impl KeywordMatcher {
    pub fn new<I: IntoIterator<Item = S>, S: AsRef<str>>(keywords: I) -> Self {
        let mut m = Self::default();
        for kw in keywords {
            m.add(kw.as_ref());
        }
        m
    }

    pub fn add(&mut self, keyword: &str) {
        let kw = keyword.to_lowercase();
        let mut node = &mut self.root;
        for ch in kw.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.keyword = Some(kw);
    }

    /// Returns `(keyword, start, end)` spans, matching FlashText exactly.
    pub fn extract(&self, sentence: &str) -> Vec<(String, usize, usize)> {
        let mut out = Vec::new();
        if sentence.is_empty() {
            return out;
        }
        let chars: Vec<char> = sentence.to_lowercase().chars().collect();
        let n = chars.len();
        let mut current = &self.root;
        let mut seq_start = 0usize;
        let mut seq_end = 0usize;
        let mut reset = false;
        let mut idx = 0usize;
        while idx < n {
            let ch = chars[idx];
            if !is_word_char(ch) {
                let has_kw = current.keyword.is_some();
                let has_ch = current.children.contains_key(&ch);
                if has_kw || has_ch {
                    let mut longest: Option<&String> = None;
                    let mut longer_found = false;
                    if let Some(kw) = &current.keyword {
                        longest = Some(kw);
                        seq_end = idx;
                    }
                    if let Some(mut cont) = current.children.get(&ch) {
                        let mut idy = idx + 1;
                        let mut broke = false;
                        while idy < n {
                            let inner = chars[idy];
                            if !is_word_char(inner) {
                                if let Some(kw) = &cont.keyword {
                                    longest = Some(kw);
                                    seq_end = idy;
                                    longer_found = true;
                                }
                            }
                            match cont.children.get(&inner) {
                                Some(next) => cont = next,
                                None => {
                                    broke = true;
                                    break;
                                }
                            }
                            idy += 1;
                        }
                        if !broke {
                            if let Some(kw) = &cont.keyword {
                                longest = Some(kw);
                                seq_end = idy;
                                longer_found = true;
                            }
                        }
                        if longer_found {
                            idx = seq_end;
                        }
                    }
                    if let Some(kw) = longest {
                        out.push((kw.clone(), seq_start, idx));
                    }
                    current = &self.root;
                    reset = true;
                } else {
                    current = &self.root;
                    reset = true;
                }
            } else if let Some(next) = current.children.get(&ch) {
                current = next;
            } else {
                current = &self.root;
                reset = true;
                let mut idy = idx + 1;
                while idy < n {
                    if !is_word_char(chars[idy]) {
                        break;
                    }
                    idy += 1;
                }
                idx = idy;
            }
            if idx + 1 >= n {
                if let Some(kw) = &current.keyword {
                    out.push((kw.clone(), seq_start, n));
                }
            }
            idx += 1;
            if reset {
                reset = false;
                seq_start = idx;
            }
        }
        out
    }
}
