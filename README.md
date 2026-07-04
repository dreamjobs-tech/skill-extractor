# skill-extractor

**Extract skills from job postings and resumes — in Python, JavaScript, Ruby, or Rust, with identical output.**

Most open-source skill extractors are either bare gazetteer/regex matchers
(every mention of "go" becomes the Go language) or thin wrappers over an LLM
call. This one is a trained pipeline:

1. **Gazetteer** — 31,836 curated skill names propose candidate spans
   (FlashText-style longest-match, so `.net (c#)` and `c++` work).
2. **Context windows** — ±20 words around each hit.
3. **MiniLM embeddings** — `all-MiniLM-L6-v2` via ONNX Runtime (no torch).
4. **MLP context classifier** — accepts or rejects each candidate *in
   context*. Trained on **491K labeled samples**, **73% F1** on held-out job
   postings.

So "we value a can-do attitude and drive impact" produces **zero** skills,
while "5+ years Python, REST APIs with FastAPI" produces `python`,
`rest apis`, `fastapi`.

This is the same pipeline that powers [Qarera](https://www.qarera.com)'s job
analysis — including the [Most In-Demand Skills of 2026](https://www.qarera.com/reports/most-in-demand-skills-2026)
study of 360,000+ job postings.

## Install

| Language | Registry | Install |
|---|---|---|
| Python ≥3.10 | PyPI | `pip install skill-extractor` |
| JavaScript (Node ≥18) | npm | `npm install skill-extractor` |
| Ruby ≥3.0 | RubyGems | `gem install skill-extractor` |
| Rust | crates.io | `skill-extractor = "0.1"` |

Each package bundles the gazetteer + classifier weights (~2MB) and downloads
the MiniLM ONNX model (~90MB fp32, or ~23MB quantized) from the Hugging Face
Hub on first use.

## Use

```python
from skill_extractor import extract_skills          # Python
extract_skills("5+ years Python, Docker required")  # ['docker', 'python']
```

```js
import { extractSkills } from 'skill-extractor';    // JavaScript
await extractSkills('5+ years Python, Docker required');
```

```ruby
require "skill_extractor"                            # Ruby
SkillExtractor.extract_skills("5+ years Python, Docker required")
```

```rust
use skill_extractor::SkillExtractor;                 // Rust
SkillExtractor::new(false)?.extract("5+ years Python, Docker required", 0.5)?;
```

Every implementation also exposes `candidates(text)` (raw gazetteer hits +
context windows), a tunable threshold (default 0.5), and a `quantized`
option (~4x faster, near-identical accuracy).

## Identical across languages — by test, not by promise

All four implementations run the same parity suite in `fixtures/`:

- **300 fuzz cases / 3,704 spans** — the gazetteer matcher must reproduce the
  reference spans exactly (it's a 1:1 port of FlashText's algorithm).
- **8 end-to-end cases** — candidates, context windows, classifier inputs,
  probabilities (tolerance 2e-3), and final skill sets must match the Python
  reference, which is itself verified against the production model
  (bit-exact MLP, same ONNX weights).

## Repo layout

```
python/   PyPI package  (onnxruntime + tokenizers)
js/       npm package   (@huggingface/transformers)
ruby/     RubyGems gem  (informers)
rust/     crates.io crate (ort + tokenizers)
fixtures/ shared parity fixtures — regenerate from python/
artifacts/ canonical exported weights + gazetteer
```

## Notes

- Gazetteer skill names come from freely redistributable sources: O*NET
  (CC BY 4.0), Wikidata software/technology names (CC0), curated public
  lists, and vocabulary observed in Qarera's own job-postings corpus —
  minus a prose-noise denylist. It is a slightly reduced set of the
  production gazetteer (long-tail proprietary-taxonomy entries removed);
  the classifier and the F1 evaluation pipeline are identical.
- English job postings/resumes. Hard skills ("python", "aws") match in any
  language's text, but the classifier context model is English-trained.
- Output is lowercase canonical skill names from the gazetteer.
- The classifier measures *is this really a skill requirement in context* —
  it does not deduplicate synonyms ("react.js" vs "react").

MIT. Built by [Qarera](https://www.qarera.com) — free AI job-search tools
(resume builder, ATS checker, job matching).
