# skill-extractor (Rust)

Extract skills from job postings and resumes. A 32,000-skill gazetteer proposes
candidates; a MiniLM + MLP context classifier accepts or rejects each one, so
"we value a can-do attitude" doesn't become a skill. Trained on 491K labeled
samples, 73% F1 on held-out job postings.

Runs on [ort](https://github.com/pykeio/ort) (ONNX Runtime) +
[tokenizers](https://github.com/huggingface/tokenizers). The gazetteer and
classifier weights are embedded in the crate; the MiniLM model downloads from
the Hugging Face Hub on first use and is cached.

```toml
[dependencies]
skill-extractor = "0.1"
```

```rust
use skill_extractor::SkillExtractor;

let mut ex = SkillExtractor::new(false)?; // true = quantized (23MB, ~4x faster)

let text = "Senior Backend Engineer. 5+ years Python or Go, REST APIs with \
            FastAPI, PostgreSQL, AWS (ECS, Lambda). Docker and Kubernetes required.";

let skills = ex.extract(text, 0.5)?;
// ["aws", "docker", "ecs", "fastapi", "go", "kubernetes", "lambda",
//  "postgresql", "python", "rest apis"]
```

`ex.candidates(text)` exposes the raw gazetteer hits with their ±20-word
context windows; `ex.classify(&inputs)` returns per-candidate probabilities.

macOS note: linking needs a macOS 14.4+ SDK (current Xcode / Command Line
Tools). On older SDKs, add
`RUSTFLAGS='-C link-arg=-Wl,-U,_OBJC_CLASS_$_MLComputePlan -C link-arg=-Wl,-U,_OBJC_CLASS_$_MLOptimizationHints'`.

Part of the [skill-extractor](https://github.com/dreamjobs-tech/skill-extractor)
family (Python · JavaScript · Ruby · Rust — identical output, shared parity
fixtures). Built by [Qarera](https://www.qarera.com) — see our
[analysis of 360,000+ job postings](https://www.qarera.com/reports/most-in-demand-skills-2026)
for what this pipeline extracts at scale. MIT.
