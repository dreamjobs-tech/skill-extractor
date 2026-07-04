# skill-extractor (Ruby)

Extract skills from job postings and resumes. A 32,000-skill gazetteer proposes
candidates; a MiniLM + MLP context classifier accepts or rejects each one, so
"we value a can-do attitude" doesn't become a skill. Trained on 491K labeled
samples, 73% F1 on held-out job postings.

Runs on ONNX Runtime via [informers](https://github.com/ankane/informers) — the
MiniLM model downloads from the Hugging Face Hub on first use and is cached.

```bash
gem install skill-extractor
```

```ruby
require "skill_extractor"

text = "Senior Backend Engineer. 5+ years Python or Go, REST APIs with " \
       "FastAPI, PostgreSQL, AWS (ECS, Lambda). Docker and Kubernetes required."

SkillExtractor.extract_skills(text)
# => ["aws", "docker", "ecs", "fastapi", "go", "kubernetes", "lambda",
#     "postgresql", "python", "rest apis"]
```

Lower-level API:

```ruby
ex = SkillExtractor::Extractor.new(quantized: true) # 23MB model, ~4x faster
ex.extract(text, threshold: 0.7)  # stricter
ex.candidates(text)               # raw gazetteer hits + context windows
```

Part of the [skill-extractor](https://github.com/dreamjobs-tech/skill-extractor)
family (Python · JavaScript · Ruby · Rust — identical output, shared parity
fixtures). Built by [Qarera](https://www.qarera.com) — see our
[analysis of 360,000+ job postings](https://www.qarera.com/reports/most-in-demand-skills-2026)
for what this pipeline extracts at scale. MIT.
