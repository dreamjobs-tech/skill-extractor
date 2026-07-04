# skill-extractor (JavaScript)

Extract skills from job postings and resumes. A 32,000-skill gazetteer proposes
candidates; a MiniLM + MLP context classifier accepts or rejects each one, so
"we value a can-do attitude" doesn't become a skill. Trained on 491K labeled
samples, 73% F1 on held-out job postings.

Runs on [transformers.js](https://github.com/huggingface/transformers.js) —
the MiniLM model downloads from the Hugging Face Hub on first use and is cached.

```bash
npm install skill-extractor
```

```js
import { extractSkills } from 'skill-extractor';

const text = `Senior Backend Engineer. 5+ years Python or Go, REST APIs with
FastAPI, PostgreSQL, AWS (ECS, Lambda). Docker and Kubernetes required.
Strong communication skills.`;

console.log(await extractSkills(text));
// ['aws', 'communication', 'docker', 'ecs', 'fastapi', 'go', 'kubernetes',
//  'lambda', 'postgresql', 'python', 'rest apis']
```

Lower-level API:

```js
import { SkillExtractor } from 'skill-extractor';

const ex = new SkillExtractor({ quantized: true }); // 23MB model, ~4x faster
await ex.extract(text, 0.7);  // stricter threshold
ex.candidates(text);          // raw gazetteer hits + context windows
```

Part of the [skill-extractor](https://github.com/dreamjobs-tech/skill-extractor)
family (Python · JavaScript · Ruby · Rust — identical output, shared parity
fixtures). Built by [Qarera](https://www.qarera.com) — see our
[analysis of 360,000+ job postings](https://www.qarera.com/reports/most-in-demand-skills-2026)
for what this pipeline extracts at scale. MIT.
