/** Parity test against the shared fixtures — must match the Python reference. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

import { SkillExtractor } from '../src/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = JSON.parse(
  readFileSync(join(here, '..', '..', 'fixtures', 'extraction_fixtures.json'), 'utf8')
);

const ex = new SkillExtractor();
let failures = 0;

for (const c of fixtures.cases) {
  const label = JSON.stringify(c.text.slice(0, 40));
  try {
    const cands = ex.candidates(c.text);
    assert.deepEqual(cands, c.candidates, 'candidates mismatch');

    const inputs = cands.map((x) => `${x.skill} : ${x.context}`);
    assert.deepEqual(inputs, c.embed_inputs, 'embed inputs mismatch');

    if (inputs.length) {
      const probs = await ex.classify(inputs);
      probs.forEach((p, i) => {
        assert.ok(
          Math.abs(p - c.probs[i]) < fixtures.prob_tolerance,
          `prob[${i}] ${p} vs ${c.probs[i]}`
        );
      });
    }

    const skills = await ex.extract(c.text);
    assert.deepEqual(skills, c.skills, 'final skills mismatch');
    console.log(`ok   ${label}`);
  } catch (err) {
    failures += 1;
    console.error(`FAIL ${label}: ${err.message}`);
  }
}

if (failures) {
  console.error(`\n${failures} case(s) failed`);
  process.exit(1);
}
console.log(`\nall ${fixtures.cases.length} cases passed`);
