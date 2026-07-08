# Changelog

## 0.2.0 — 2026-07-08

Gazetteer cleanup and extension: 31,836 → 30,957 entries. Match spans and
extraction results change where the removed/added names occur; the model and
pipeline are unchanged.

### Removed (1,003)
- **892 O*NET occupation titles** that had leaked in as skills ("registered
  nurses", "home appliance repairers", …). Occupations are not skills; they
  produced false extractions on any posting that named its own role.
- **94 ambiguous 1–2 character tokens** ("al", "bc", "bs", "ga", …) that
  collide with state codes, name fragments, or degree abbreviations and
  matched constantly in unrelated text. Unambiguous short technical terms
  ("5g", "f#", "d3", "k6", "qt", "rf", "qc", …) are kept.
- **17 standalone generic words** ("security", "testing", "automation", …).
  The specific skills containing them ("penetration testing",
  "test automation", …) remain.

### Added (124)
- Skill names mined from real resumes parsed by Qarera's production pipeline,
  each seen in three or more independent resumes ("amazon web services",
  "seaborn", "datadog", "spring security", "webrtc", "mern stack", …).
  Qarera-originated, same attestation basis as the published skills-2026
  dataset.

### Tooling
- `fixtures/generate.py`: parity fixtures are now regenerated
  deterministically from the gazetteer (seeded fuzz cases + fixed e2e texts
  in `fixtures/e2e_texts.json`), so gazetteer changes can be re-fixtured
  reproducibly. The Python implementation is the reference.

## 0.1.0 — 2026-07-04

Initial release: gazetteer + MiniLM (ONNX) + MLP pipeline in Python,
JavaScript, Ruby, and Rust with shared parity fixtures.
