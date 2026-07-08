"""Regenerate the shared parity fixtures from the current gazetteer.

Run whenever artifacts/skills.json changes:

    python fixtures/generate.py

The Python implementation is the reference: matcher spans and extraction
outputs are computed with it, and every other language implementation must
reproduce them exactly (probabilities within prob_tolerance).

Deterministic — the fuzz cases are seeded, so reruns on the same gazetteer
are byte-identical.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))

from skill_extractor import SkillExtractor  # noqa: E402
from skill_extractor.matcher import KeywordMatcher  # noqa: E402

N_FUZZ = 300
SEED = 20260708
SEPARATORS = [" ", "  ", "/", ", ", ";", " - ", "\n", "//", "/?/", " & ", ": "]
DISTRACTORS = [
    "experience", "with", "and", "required", "plus", "team", "our", "the",
    "years", "strong", "knowledge", "of", "in", "a", "candidates", "must",
    "have", "excellent", "background", "preferred",
]

# Fixed job-ad style texts exercising the full pipeline: German/English mix,
# casing, punctuation boundaries, adjacent-word non-matches, empty results.
E2E_TEXTS_FILE = Path(__file__).parent / "e2e_texts.json"


def gen_matcher_fixtures(skills: list[str]) -> list[dict]:
    rng = random.Random(SEED)
    matcher = KeywordMatcher(skills)
    cases = []
    for _ in range(N_FUZZ):
        parts = []
        for _ in range(rng.randint(2, 8)):
            if rng.random() < 0.25:
                parts.append(rng.choice(DISTRACTORS))
            else:
                kw = rng.choice(skills)
                # exercise case-insensitivity
                parts.append(kw.upper() if rng.random() < 0.15 else kw)
        text = ""
        for i, p in enumerate(parts):
            text += p
            if i < len(parts) - 1:
                text += rng.choice(SEPARATORS)
        spans = [[kw, s, e] for kw, s, e in matcher.extract(text)]
        cases.append({"text": text, "spans": spans})
    return cases


def gen_extraction_fixtures(extractor: SkillExtractor) -> dict:
    texts = json.loads(E2E_TEXTS_FILE.read_text())
    cases = []
    for text in texts:
        cands = extractor.candidates(text)
        inputs = [f"{s} : {c}" for s, c in cands]
        probs = (
            [float(p) for p in extractor.mlp.predict_proba(extractor.embedder.encode(inputs))]
            if inputs else []
        )
        cases.append({
            "text": text,
            "candidates": [{"skill": s, "context": c} for s, c in cands],
            "embed_inputs": inputs,
            "probs": probs,
            "skills": extractor.extract(text),
        })
    return {"threshold": 0.5, "prob_tolerance": 0.002, "cases": cases}


def main() -> None:
    skills = json.loads((ROOT / "artifacts" / "skills.json").read_text())
    print(f"gazetteer: {len(skills)} entries")

    matcher_cases = gen_matcher_fixtures(skills)
    out = Path(__file__).parent / "matcher_fixtures.json"
    out.write_text(json.dumps(matcher_cases, ensure_ascii=False) + "\n")
    n_spans = sum(len(c["spans"]) for c in matcher_cases)
    print(f"wrote {out.name}: {len(matcher_cases)} cases, {n_spans} spans")

    extraction = gen_extraction_fixtures(SkillExtractor())
    out = Path(__file__).parent / "extraction_fixtures.json"
    out.write_text(json.dumps(extraction, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out.name}: {len(extraction['cases'])} cases")


if __name__ == "__main__":
    main()
