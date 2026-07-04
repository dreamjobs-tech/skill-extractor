"""Parity tests against the shared fixtures (fixtures/extraction_fixtures.json).

Every language implementation of skill-extractor runs the same cases and must
produce identical candidates and final skill sets, with probabilities within
prob_tolerance of the reference.
"""
import json
from pathlib import Path

import pytest

from skill_extractor import SkillExtractor

FIXTURES = json.loads(
    (Path(__file__).parent.parent.parent / "fixtures" / "extraction_fixtures.json")
    .read_text()
)


@pytest.fixture(scope="module")
def extractor():
    return SkillExtractor()


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["text"][:40])
def test_case(extractor, case):
    cands = extractor.candidates(case["text"])
    assert [{"skill": s, "context": c} for s, c in cands] == case["candidates"]

    inputs = [f"{s} : {c}" for s, c in cands]
    assert inputs == case["embed_inputs"]

    if inputs:
        probs = extractor.mlp.predict_proba(extractor.embedder.encode(inputs))
        for got, want in zip(probs, case["probs"]):
            assert abs(got - want) < FIXTURES["prob_tolerance"]

    assert extractor.extract(case["text"]) == case["skills"]


def test_empty():
    # no model needed — matcher short-circuits
    from skill_extractor.matcher import KeywordMatcher

    m = KeywordMatcher(["python"])
    assert m.extract("") == []
    assert m.extract("no hits here") == []
    assert m.extract("python") == [("python", 0, 6)]
