"""Skill extraction pipeline: gazetteer match -> context window -> MiniLM
embed -> MLP classify. Mirrors the production extractor trained on 491K
labeled samples (73% F1 on held-out job postings)."""
import json
from importlib import resources

from .embedder import Embedder
from .matcher import KeywordMatcher
from .mlp import MLP

CONTEXT_WORDS_BEFORE = 20
CONTEXT_WORDS_AFTER = 21
DEFAULT_THRESHOLD = 0.5


class SkillExtractor:
    def __init__(self, quantized: bool = False):
        data = resources.files("skill_extractor") / "data"
        with (data / "skills.json").open() as f:
            self.matcher = KeywordMatcher(json.load(f))
        self.mlp = MLP(str(data / "mlp.json"))
        self.embedder = Embedder(quantized=quantized)

    def candidates(self, text: str) -> list[tuple[str, str]]:
        """Gazetteer matches with their +/-20-word context windows."""
        matches = self.matcher.extract(text)
        if not matches:
            return []
        words = text.split()
        out = []
        for skill, start, _end in matches:
            word_idx = len(text[:start].split())
            lo = max(0, word_idx - CONTEXT_WORDS_BEFORE)
            hi = min(len(words), word_idx + CONTEXT_WORDS_AFTER)
            out.append((skill, " ".join(words[lo:hi])))
        return out

    def extract(
        self, text: str, threshold: float = DEFAULT_THRESHOLD
    ) -> list[str]:
        """Extract confirmed skills from a job posting or resume text."""
        cands = self.candidates(text)
        if not cands:
            return []
        probs = self.mlp.predict_proba(
            self.embedder.encode([f"{s} : {c}" for s, c in cands])
        )
        return sorted({cands[i][0] for i, p in enumerate(probs) if p >= threshold})
