"""skill-extractor: extract skills from job postings and resumes.

Gazetteer (32K skills) + MiniLM embeddings + MLP context classifier,
trained on 491K labeled samples (73% F1 held-out). Runs on ONNX Runtime —
no torch required.
"""
from .extractor import SkillExtractor
from .matcher import KeywordMatcher

__version__ = "0.1.0"
__all__ = ["SkillExtractor", "KeywordMatcher", "extract_skills"]

_default = None


def extract_skills(text: str, threshold: float = 0.5) -> list[str]:
    """Module-level convenience wrapper around a shared SkillExtractor."""
    global _default
    if _default is None:
        _default = SkillExtractor()
    return _default.extract(text, threshold=threshold)
