import re
from collections import Counter
from typing import Dict, List, Set, Tuple


DOMAINS: Dict[str, Set[str]] = {
    "technology": {"algorithm", "software", "network", "data", "model", "system"},
    "biology": {"cell", "species", "genome", "protein", "organism"},
    "art": {"painting", "poetry", "novel", "canvas", "aesthetic"},
    "politics": {"election", "policy", "state", "government", "vote"},
    "finance": {"market", "risk", "equity", "capital", "investment"},
    "sports": {"match", "score", "team", "league", "coach"},
}


def ner_coverage(text: str) -> float:
    """Named-entity coverage in [0,1].

    Behavior:
    - Prefer cached result from central NLP cache.
    - If Ollama NLP is configured, reuse a single cached nlp() call and compute once.
    - Otherwise, fall back to capitalization heuristic and cache the result.
    """
    # Try cached value first
    try:
        from ..nlp_cache import get_cached_field, set_cached_field, get_nlp_result  # type: ignore
        cached_cov = get_cached_field(text, "ner_coverage")
        if isinstance(cached_cov, (int, float)):
            return float(cached_cov)
        # Attempt to use a single cached Ollama nlp() result
        res = get_nlp_result(text)
        if isinstance(res, dict):
            ents = res.get("entities", [])
            toks = res.get("tokens", [])
            total = len(toks) if isinstance(toks, list) and toks else None
            if isinstance(ents, list) and ents:
                if not total:
                    total = len(re.findall(r"[\w']+", text))
                total = max(1, int(total or 0))
                cov = len(ents) / total
                cov = max(0.0, min(1.0, cov))
                set_cached_field(text, "ner_coverage", cov)
                return cov
    except Exception:
        pass

    # Fallback heuristic; also cache the result
    tokens = re.findall(r"[\w']+", text)
    if not tokens:
        cov = 0.0
        try:
            from ..nlp_cache import set_cached_field  # type: ignore
            set_cached_field(text, "ner_coverage", cov)
        except Exception:
            pass
        return cov
    ent_like = [t for t in tokens if t[0:1].isupper() and t.lower() not in {"i"}]
    cov = min(1.0, len(ent_like) / max(1, len(tokens)))
    try:
        from ..nlp_cache import set_cached_field  # type: ignore
        set_cached_field(text, "ner_coverage", cov)
    except Exception:
        pass
    return cov


def domain_coverage_score(text: str) -> float:
    """Score in [0,1] for how many domain keyword sets are hit.
    Multi-domain coverage increases intertextuality.
    """
    words = set(w.lower() for w in re.findall(r"[\w']+", text))
    hits = 0
    for ws in DOMAINS.values():
        if words & ws:
            hits += 1
    return hits / max(1, len(DOMAINS))

