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
    - If an Ollama NLP model is configured (e.g., 'gpt-oss'), use its extracted entities.
    - Otherwise, fall back to a capitalization-based heuristic proxy.
    """
    try:
        from ..config import get_runtime_config
        cfg = get_runtime_config()
        model = getattr(cfg, "nlp_ollama_model", "").strip()
        if model:
            try:
                from ..surrogates.ollama_client import get_shared_client  # type: ignore
                client = get_shared_client(host=getattr(cfg, "nlp_ollama_host", "http://localhost:11434"))
                res = client.nlp(model=model, text=text)
                ents = res.get("entities", [])
                toks = res.get("tokens", [])
                total = len(toks) if isinstance(toks, list) and toks else None
                if isinstance(ents, list) and ents:
                    if not total:
                        # If tokenizer not provided or empty, estimate using regex length
                        total = len(re.findall(r"[\w']+", text))
                    total = max(1, int(total or 0))
                    cov = len(ents) / total
                    return max(0.0, min(1.0, cov))
            except Exception:
                pass
    except Exception:
        pass
    tokens = re.findall(r"[\w']+", text)
    if not tokens:
        return 0.0
    ent_like = [t for t in tokens if t[0:1].isupper() and t.lower() not in {"i"}]
    return min(1.0, len(ent_like) / max(1, len(tokens)))


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

