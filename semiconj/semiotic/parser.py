import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from ..metrics.embeddings import sentence_embedding_dispersion, _cosine


SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[\w']+")


@dataclass
class Triad:
    sign: str
    obj: str
    interpretant: str


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Behavior:
    - Prefer cached results (central NLP cache) to avoid recomputation.
    - If an Ollama NLP model is configured, reuse a single cached nlp() call.
    - Otherwise, fall back to regex-based split and cache locally.
    """
    # Try cache first
    try:
        from ..nlp_cache import get_cached_field, set_cached_field, get_nlp_result  # type: ignore
        cached = get_cached_field(text, "sentences")
        if isinstance(cached, list) and cached:
            return list(cached)
        # Ensure at most one Ollama NLP call if configured
        res = get_nlp_result(text)
        if isinstance(res, dict):
            sents = res.get("sentences")
            if isinstance(sents, list) and sents:
                norm = [str(s).strip() for s in sents if str(s).strip()]
                set_cached_field(text, "sentences", norm)
                return norm
    except Exception:
        pass

    # Original behavior (regex-based), also cache the result
    parts = SENT_RE.split(text.strip())
    out = [p for p in parts if p]
    try:
        from ..nlp_cache import set_cached_field  # type: ignore
        set_cached_field(text, "sentences", out)
    except Exception:
        pass
    return out


def words(s: str) -> List[str]:
    """Tokenize a sentence string into words, delegating to central tokenizer.
    Uses metrics.complexity.tokenize to honor Ollama-based tokenization when enabled.
    """
    try:
        from ..metrics.complexity import tokenize  # type: ignore
        return tokenize(s)
    except Exception:
        return WORD_RE.findall(s.lower())


def extract_triads(text: str) -> List[Triad]:
    """Rule-based triad extraction heuristics.
    - sign: prominent noun-like token (capitalized word or noun-ish pattern)
    - object: next capitalized token or repeated head noun
    - interpretant: definitional or summary sentence containing 'is/means/represents/defines'
    """
    sents = split_sentences(text)
    triads: List[Triad] = []
    for s in sents:
        w = WORD_RE.findall(s)
        if not w:
            continue
        # sign
        sign_candidates = [t for t in w if (t[0:1].isupper() and len(t) > 1)]
        sign = sign_candidates[0] if sign_candidates else (w[0] if w else "")
        # object
        obj_candidates = [t for t in w[1:] if (t[0:1].isupper() and len(t) > 1)]
        obj = obj_candidates[0] if obj_candidates else (w[1] if len(w) > 1 else sign)
        # interpretant
        if re.search(r"\b(is|means|represents|defines)\b", s, flags=re.IGNORECASE):
            interpretant = s.strip()
        else:
            interpretant = s.strip()
        if sign and obj and interpretant:
            triads.append(Triad(sign=str(sign), obj=str(obj), interpretant=interpretant))
    return triads




def interpretant_vectors(interps: List[str]) -> List[List[float]]:
    # reuse hash-avg embeddings from words
    from ..metrics.embeddings import _avg_vec  # type: ignore
    return [_avg_vec(words(s)) for s in interps]


def global_coherence(text: str) -> float:
    sents = split_sentences(text)
    if len(sents) < 2:
        return 0.0
    tokenized = [words(s) for s in sents]
    disp = sentence_embedding_dispersion(tokenized)
    # coherence high when dispersion low; invert and normalize
    return max(0.0, min(1.0, 1.0 - disp))


def interpretive_entropy(interps: List[str]) -> float:
    if not interps:
        return 1.0
    vecs = interpretant_vectors(interps)
    # pairwise similarity distribution
    sims = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sims.append(_cosine(vecs[i], vecs[j]))
    if not sims:
        return 1.0
    # Entropy proxy: 1 - mean similarity (higher dissimilarity = higher entropy)
    mean_sim = sum(sims) / len(sims)
    return max(0.0, min(1.0, 1.0 - mean_sim))


def compute_d_intr(text: str) -> Dict[str, float]:
    """Compute intrinsic decodability components and aggregate.
    Returns dict with: structural_precision, convergence, coherence, entropy, D_intr.
    """
    triads = extract_triads(text)
    sents = split_sentences(text)
    structural_precision = 0.0
    if sents:
        structural_precision = sum(1 for s in sents if s.strip())
        structural_precision = min(1.0, len(triads) / max(1, len(sents)))
    interps = [t.interpretant for t in triads]
    # convergence: mean pairwise similarity among interpretants
    if len(interps) < 2:
        convergence = 0.0
    else:
        vecs = interpretant_vectors(interps)
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(_cosine(vecs[i], vecs[j]))
        convergence = max(0.0, sum(sims) / max(1, len(sims)))
    coherence = global_coherence(text)
    entropy = interpretive_entropy(interps)
    # Aggregate: geometric-like combination with entropy penalty
    base = (structural_precision + convergence + coherence) / 3.0
    d_intr = max(0.0, min(1.0, base * (1.0 - 0.5 * entropy)))
    return {
        "structural_precision": structural_precision,
        "convergence": convergence,
        "coherence": coherence,
        "entropy": entropy,
        "D_intr": d_intr,
    }

