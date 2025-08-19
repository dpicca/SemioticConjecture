import math
import random
import logging
import hashlib
from typing import List

from ..config import get_runtime_config

logger = logging.getLogger(__name__)


def _stable_word_seed(word: str, base_seed: int) -> int:
    """Derive a stable per-word seed from a base seed using blake2b.

    This is independent of Python's hash randomization, ensuring reproducible
    pseudo-random vectors for the same word and base seed across processes.

    Args:
        word: Input token string.
        base_seed: Global/base integer seed.

    Returns:
        A 32-bit integer seed derived from the word and base seed.

    Examples:
        >>> s1 = _stable_word_seed('hello', 42)
        >>> s2 = _stable_word_seed('hello', 42)
        >>> s1 == s2
        True
    """
    h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    word_int = int.from_bytes(h, byteorder="big", signed=False)
    return (word_int ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF


from typing import Optional

def _hash_vec(word: str, dim: Optional[int] = None, seed: Optional[int] = None):
    """Generate a deterministic pseudo-random vector for a word.

    Uses a stable per-word seed derived from ``word`` and the provided/global seed.

    Args:
        word: Input token string.
        dim: Embedding dimensionality. Defaults to RuntimeConfig.embedding_dim.
        seed: Base seed. Defaults to RuntimeConfig.seed.

    Returns:
        A list of floats of length ``dim`` with values in [-1, 1].

    Examples:
        >>> v1 = _hash_vec('hello', dim=4, seed=42)
        >>> v2 = _hash_vec('hello', dim=4, seed=42)
        >>> v1 == v2
        True
    """
    cfg = get_runtime_config()
    d = int(dim if dim is not None else cfg.embedding_dim)
    s = int(seed if seed is not None else cfg.seed)
    rnd = random.Random(_stable_word_seed(word, s))
    return [rnd.uniform(-1.0, 1.0) for _ in range(d)]


def _avg_vec(words: List[str], dim: Optional[int] = None) -> List[float]:
    """Average hash-based word vectors to obtain a sentence/segment embedding.

    Args:
        words: List of token strings.
        dim: Optional embedding dimensionality. Defaults to RuntimeConfig.embedding_dim.

    Returns:
        A list of floats of length ``dim`` representing the average vector.

    Examples:
        >>> _avg_vec(['hello', 'world'], dim=4)
        [...]  # doctest: +ELLIPSIS
    """
    cfg = get_runtime_config()
    d = int(dim if dim is not None else cfg.embedding_dim)
    if not words:
        logger.debug("_avg_vec: empty word list -> zero vector")
        return [0.0] * d
    acc = [0.0] * d
    for w in words:
        v = _hash_vec(w, d)
        for i in range(d):
            acc[i] += v[i]
    n = max(1, len(words))
    return [a / n for a in acc]


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors with numeric stability.

    Args:
        a: First vector of floats.
        b: Second vector of floats (same length as ``a``).

    Returns:
        Cosine similarity in [-1, 1].

    Examples:
        >>> round(_cosine([1.0, 0.0], [1.0, 0.0]), 6)
        1.0
        >>> round(_cosine([1.0, 0.0], [0.0, 1.0]), 6)
        0.0
    """
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(y*y for y in b)) or 1e-9
    return max(-1.0, min(1.0, dot / (na * nb)))


def sentence_embedding_dispersion(sentences: List[List[str]]) -> float:
    """Compute mean pairwise cosine distance among sentence embeddings.

    Behavior:
    - If an Ollama embedding model is configured (e.g., 'nomic-embed-text'), use it to embed each sentence string.
    - Otherwise, or on any failure, fall back to hash-averaged word vectors for offline robustness.
    - Returns a value in [0,1]; higher means more dispersion (semantic variety).
    """
    if not sentences:
        logger.debug("sentence_embedding_dispersion: empty sentences -> 0.0")
        return 0.0

    cfg = get_runtime_config()
    vecs: List[List[float]]

    use_ollama_model = getattr(cfg, "embeddings_ollama_model", "").strip()
    if use_ollama_model:
        try:
            # Any empty token list would complicate dimension handling; fallback in that case
            if any(len(s) == 0 for s in sentences):
                raise ValueError("empty sentence token list")
            # Lazy import to avoid hard dependency when unused
            from ..surrogates.ollama_client import get_shared_client  # type: ignore
            client = get_shared_client(host=getattr(cfg, "embeddings_ollama_host", "http://localhost:11434"))
            vecs = [client.embed(use_ollama_model, " ".join(s)) for s in sentences]
        except Exception as e:
            logger.warning(
                "sentence_embedding_dispersion: Ollama embeddings failed or unavailable; falling back to hash-avg (%s)",
                e,
            )
            vecs = [_avg_vec(s) for s in sentences]
    else:
        vecs = [_avg_vec(s) for s in sentences]

    n = len(vecs)
    if n < 2:
        logger.debug("sentence_embedding_dispersion: <2 sentences -> 0.0")
        return 0.0
    dsum = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(vecs[i], vecs[j])
            dsum += (1.0 - sim)  # distance
            cnt += 1
    return dsum / max(1, cnt)

