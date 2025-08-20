"""
Centralized NLP cache to avoid repeated heavy computations per text.

This module provides lightweight, in-memory memoization for:
- Results from Ollama client's nlp(model, text), if configured.
- Locally computed fields (tokens, sentences, ner_coverage) when Ollama is not used.

API:
- get_nlp_result(text) -> dict | None: returns and caches the Ollama nlp result if a model is configured.
- get_cached_field(text, field) -> Any | None: returns a cached field (e.g., 'tokens', 'sentences', 'entities', 'ner_coverage') if available.
- set_cached_field(text, field, value) -> None: stores a field for this text in the cache.

The cache key includes the runtime model and host when Ollama-based NLP is enabled, so varying models/hosts don't collide.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Internal cache maps (mode, host, model, text) -> dict(fields)
_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}


def _runtime_key(text: str) -> Tuple[str, str, str, str]:
    """Build a cache key for the given text considering runtime NLP config."""
    mode = "local"
    host = ""
    model = ""
    try:
        from ..config import get_runtime_config  # lazy import
        cfg = get_runtime_config()
        model = getattr(cfg, "nlp_ollama_model", "").strip()
        host = getattr(cfg, "nlp_ollama_host", "http://localhost:11434").strip()
        if model:
            mode = "ollama"
    except Exception:
        # If config is unavailable, treat as local
        mode = "local"
        host = ""
        model = ""
    return (mode, host, model, text)


def _get_bucket(text: str) -> Dict[str, Any]:
    key = _runtime_key(text)
    bucket = _CACHE.get(key)
    if bucket is None:
        bucket = {}
        _CACHE[key] = bucket
    return bucket


def get_cached_field(text: str, field: str) -> Any:
    """Return a cached field if present, else None."""
    return _get_bucket(text).get(field)


def set_cached_field(text: str, field: str, value: Any) -> None:
    """Store a field value in the cache for this text."""
    _get_bucket(text)[field] = value


def get_nlp_result(text: str) -> Dict[str, Any] | None:
    """
    If an Ollama NLP model is configured, call client.nlp(model, text) ONCE per
    (model, host, text) and cache the entire result. Returns the JSON-like dict
    or None if no model is configured or the call fails.
    """
    bucket = _get_bucket(text)
    # Return cached full nlp result if present
    if "nlp_result" in bucket:
        return bucket["nlp_result"]

    # Try to call Ollama NLP if configured
    try:
        from ..config import get_runtime_config  # lazy import
        cfg = get_runtime_config()
        model = getattr(cfg, "nlp_ollama_model", "").strip()
        if not model:
            return None
        host = getattr(cfg, "nlp_ollama_host", "http://localhost:11434").strip()
        try:
            from ..surrogates.ollama_client import get_shared_client  # type: ignore
            client = get_shared_client(host=host)
            res = client.nlp(model=model, text=text)
            if isinstance(res, dict):
                # Normalize and place salient fields into bucket as well
                bucket["nlp_result"] = res
                toks = res.get("tokens")
                if isinstance(toks, list):
                    bucket.setdefault("tokens", [str(t).lower() for t in toks if isinstance(t, str)])
                sents = res.get("sentences")
                if isinstance(sents, list):
                    norm_sents = [str(s).strip() for s in sents if str(s).strip()]
                    bucket.setdefault("sentences", norm_sents)
                ents = res.get("entities")
                if isinstance(ents, list):
                    bucket.setdefault("entities", ents)
                return res
        except Exception:
            logger.warning("Ollama NLP call failed; proceeding without cached nlp_result.", exc_info=True)
            return None
    except Exception:
        # No runtime config; treat as local
        return None

    return None
