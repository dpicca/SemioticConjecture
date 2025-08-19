import re
from typing import List

from ..config import get_runtime_config

_METAPHOR_HINTS = [
    r"\blike\b", r"\bas\s+if\b", r"\bas\s+a\b", r"\bmetaphor\b", r"\bsymbol\w*\b",
]
_IRONY_HINTS = [
    r"\byeah\s+right\b", r"\bsure\b", r"\bgreat\b", r"\bwhat\s+a\s+great\b", r"\bclassic\b",
]


def figures_score(text: str) -> float:
    """Score figurative language intensity in [0,1].

    Behavior:
    - If a figures Ollama model is configured via runtime config, use it to score the text.
    - Otherwise, fall back to the original heuristic (metaphor/irony cue density).
    - In all cases, apply ``cfg.figures_multiplier`` and clip to [0,1].
    """
    from logging import getLogger
    logger = getLogger(__name__)
    cfg = get_runtime_config()

    def _heuristic(t: str) -> float:
        if not t.strip():
            logger.debug("figures_score: empty text -> 0.0")
            return 0.0
        L = max(1, len(t.split()))
        m = sum(len(re.findall(p, t.lower())) for p in _METAPHOR_HINTS)
        i = sum(len(re.findall(p, t.lower())) for p in _IRONY_HINTS)
        raw = (m + i) / L
        return max(0.0, min(1.0, 5.0 * raw))

    def _ollama(t: str) -> float:
        try:
            # Lazy imports to avoid hard dependency when unused
            from ..surrogates.ollama_client import get_shared_client  # type: ignore
            import json
            import re as _re
            if not getattr(cfg, "figures_ollama_model", "").strip():
                raise RuntimeError("No figures_ollama_model configured")
            client = get_shared_client(host=getattr(cfg, "figures_ollama_host", "http://localhost:11434"))
            system = (
                "You are an expert linguist. Respond in strict JSON only with a single key 'score' in [0,1]."
            )
            rubric = (
                "Rate the intensity of figurative language (metaphor, simile, irony, symbolism) in the provided text "
                "on a continuous scale from 0 to 1, where 0 means none and 1 means heavy. Do not include explanations."
            )
            prompt = "Text:\n" + t.strip() + "\n\n" + rubric + "\nRespond as JSON: {\"score\": <float between 0 and 1>}"
            raw = client.generate(
                model=getattr(cfg, "figures_ollama_model"),
                prompt=prompt,
                system=system,
                temperature=0.2,
                seed=getattr(cfg, "seed", None),
            )
            score_val = None
            if raw:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        obj = json.loads(raw[start:end+1])
                        score_val = obj.get("score", None)
                        if score_val is not None:
                            score_val = float(score_val)
                    except Exception:
                        score_val = None
            if score_val is None and raw:
                m = _re.search(r"([01](?:\\.\\d+)?)", raw)
                if m:
                    try:
                        score_val = float(m.group(1))
                    except Exception:
                        score_val = None
            if score_val is None:
                raise ValueError("Could not parse 'score' from Ollama response")
            return max(0.0, min(1.0, float(score_val)))
        except Exception as e:
            logger.warning("figures_score: Ollama scoring failed or unavailable; falling back to heuristic (%s)", e)
            raise

    # Try Ollama if configured; otherwise heuristic
    base_score: float
    if getattr(cfg, "figures_ollama_model", "").strip():
        try:
            base_score = _ollama(text)
            return max(0.0, min(1.0, float(cfg.figures_multiplier) * base_score))
        except Exception:
            logger.warning("figures_score: Ollama scoring failed; falling back to heuristic")
            pass  # fall through to heuristic
    base_score = _heuristic(text)
    return max(0.0, min(1.0, float(cfg.figures_multiplier) * base_score))

