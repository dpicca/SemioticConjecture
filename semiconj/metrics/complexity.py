import logging
import math
import re
from collections import Counter
from typing import List, Tuple


_WORD_RE = re.compile(r"[\w']+")


def tokenize(text: str) -> List[str]:
    """Tokenize text.

    Behavior:
    - If an Ollama NLP model is configured (e.g., 'gpt-oss'), use it to obtain tokens.
    - Otherwise, fall back to a simple regex-based tokenizer.
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
                toks = res.get("tokens", [])
                if isinstance(toks, list) and toks:
                    # Normalize to lower-case to preserve previous behavior
                    return [str(t).lower() for t in toks if isinstance(t, str)]
            except Exception:
                # fall back to regex below
                logging.getLogger(__name__).warning(
                    f"Ollama NLP model '{model}' not found. Falling back to regex-based tokenization."
                )
                pass
    except Exception:
        # If config import fails for any reason, use regex
        pass
    return _WORD_RE.findall(text.lower())


def mtld(tokens: List[str], ttr_threshold: float = 0.72, min_segment: int = 10) -> float:
    """Approximate MTLD (Measure of Textual Lexical Diversity).

    Reference: McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D:
    A validation study of sophisticated approaches to lexical diversity assessment.

    Returns a positive value, roughly stable across lengths; higher implies more diversity.
    Note: This function returns a raw MTLD-like score, not normalized to [0,1].
    """
    if not tokens:
        return 0.0
    types = set()
    factor_count = 0
    token_count = 0
    ttr = 1.0
    for tok in tokens:
        token_count += 1
        types.add(tok)
        ttr = len(types) / token_count
        if ttr <= ttr_threshold and token_count >= min_segment:
            factor_count += 1
            types.clear()
            token_count = 0
    if token_count > 0:
        partial = (1 - (len(types) / max(1, token_count))) / max(1e-9, 1 - ttr_threshold)
        factor_count += partial
    return (len(tokens) / max(1e-9, factor_count))


def yules_k(tokens: List[str]) -> float:
    """Yule's K lexical diversity measure.

    Reference: Yule, G. U. (1944). The Statistical Study of Literary Vocabulary.
    Lower K implies higher lexical diversity. We return 1/(1+K) to map to (0,1],
    keeping a larger-is-better scale and bounding at 1 when K→0.
    """
    if not tokens:
        return 0.0
    freqs = Counter(tokens)
    N = sum(freqs.values())
    M2 = sum(n * n * v for n, v in Counter(freqs.values()).items())
    K = 1e4 * (M2 - N) / (N * N)
    # invert to larger-better, normalized roughly to [0,1]
    return 1.0 / (1.0 + K)


def naive_pos_tags(tokens: List[str]) -> List[str]:
    """Very rough POS tags if no tagger available.
    This is a heuristic; replace with spaCy/UD tagger if available.
    """
    tags = []
    for w in tokens:
        if w.endswith('ly'):
            tags.append('RB')
        elif w.endswith('ing') or w.endswith('ed'):
            tags.append('VB')
        elif w[0:1].isupper():
            tags.append('NNP')
        elif w in {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had"}:
            tags.append('VB')
        elif w in {"the", "a", "an"}:
            tags.append('DT')
        elif w in {"and", "or", "but", "so", "because"}:
            tags.append('CC')
        else:
            tags.append('NN')
    return tags


# Universal POS tagset (UPOS) for normalization reference
_UPOS_TAGS = {
    "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
}


def _maybe_spacy_pos(tokens: List[str]) -> List[str]:
    try:
        import spacy  # type: ignore
    except Exception as e:
        raise ImportError("spaCy is required for pos_tagger='spacy'. Install spacy and the 'en_core_web_sm' model.") from e
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e
    doc = nlp(" ".join(tokens))
    return [t.pos_ or "X" for t in doc]


def pos_tags(tokens: List[str]) -> List[str]:
    """Return POS tags for the provided tokens.

    Behavior:
    - If an Ollama NLP model is configured (e.g., 'gpt-oss'), request UPOS tags via the Ollama API.
      The model is prompted to return strict JSON {"pos": [UPOS...]} aligned with the given tokens.
    - Otherwise, if cfg.pos_tagger == 'spacy', use spaCy (requires en_core_web_sm).
    - Otherwise, fall back to the naive heuristic tagger.
    """
    from ..config import get_runtime_config
    cfg = get_runtime_config()

    # Try Ollama-based POS tagging first if configured
    model = getattr(cfg, "nlp_ollama_model", "").strip()
    if model and tokens:
        try:
            from ..surrogates.ollama_client import OllamaClient  # type: ignore
            import json
            allowed = sorted(list(_UPOS_TAGS))
            client = OllamaClient(host=getattr(cfg, "nlp_ollama_host", "http://localhost:11434"))
            # Build a strict prompt to tag exactly the provided tokens
            system = (
                "You are a POS tagger. Respond in strict JSON only with a single key 'pos' "
                "containing Universal POS tags (UPOS) for each input token in order."
            )
            # Use json.dumps to pass the exact token list to avoid tokenization drift
            tokens_json = json.dumps(tokens)
            rubric = (
                "Tag each token with one of the UPOS tags: " + ", ".join(allowed) + ". "
                + "Return JSON exactly as: {\"pos\": [\"TAG1\", \"TAG2\", \"...\"]} with length equal to the number of input tokens."
            )
            prompt = (
                "Tokens:\n" + tokens_json + "\n\n" + rubric + "\n"
                "Do not include explanations or additional keys."
            )
            raw = client.generate(model=model, prompt=prompt, system=system, temperature=0.0, seed=getattr(cfg, "seed", None))
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(raw[start:end+1])
                    pos_list = obj.get("pos", [])
                    if isinstance(pos_list, list) and len(pos_list) == len(tokens):
                        tags: List[str] = []
                        for t in pos_list:
                            tag = str(t).upper()
                            tags.append(tag if tag in _UPOS_TAGS else "X")
                        return tags
                except Exception:
                    pass
        except Exception:
            # Fall through to spaCy/naive
            logging.getLogger(__name__).warning('Failed to load Ollama NLP model. Falling back to naive POS tagger.')
            pass

    # Fallbacks
    if cfg.pos_tagger == "spacy":
        return _maybe_spacy_pos(tokens)
    return naive_pos_tags(tokens)


def pos_entropy(tokens: List[str]) -> float:
    """Shannon entropy of POS tag distribution normalized to [0,1].

    Normalization uses log2(|UPOS|) as the denominator to make scores comparable
    across texts, regardless of how many tag types are observed in a short sample.
    """
    tags = pos_tags(tokens)
    total = len(tags) or 1
    counts = Counter(tags)
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p + 1e-12, 2)
    max_H = math.log(max(1, len(_UPOS_TAGS)), 2)
    return max(0.0, min(1.0, H / (max_H + 1e-9)))


def senses_per_lemma(tokens: List[str]) -> float:
    """Mean WordNet senses per lemma (raw; not normalized).

    Strict behavior: requires NLTK WordNet corpus to be installed and available.
    Raises ImportError/LookupError if NLTK or its WordNet data is missing.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception as e:
        raise ImportError("NLTK is required for senses_per_lemma. Install nltk and wordnet data.") from e
    lemmas = set(tokens)
    if not lemmas:
        return 0.0
    total = 0
    try:
        for w in lemmas:
            total += len(wn.synsets(w))
    except Exception as e:
        # Typically LookupError when wordnet data is missing
        raise LookupError("NLTK WordNet data not found. Run: python -m nltk.downloader wordnet") from e
    return total / max(1, len(lemmas))

