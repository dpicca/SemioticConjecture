from dataclasses import dataclass
from typing import Dict, List


# Weights for S(M) components (sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantics": 0.30,        # embedding dispersion, domain coverage
    "intertextuality": 0.25,  # NER+EL coverage, domain diversity
    "figures": 0.15,          # metaphor/irony proxies
    "lexicon": 0.15,          # MTLD, Yule's K, senses/lemma
    "pos": 0.10,              # POS entropy
    "codeswitch": 0.05,       # code-switching index
}


@dataclass(frozen=True)
class Context:
    """Context configuration describing interpretive rigidity.

    Attributes:
        name: Context identifier (e.g., 'C_open', 'C_medium', 'C_rigid').
        description: Human-readable description of the context constraints.

    Examples:
        >>> Context('C_open', 'No instructions.')
        Context(name='C_open', description='No instructions.')
    """
    name: str
    description: str


CONTEXTS: List[Context] = [
    Context("C_open", "No instructions (free interpretive mode)."),
    Context("C_medium", "Short framing instruction; light rubric."),
    Context("C_rigid", "Explicit rubric/criteria; constrained outputs."),
]


@dataclass
class RuntimeConfig:
    """Centralized runtime configuration.

    Attributes:
        seed: Global random seed for reproducibility.
        embedding_dim: Default embedding dimension for hashing/averaging embeddings.
        figures_multiplier: Multiplier applied to figures_score to tune its contribution.
        pos_tagger: Which POS tagger to use ("naive" or "spacy").
        figures_ollama_model: If non-empty, use this Ollama model to compute figures_score.
        figures_ollama_host: Ollama server host URL for figures scoring.
        embeddings_ollama_model: If non-empty, use this Ollama embedding model (e.g., 'nomic-embed-text') for sentence embeddings.
        embeddings_ollama_host: Ollama server host URL for embeddings (defaults to localhost).
        nlp_ollama_model: If non-empty, use this Ollama model (e.g., 'gpt-oss') for tokenization, sentence splitting, and NER.
        nlp_ollama_host: Ollama server host URL for NLP tasks (defaults to localhost).

    Examples:
        >>> cfg = RuntimeConfig(seed=123, embedding_dim=64, figures_multiplier=1.5, pos_tagger='naive')
        >>> cfg.seed, cfg.embedding_dim, cfg.pos_tagger
        (123, 64, 'naive')
    """
    seed: int = 42
    embedding_dim: int = 128
    figures_multiplier: float = 1.0
    pos_tagger: str = ""
    figures_ollama_model: str = "gpt-oss:latest"
    figures_ollama_host: str = "http://localhost:11434"
    embeddings_ollama_model: str = "nomic-embed-text:v1.5"
    embeddings_ollama_host: str = "http://localhost:11434"
    nlp_ollama_model: str = "gpt-oss:latest"
    nlp_ollama_host: str = "http://localhost:11434"


_runtime_config = RuntimeConfig()


def get_runtime_config() -> RuntimeConfig:
    """Return the current runtime configuration object.

    Returns:
        The live RuntimeConfig instance used across the library.

    Examples:
        >>> cfg = get_runtime_config()
        >>> isinstance(cfg, RuntimeConfig)
        True
    """
    return _runtime_config


from typing import Optional

def set_runtime_config(seed: Optional[int] = None, embedding_dim: Optional[int] = None, figures_multiplier: Optional[float] = None, pos_tagger: Optional[str] = None, figures_ollama_model: Optional[str] = None, figures_ollama_host: Optional[str] = None, embeddings_ollama_model: Optional[str] = None, embeddings_ollama_host: Optional[str] = None, nlp_ollama_model: Optional[str] = None, nlp_ollama_host: Optional[str] = None) -> None:
    """Update the global runtime configuration with provided values.

    Args:
        seed: Global random seed to set.
        embedding_dim: Default embedding dimension for hashing/averaging embeddings.
        figures_multiplier: Multiplier applied to figures_score for tuning.
        pos_tagger: Either "naive" or "spacy" (strictly validated).
        figures_ollama_model: If provided, enable Ollama-based figures scoring using this model name.
        figures_ollama_host: Ollama HTTP host used for figures scoring (defaults to localhost).
        embeddings_ollama_model: If provided, enable Ollama-based sentence embeddings using this model name.
        embeddings_ollama_host: Ollama HTTP host used for embeddings (defaults to localhost).
        nlp_ollama_model: If provided, enable Ollama-based NLP (tokenization, sentence splitting, NER) using this model name.
        nlp_ollama_host: Ollama HTTP host used for NLP tasks (defaults to localhost).

    Raises:
        ValueError: If ``pos_tagger`` is provided and not one of {"naive", "spacy"}.

    Examples:
        >>> set_runtime_config(seed=123, embedding_dim=64, pos_tagger='naive')  # doctest: +SKIP
        >>> get_runtime_config().seed  # doctest: +SKIP
        123
    """
    if seed is not None:
        _runtime_config.seed = int(seed)
    if embedding_dim is not None:
        _runtime_config.embedding_dim = int(embedding_dim)
    if figures_multiplier is not None:
        _runtime_config.figures_multiplier = float(figures_multiplier)
    if pos_tagger is not None:
        pt = str(pos_tagger).lower()
        if pt not in {"naive", "spacy"}:
            raise ValueError("pos_tagger must be 'naive' or 'spacy'")
        _runtime_config.pos_tagger = pt
    if figures_ollama_model is not None:
        _runtime_config.figures_ollama_model = str(figures_ollama_model)
    if figures_ollama_host is not None:
        _runtime_config.figures_ollama_host = str(figures_ollama_host)
    if embeddings_ollama_model is not None:
        _runtime_config.embeddings_ollama_model = str(embeddings_ollama_model)
    if embeddings_ollama_host is not None:
        _runtime_config.embeddings_ollama_host = str(embeddings_ollama_host)
    if nlp_ollama_model is not None:
        _runtime_config.nlp_ollama_model = str(nlp_ollama_model)
    if nlp_ollama_host is not None:
        _runtime_config.nlp_ollama_host = str(nlp_ollama_host)


def validate_config(weights: Dict[str, float], contexts: List[Context]) -> None:
    """Validate configuration consistency.

    Args:
        weights: Mapping of S(M) component names to weights expected to sum to 1.0.
        contexts: List of Context objects describing available interpretive contexts.

    Raises:
        ValueError: If weights do not sum to 1.0 (±1e-6), if no contexts are provided,
            or if any context has an empty name.

    Notes:
        This function enforces strict invariants used by the pipeline. Call it early
        to fail fast on misconfiguration.

    Examples:
        >>> good_w = {'a': 0.5, 'b': 0.5}
        >>> good_c = [Context('C_open', '...')]
        >>> validate_config(good_w, good_c)
        >>> bad_w = {'a': 0.7, 'b': 0.2}
        >>> try:
        ...     validate_config(bad_w, good_c)
        ... except ValueError:
        ...     print('bad weights')
        bad weights
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"S component weights must sum to 1.0 (got {total:.6f}).")
    if not contexts:
        raise ValueError("At least one context must be defined.")
    if any(not c.name for c in contexts):
        raise ValueError("Each context must have a non-empty name.")

