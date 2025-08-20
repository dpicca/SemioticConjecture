from typing import Dict


def effective_decodability(D_intr: float, omega: float, rho: float) -> float:
    """Compute effective decodability D given intrinsic decodability and community/context factors.

    This clips the product to the [0, 1] interval to ensure a bounded result.

    Args:
        D_intr: Intrinsic decodability in [0, 1].
        omega: Community homogeneity/agreement in [0, 1].
        rho: Context rigidity/constraint in [0, 1].

    Returns:
        A float in [0, 1] equal to ``clip(D_intr * omega * rho)``.

    Examples:
        >>> effective_decodability(0.6, 0.8, 0.5)
        0.24
        >>> effective_decodability(2.0, 0.5, 0.5)  # clipped to 1.0 then multiplied
        0.25
    """
    return max(0.0, min(1.0, float(D_intr) * float(omega) * float(rho)))


def aggregate_S(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Aggregate S(M) components with weights and clip to [0, 1].

    The metrics dict should contain at least the components present in ``weights``
    (typically: ``semantics``, ``intertextuality``, ``figures``, ``lexicon``,
    ``pos``, ``codeswitch``). Each metric is individually clipped to [0,1]
    before weighting and summation; the final score is clipped to [0,1].

    Args:
        metrics: Mapping from component name to score (expected in [0,1]).
        weights: Mapping from component name to non-negative weights that sum to 1.

    Returns:
        Aggregated S score in [0, 1].

    Raises:
        KeyError: If any required component present in ``weights`` is missing from ``metrics``.

    Examples:
        >>> metrics = {
        ...     'semantics': 0.7, 'intertextuality': 0.6, 'figures': 0.3,
        ...     'lexicon': 0.5, 'pos': 0.4, 'codeswitch': 0.1
        ... }
        >>> weights = {
        ...     'semantics': 0.30, 'intertextuality': 0.25, 'figures': 0.15,
        ...     'lexicon': 0.15, 'pos': 0.10, 'codeswitch': 0.05
        ... }
        >>> round(aggregate_S(metrics, weights), 6)  # doctest: +ELLIPSIS
        0.5...
    """
    missing = [k for k in weights.keys() if k not in metrics]
    if missing:
        raise KeyError(f"Missing required S components: {missing}")
    score = 0.0
    for k, w in weights.items():
        v = float(metrics[k])
        score += w * max(0.0, min(1.0, v))
    return max(0.0, min(1.0, score))

