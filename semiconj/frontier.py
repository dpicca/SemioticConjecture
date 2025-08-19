from typing import Dict, List, Tuple


def estimate_frontier(S: List[float], D: List[float], n_bins: int = 6) -> List[Tuple[float, float]]:
    """Estimate a non-parametric frontier curve using rank bins and 95th percentiles.

    For each item i, compute x_i = S_i * D_i. Sort items by S_i and partition into
    ``n_bins`` rank-based bins. Within each bin, take the 95th percentile of x as
    the frontier value. The returned points are evenly spaced S-bin centers in [0,1].

    Args:
        S: List of S(M) scores in [0,1]. Must have the same length as D.
        D: List of decodability scores in [0,1]. Must have the same length as S.
        n_bins: Number of rank bins to produce along the S axis.

    Returns:
        List of (S_bin_center, k95) pairs where k95 is the 95th percentile of x=S·D
        within the corresponding bin.

    Raises:
        AssertionError: If ``len(S) != len(D)``.

    Examples:
        >>> S = [0.1, 0.2, 0.8, 0.9]
        >>> D = [0.5, 0.6, 0.7, 0.4]
        >>> pts = estimate_frontier(S, D, n_bins=2)
        >>> len(pts)
        2
        >>> all(0.0 <= s <= 1.0 and 0.0 <= k <= 1.0 for s, k in pts)
        True
    """
    assert len(S) == len(D)
    n = len(S)
    if n == 0:
        return []
    pairs = sorted((S[i], S[i]*D[i]) for i in range(n))
    bins: List[List[float]] = [[] for _ in range(n_bins)]
    edges = [i*(1.0/n_bins) for i in range(n_bins+1)]
    # Bin by rank of S
    for rank, (s, x) in enumerate(pairs):
        b = min(n_bins-1, int(rank / max(1, n//n_bins)))
        bins[b].append(x)
    def perc95(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        k = max(0, int(0.95 * (len(vals)-1)))
        return vals[k]
    frontier = []
    for b in range(n_bins):
        center = (b + 0.5) / n_bins
        frontier.append((center, perc95(bins[b])))
    return frontier

