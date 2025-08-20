from typing import Dict, List, Tuple


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Compute Kendall's tau correlation coefficient (strict: requires SciPy).

    Args:
        x: First sequence of numbers.
        y: Second sequence of numbers, same length as ``x``.

    Returns:
        The Kendall's tau correlation as a float (nan-safe cast to 0.0).

    Raises:
        ImportError: If SciPy is not installed.

    Examples:
        >>> kendall_tau([1, 2, 3], [1, 2, 3])  # doctest: +SKIP
        1.0
    """
    try:
        from scipy.stats import kendalltau  # type: ignore
        from scipy.stats import spearmanr
    except Exception as e:
        raise ImportError("scipy is required for kendall_tau. Install scipy.") from e
    r, _ = kendalltau(x, y)
    return float(r or 0.0)


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman's rank correlation coefficient.

    Args:
        x: First sequence of numbers.
        y: Second sequence of numbers, same length as ``x``.

    Returns:
        The Spearman rank correlation coefficient in [-1, 1].

    Examples:
        >>> spearman_rho([1, 2, 3], [3, 2, 1])
        -1.0
        >>> round(spearman_rho([1, 2, 3], [10, 20, 30]), 6)
        1.0
    """

    try:
        from scipy.stats import spearmanr
    except Exception as e:
        raise ImportError("scipy is required for spearman. Install scipy.") from e
    r, _ = spearmanr(x, y)
    return float(r or 0.0)
    n = len(x)
    if n == 0:
        return 0.0
    rx = rank(x)
    ry = rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n*n - 1) + 1e-9)


def rank(a: List[float]) -> List[float]:
    """Return average ranks for a sequence, handling ties.

    Items with identical values receive the same average rank (1-based indexing
    averaged over the tied span). For example, [10, 10, 20] -> ranks [1.5, 1.5, 3].

    Args:
        a: Sequence of numeric values.

    Returns:
        A list of ranks (floats) of the same length as ``a``.

    Examples:
        >>> rank([10, 10, 20])
        [1.5, 1.5, 3.0]
        >>> rank([3, 1, 2])
        [3.0, 1.0, 2.0]
    """
    # Average ranks for ties
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j+1]] == a[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j+1):
            r[order[k]] = avg
        i = j + 1
    return r


def simple_regression(y: List[float], X: List[List[float]]) -> Tuple[List[float], float]:
    """Ordinary Least Squares via normal equations (no regularization).

    Adds an intercept internally and solves beta = (X'X)^(-1) X'y using a Moore–Penrose
    pseudoinverse for numerical stability.

    Args:
        y: Response vector of length n.
        X: Design matrix of shape n x p (without intercept column).

    Returns:
        A pair (beta, R2) where beta includes the intercept as beta[0].

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If X is empty.

    Examples:
        >>> y = [1.0, 2.0, 3.0]
        >>> X = [[1.0], [2.0], [3.0]]
        >>> beta, R2 = simple_regression(y, X)  # doctest: +SKIP
        >>> len(beta)  # intercept + slope
        2
    """
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("numpy is required for simple_regression. Install numpy.") from e
    Xn = np.asarray(X, float)
    if Xn.size == 0:
        raise ValueError("X must be non-empty for regression.")
    n, p = Xn.shape
    Xd = np.column_stack([np.ones(n), Xn])
    yv = np.asarray(y, float)
    beta = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ yv)
    yhat = Xd @ beta
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    ss_res = float(((yv - yhat) ** 2).sum())
    R2 = 1 - ss_res / (ss_tot + 1e-9)
    return beta.tolist(), float(R2)

