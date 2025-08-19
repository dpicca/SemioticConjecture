import math
from collections import Counter
from typing import Iterable, List, Tuple


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    """Yield consecutive n-grams from a list of tokens.

    Args:
        tokens: Sequence of token strings.
        n: Size of the n-gram (n >= 1).

    Yields:
        Tuples of length ``n`` representing each n-gram.

    Examples:
        >>> list(ngrams(["a", "b", "c"], 2))
        [('a', 'b'), ('b', 'c')]
    """
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])


def entropy_from_counts(counts: Counter) -> float:
    """Compute normalized Shannon entropy from a Counter of event counts.

    The entropy is computed in bits and normalized by log2(K), where K is the
    number of unique events, yielding a value in [0,1].

    Args:
        counts: Counter mapping events to their frequencies.

    Returns:
        Normalized entropy in [0, 1]. Returns 0.0 for degenerate distributions.

    Examples:
        >>> from collections import Counter
        >>> round(entropy_from_counts(Counter({'a': 1, 'b': 1})), 6)
        1.0
        >>> round(entropy_from_counts(Counter({'a': 2, 'b': 0})), 6)
        0.0
    """
    total = sum(counts.values()) or 1
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p + 1e-12, 2)
    max_H = math.log(max(1, len(counts)), 2)
    if max_H == 0:
        return 0.0
    return H / max_H


def ngram_entropy(tokens: List[str], n: int = 1) -> float:
    """Compute normalized entropy over token n-grams.

    Args:
        tokens: List of tokens.
        n: N-gram size (default 1 for unigrams).

    Returns:
        Normalized Shannon entropy of the n-gram distribution in [0,1].

    Examples:
        >>> round(ngram_entropy(["a", "b", "a", "b"], n=1), 6)
        1.0
        >>> round(ngram_entropy(["a", "a", "a"], n=1), 6)
        0.0
    """
    if len(tokens) < n:
        return 0.0
    return entropy_from_counts(Counter(ngrams(tokens, n)))

