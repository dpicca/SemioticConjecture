from pathlib import Path
from typing import Dict, Iterable, List
import logging

from .corpus import write_csv

logger = logging.getLogger(__name__)


def save_metrics(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Save a sequence of metric rows to CSV.

    This is a thin wrapper over ``semiconj.corpus.write_csv`` that preserves the
    union of keys across rows and formats floats to 6 decimals.

    Args:
        path: Output CSV path.
        rows: Iterable of dictionaries representing records to write.

    Examples:
        >>> from pathlib import Path
        >>> data = [{'id': 'a1', 'S': 0.5}, {'id': 'a2', 'S': 0.7}]
        >>> save_metrics(Path('out.csv'), data)  # doctest: +SKIP
        # File 'out.csv' will contain a header and two rows.
    """
    rows_list = list(rows)
    write_csv(path, rows_list)
    logger.info("Saved metrics to %s (%d rows)", path, len(rows_list))


def maybe_plot_frontier(path: Path, frontier_points: List[tuple]) -> None:
    """Plot the estimated frontier points and save the figure.

    Args:
        path: Output PNG path.
        frontier_points: Sequence of (S_bin_center, k95) pairs as returned by
            ``semiconj.frontier.estimate_frontier``.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If ``frontier_points`` is empty.

    Examples:
        >>> from pathlib import Path
        >>> pts = [(0.1, 0.05), (0.5, 0.2), (0.9, 0.3)]
        >>> maybe_plot_frontier(Path('frontier.png'), pts)  # doctest: +SKIP
        # Saves a simple line plot to frontier.png
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError("matplotlib is required to plot the frontier. Install matplotlib.") from e
    if not frontier_points:
        raise ValueError("Frontier points are required to plot the frontier (no fallback plotting).")
    xs = [s for s, _ in frontier_points]
    ys = [y for _, y in frontier_points]
    plt.figure(figsize=(5,3))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('S (binned)')
    plt.ylabel('95th percentile of S·D')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)

