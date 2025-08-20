from pathlib import Path
from typing import Dict, Iterable, List
import logging

from ..corpus import write_csv

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



