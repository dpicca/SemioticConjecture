import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class Excerpt:
    """A single corpus record with text and metadata.

    Attributes:
        id: Unique identifier of the record (stringified).
        text: The raw text excerpt.
        meta: Arbitrary metadata as a mapping from column name to value.

    Examples:
        >>> Excerpt(id="001", text="Hello world.", meta={"meta_domain": "news"})
        Excerpt(id='001', text='Hello world.', meta={'meta_domain': 'news'})
    """
    id: str
    text: str
    meta: Dict[str, str]


def read_corpus(csv_path: Path) -> List[Excerpt]:
    """Read a CSV corpus into a list of Excerpt records.

    The CSV must contain at least the columns ``id`` and ``text``. Any other
    columns are stored as metadata in the ``meta`` dict.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        A list of Excerpt instances parsed from the CSV.

    Raises:
        ValueError: If required columns are missing.

    Examples:
        >>> from pathlib import Path
        >>> rows = read_corpus(Path('data/sample_corpus.csv'))  # doctest: +SKIP
        >>> isinstance(rows, list)  # doctest: +SKIP
        True
    """
    rows: List[Excerpt] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'id' not in reader.fieldnames or 'text' not in reader.fieldnames:
            raise ValueError("CSV must contain 'id' and 'text' columns")
        for r in reader:
            meta = {k: (v if v is not None else '') for k, v in r.items() if k not in {"id", "text"}}
            rows.append(Excerpt(id=str(r.get('id', '')).strip(), text=str(r.get('text', '') or '').strip(), meta=meta))
    return rows


def validate_corpus(excerpts: List[Excerpt], min_words: int = 10) -> Tuple[List[Excerpt], Dict[str, int]]:
    """Validate a corpus and return cleaned excerpts plus a summary report.

    Rules applied in order:
    - Drop rows with missing/empty id or text.
    - Drop duplicate ids (keep the first occurrence).
    - Drop extremely short texts (fewer than ``min_words`` tokens).

    Args:
        excerpts: List of Excerpt instances to validate.
        min_words: Minimum whitespace-separated token count required to keep a text.

    Returns:
        A pair ``(cleaned, report)`` where ``cleaned`` is the filtered list of excerpts
        and ``report`` is a dict with counts: ``total``, ``dropped_missing``, ``dropped_dupe``,
        ``dropped_short``, ``kept``.

    Examples:
        >>> ex = [
        ...     Excerpt('1', 'too short', {}),
        ...     Excerpt('1', 'duplicate id', {}),
        ...     Excerpt('2', 'this is long enough text for validation', {}),
        ... ]
        >>> cleaned, report = validate_corpus(ex, min_words=3)
        >>> [e.id for e in cleaned]
        ['2']
    """
    report = {"total": len(excerpts), "dropped_missing": 0, "dropped_dupe": 0, "dropped_short": 0, "kept": 0}
    seen: Set[str] = set()
    cleaned: List[Excerpt] = []
    for ex in excerpts:
        if not ex.id or not ex.text:
            report["dropped_missing"] += 1
            continue
        if ex.id in seen:
            report["dropped_dupe"] += 1
            continue
        if len(ex.text.split()) < max(0, min_words):
            report["dropped_short"] += 1
            continue
        seen.add(ex.id)
        cleaned.append(ex)
    report["kept"] = len(cleaned)
    logger.info(
        "Validated corpus: total=%d kept=%d dropped_missing=%d dropped_dupe=%d dropped_short=%d",
        report["total"], report["kept"], report["dropped_missing"], report["dropped_dupe"], report["dropped_short"],
    )
    return cleaned, report


def excerpts_to_rows(excerpts: List[Excerpt]) -> List[Dict[str, object]]:
    """Convert excerpts into CSV-ready row dicts, preserving meta_* keys.

    For each Excerpt, this function emits a dict with at least ``id`` and ``text``
    fields. Any metadata keys not starting with ``meta_`` are prefixed accordingly.

    Args:
        excerpts: List of Excerpt objects.

    Returns:
        A list of dictionaries suitable for CSV writing with ``write_csv``.

    Examples:
        >>> ex = [Excerpt('1', 'Hello', {'domain': 'news'})]
        >>> rows = excerpts_to_rows(ex)
        >>> rows[0]['id'], rows[0]['text'], rows[0]['meta_domain']
        ('1', 'Hello', 'news')
    """
    rows: List[Dict[str, object]] = []
    for ex in excerpts:
        row: Dict[str, object] = {"id": ex.id, "text": ex.text}
        for k, v in ex.meta.items():
            col = k if k.startswith("meta_") else f"meta_{k}"
            row[col] = v
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Write dictionaries to CSV, preserving all observed columns.

    This function computes the union of keys across all rows to avoid dropping
    sparse columns. Floats are formatted to 6 decimals to ensure consistency.

    Args:
        path: Destination CSV path.
        rows: Iterable of mapping objects to serialize.

    Examples:
        >>> from pathlib import Path
        >>> rows = [{'id': '1', 'S': 0.5}, {'id': '2', 'S': 0.75, 'extra': 'x'}]
        >>> write_csv(Path('tmp.csv'), rows)  # doctest: +SKIP
        # Creates tmp.csv with header: id,S,extra (order preserved by first occurrence)
    """
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Determine union of fieldnames across rows to avoid dropping columns
    fieldnames: List[str] = []
    seen_fields: Set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen_fields:
                seen_fields.add(k)
                fieldnames.append(k)
    def _format(v: object) -> object:
        # Ensure numeric formatting consistency
        if isinstance(v, float):
            return f"{v:.6f}"
        return v
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction='ignore',
            restval='',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator='\n',
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _format(r.get(k, '')) for k in fieldnames})

