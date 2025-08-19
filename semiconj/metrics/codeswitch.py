import re
from collections import Counter
from typing import List


STOPWORDS = {
    "en": {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with"},
    "es": {"el", "la", "los", "las", "y", "o", "pero", "de", "a", "en", "para", "con"},
    "fr": {"le", "la", "les", "et", "ou", "mais", "de", "à", "en", "pour", "avec"},
    "it": {"il", "la", "i", "gli", "e", "o", "ma", "di", "a", "in", "per", "con"},
}


def codeswitch_index(text: str) -> float:
    """Fraction of tokens that appear to be from a non-dominant language.
    Heuristic using stopword overlaps across a few languages.
    """
    tokens = [t.lower() for t in re.findall(r"[\w']+", text)]
    if not tokens:
        return 0.0
    counts = {lang: 0 for lang in STOPWORDS}
    for t in tokens:
        for lang, sw in STOPWORDS.items():
            if t in sw:
                counts[lang] += 1
    dominant = max(counts, key=counts.get)
    dom_words = STOPWORDS[dominant]
    non_dom = sum(1 for t in tokens if any((t in STOPWORDS[l]) for l in STOPWORDS if l != dominant))
    return min(1.0, non_dom / max(1, len(tokens)))

