import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..metrics.complexity import tokenize
from ..metrics.embeddings import _avg_vec as avg_vec, _cosine as cosine

# optional tqdm progress bar
try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(iterable=None, **kwargs):
        return iterable


LABELS = ["technology", "biology", "art", "politics", "finance", "sports", "general"]


def label_text(text: str) -> str:
    """Heuristic domain labeler using keyword matching.

    Args:
        text: Input text to classify.

    Returns:
        One of LABELS: 'technology', 'biology', 'art', 'politics', 'finance', 'sports', or 'general'.

    Examples:
        >>> label_text('The algorithm optimizes network throughput.')
        'technology'
        >>> label_text('A canvas and painting techniques are discussed.')
        'art'
    """
    t = text.lower()
    # lightweight keyword-based labeler
    keywords = {
        "technology": ["algorithm", "software", "network", "data", "model", "system"],
        "biology": ["cell", "species", "genome", "protein", "organism"],
        "art": ["painting", "poetry", "novel", "canvas", "aesthetic"],
        "politics": ["election", "policy", "state", "government", "vote"],
        "finance": ["market", "risk", "equity", "capital", "investment"],
        "sports": ["match", "score", "team", "league", "coach"],
    }
    for lab, ks in keywords.items():
        if any(k in t for k in ks):
            return lab
    return "general"


def summarize(text: str, n_sentences: int = 1, seed: int = 0) -> str:
    """Extractive summary by selecting top-scoring sentences.

    Sentences are scored by the average term frequency of their unique tokens,
    with tiny random tie-breakers controlled by ``seed``.

    Args:
        text: Input text to summarize.
        n_sentences: Number of sentences to include in the summary (>=1).
        seed: Random seed for tie-breaking.

    Returns:
        A string containing the selected sentence(s), joined by '. '. If the text
        has no sentence-like delimiters, returns the stripped input.

    Examples:
        >>> summarize('A. B. C.', n_sentences=2, seed=0)
        'A. B'
    """
    # Simple extractive summary: pick highest-average TF word sentences, randomized tie-breakers
    sents = [s for s in text.split('.') if s.strip()]
    if not sents:
        return text.strip()
    toks = [tokenize(s) for s in sents]
    tf = Counter([w for ts in toks for w in ts])
    scores = []
    rnd = random.Random(seed)
    for i, ts in enumerate(toks):
        score = sum(tf[w] for w in set(ts)) / max(1, len(set(ts)))
        score += 1e-3 * rnd.random()
        scores.append((score, i))
    scores.sort(reverse=True)
    idxs = sorted(i for _, i in scores[:n_sentences])
    return '. '.join(sents[i].strip() for i in idxs)


@dataclass
class Interpreter:
    name: str
    seed: int
    temperature: float
    rigidity: float  # 0=open, 0.5=medium, 1=rigid

    def produce(self, text: str, context: str) -> Tuple[str, str]:
        # Context affects rigidity (more rule-like summary and label bias)
        if context == "C_open":
            self.rigidity = 0.0
        elif context == "C_medium":
            self.rigidity = 0.5
        else:
            self.rigidity = 1.0
        # Labeling with small bias for rigid contexts
        lab = label_text(text)
        if self.rigidity > 0.7:
            # encourage specific labels over 'general'
            if lab == 'general':
                lab = 'technology'
        # Summarization: more constrained under rigid (short, definition-like)
        rnd = random.Random(self.seed)
        ns = 1 if self.rigidity >= 0.5 else 2
        summ = summarize(text, n_sentences=ns, seed=rnd.randint(0, 10**6))
        if self.rigidity >= 0.5:
            summ = summ.split('.')[0].strip()
        return lab, summ


def build_community(k: int = 12) -> List[Interpreter]:
    interps = []
    for i in range(k):
        interps.append(Interpreter(name=f"H{i:02d}", seed=1337 + i * 17, temperature=0.5 + 0.05 * (i % 3), rigidity=0.0))
    return interps


def vec(text: str) -> List[float]:
    return avg_vec(tokenize(text))




def fleiss_kappa(assignments: List[List[str]]) -> float:
    """Compute Fleiss' kappa for categorical labels.
    assignments: list of items, each is list of labels from raters.
    """
    if not assignments:
        return 0.0
    cats = sorted(set(c for row in assignments for c in row))
    cat_idx = {c: i for i, c in enumerate(cats)}
    n_cat = len(cats)
    n = len(assignments)
    m = len(assignments[0]) if assignments[0] else 0
    if m == 0:
        return 0.0
    # count matrix n x k
    nij = [[0] * n_cat for _ in range(n)]
    for i, row in enumerate(assignments):
        for lab in row:
            nij[i][cat_idx[lab]] += 1
    pj = [sum(nij[i][j] for i in range(n)) / (n * m) for j in range(n_cat)]
    Pi = [
        (sum(nij[i][j] * nij[i][j] for j in range(n_cat)) - m) / (m * (m - 1) + 1e-9)
        for i in range(n)
    ]
    Pbar = sum(Pi) / n
    Pe = sum(p * p for p in pj)
    if Pe >= 1.0:
        return 0.0
    return (Pbar - Pe) / (1.0 - Pe + 1e-9)


def omega_and_rho(texts: List[str], contexts: List[str], k: int = 12, strategy: str = "ollama", models: List[str] = None, ollama_host: str = "http://localhost:11434") -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute Ω(H) and ρ(C) across contexts for given texts.
    Returns (Omega_by_context, Rho_by_context).
    """
    if strategy == "ollama":
        if not models:
            raise ValueError("In 'ollama' mode, a non-empty list of models must be provided.")
        comm = build_ollama_community(models=models, k=k, host=ollama_host)
    else:
        comm = build_community(k)
    # Collect labels and summary vectors
    per_context_labels: Dict[str, List[List[str]]] = {c: [] for c in contexts}
    per_context_vecs: Dict[str, List[List[List[float]]]] = {c: [] for c in contexts}
    for c in _tqdm(contexts, desc="Ω/ρ contexts", total=len(contexts)):
        for t in _tqdm(texts, desc=f"{c} items", total=len(texts), leave=False):
            labs = []
            vecs = []
            for h in comm:
                lab, summ = h.produce(t, c)
                labs.append(lab)
                vecs.append(vec(summ))
            per_context_labels[c].append(labs)
            per_context_vecs[c].append(vecs)
    # Ω(H): agreement + convergence mapped to [0,1]
    omega: Dict[str, float] = {}
    for c in contexts:
        kappa = fleiss_kappa(per_context_labels[c])
        # summary convergence: mean pairwise cosine among raters, averaged over items
        sims = []
        for item_vecs in per_context_vecs[c]:
            m = len(item_vecs)
            if m < 2:
                continue
            item_sims = []
            for i in range(m):
                for j in range(i+1, m):
                    item_sims.append(cosine(item_vecs[i], item_vecs[j]))
            if item_sims:
                sims.append(sum(item_sims) / len(item_sims))
        conv = sum(sims) / len(sims) if sims else 0.0
        # map to [0,1]
        omega[c] = max(0.0, min(1.0, 0.5 * (kappa + conv)))
    # ρ(C): 1 - var_H(C) / var_H(C_open)
    # Measure variance of summary vectors across H, averaged across items
    def variance_of_vectors(vecs: List[List[float]]) -> float:
        if not vecs:
            return 0.0
        d = len(vecs[0])
        # component-wise variance average
        means = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
        var = sum(sum((v[i] - means[i]) ** 2 for v in vecs) / len(vecs) for i in range(d)) / d
        return var
    var_by_context = {}
    for c in contexts:
        item_vars = [variance_of_vectors(vv) for vv in per_context_vecs[c]]
        var_by_context[c] = sum(item_vars) / len(item_vars) if item_vars else 0.0
    if "C_open" not in var_by_context:
        raise ValueError("Context 'C_open' is required to compute rho baseline.")
    base = var_by_context["C_open"]
    if base <= 0.0:
        # Degenerate case: no variance across H in the open context; define rho as 0 for all contexts
        rho = {c: 0.0 for c in contexts}
        return omega, rho
    rho: Dict[str, float] = {}
    for c in contexts:
        rho[c] = max(0.0, min(1.0, 1.0 - (var_by_context[c] / base)))
    return omega, rho


# --------------------------- Ollama-backed community ---------------------------

class OllamaInterpreter(Interpreter):
    def __init__(self, name: str, seed: int, temperature: float, model: str, host: str):
        super().__init__(name=name, seed=seed, temperature=temperature, rigidity=0.0)
        self.model = model
        self.host = host

    def produce(self, text: str, context: str) -> Tuple[str, str]:
        try:
            from .ollama_client import OllamaClient, build_prompt
        except Exception as e:
            raise RuntimeError("Ollama client not available. Ensure dependencies and Ollama are installed.") from e
        if context == "C_open":
            self.rigidity = 0.0
        elif context == "C_medium":
            self.rigidity = 0.5
        else:
            self.rigidity = 1.0
        client = OllamaClient(self.host)
        prompt, system = build_prompt(text=text, context=context, labels=LABELS, rigidity=self.rigidity)
        out = client.generate(model=self.model, prompt=prompt, system=system, temperature=self.temperature, seed=self.seed)
        label, summary = client.parse_label_summary(out, allowed_labels=LABELS)
        return label, summary


def build_ollama_community(models: List[str], k: int, host: str) -> List[OllamaInterpreter]:
    if not models:
        raise ValueError("build_ollama_community requires a non-empty list of models.")
    interps: List[OllamaInterpreter] = []
    # Distribute K across models with varied seeds/temps
    per_model = max(1, k // len(models))
    temps = [0.2, 0.5, 0.8]
    idx = 0
    for m in models:
        for r in range(per_model):
            t = temps[r % len(temps)]
            interps.append(OllamaInterpreter(name=f"{m}-H{idx:02d}", seed=1234 + 37 * idx, temperature=t, model=m, host=host))
            idx += 1
            if idx >= k:
                break
        if idx >= k:
            break
    # If we have fewer than K due to rounding, fill round-robin
    i = 0
    while len(interps) < k:
        m = models[i % len(models)]
        interps.append(OllamaInterpreter(name=f"{m}-H{idx:02d}", seed=1234 + 37 * idx, temperature=temps[idx % len(temps)], model=m, host=host))
        idx += 1
        i += 1
    return interps
