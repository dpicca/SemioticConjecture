# Allow running as a script: `python semiconj/cli.py`
# When executed directly, Python sets sys.path to the module's directory (semiconj/),
# so absolute imports like `import semiconj.*` would fail because the parent directory
# is not on sys.path. The following block ensures the project root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from semiconj.config import DEFAULT_WEIGHTS, CONTEXTS, set_runtime_config, validate_config
from semiconj.corpus import read_corpus, validate_corpus, excerpts_to_rows, write_csv
from semiconj.decodability import aggregate_S, effective_decodability
from semiconj.metrics.complexity import tokenize, mtld, yules_k, pos_entropy, senses_per_lemma
from semiconj.metrics.entropy import ngram_entropy
from semiconj.metrics.embeddings import sentence_embedding_dispersion
from semiconj.metrics.figures import figures_score
from semiconj.metrics.intertextuality import domain_coverage_score, ner_coverage
from semiconj.metrics.codeswitch import codeswitch_index
from semiconj.semiotic.parser import compute_d_intr, split_sentences, words
from semiconj.surrogates.community import omega_and_rho
from semiconj.frontier import estimate_frontier
from semiconj.reporting import save_metrics
from semiconj.plotting import plot_frontier_legacy, plot_correlation_legacy, generate_all_plots

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(iterable=None, **kwargs):
        return iterable


def compute_S_components(text: str) -> Dict[str, float]:
    toks = tokenize(text)
    sents = [words(s) for s in split_sentences(text)]
    # semantics: embedding dispersion + domain coverage (avg)
    semantics = 0.5 * sentence_embedding_dispersion(sents) + 0.5 * domain_coverage_score(text)
    # intertextuality: NER coverage + domain coverage (avg)
    intertextuality = 0.5 * ner_coverage(text) + 0.5 * domain_coverage_score(text)
    # figures: metaphor/irony proxy
    figures = figures_score(text)
    # lexicon: MTLD (normalized) + Yule's K (already in [0,1]) + senses/lemma (normalized) + bigram entropy
    mtld_raw = mtld(toks)
    mtld_norm = max(0.0, min(1.0, mtld_raw / 100.0))
    yk = max(0.0, min(1.0, yules_k(toks)))
    senses = senses_per_lemma(toks)
    # Normalize senses: typical 1–5
    senses_norm = max(0.0, min(1.0, senses / 5.0))
    bigram_H = ngram_entropy(toks, n=2)
    lexicon = (mtld_norm + yk + senses_norm + bigram_H) / 4.0
    # pos entropy
    pos = pos_entropy(toks)
    # code-switching (favor presence but avoid too high)
    cs = codeswitch_index(text)
    # Map to dict
    return {
        "semantics": semantics,
        "intertextuality": intertextuality,
        "figures": figures,
        "lexicon": lexicon,
        "pos": pos,
        "codeswitch": cs,
    }


def run_pipeline(input_csv: Path, out_dir: Path, k: int = 12, surrogates: str = "ollama", ollama_models: str = "", ollama_host: str = "http://localhost:11434") -> None:
    logger.info("Starting pipeline: input=%s out=%s K=%d surrogates=%s", input_csv, out_dir, k, surrogates)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_corpus = read_corpus(input_csv)
    logger.info("Loaded corpus: %d rows", len(raw_corpus))
    corpus, report = validate_corpus(raw_corpus, min_words=10)
    logger.info("Validated corpus summary: kept=%d total=%d (dropped_missing=%d dropped_dupe=%d dropped_short=%d)",
                report.get("kept", 0), report.get("total", 0), report.get("dropped_missing", 0), report.get("dropped_dupe", 0), report.get("dropped_short", 0))
    # Write cleaned copy
    try:
        write_csv(out_dir / "cleaned_input.csv", excerpts_to_rows(corpus))
        logger.info("Wrote cleaned corpus to %s (%d rows)", out_dir / "cleaned_input.csv", len(corpus))
    except Exception as e:
        logging.warning(f"Failed to write cleaned_input.csv: {e}")
    texts = [e.text for e in corpus]
    # 1) S(M)
    logger.info("Computing S components for %d excerpts...", len(corpus))
    S_rows = []
    S_values: List[float] = []
    for e in _tqdm(corpus, desc="S components", total=len(corpus)):
        S_comp = compute_S_components(e.text)
        S_score = aggregate_S(S_comp, DEFAULT_WEIGHTS)
        S_values.append(S_score)
        S_rows.append({"id": e.id, **S_comp, "S": S_score})
    save_metrics(out_dir / "S_metrics.csv", S_rows)
    logger.info("Computed S for %d excerpts", len(S_rows))
    # 2) D_intr(M)
    logger.info("Computing D_intr for %d excerpts...", len(corpus))
    D_intr_rows = []
    D_intr_values: List[float] = []
    for e in _tqdm(corpus, desc="D_intr", total=len(corpus)):
        d = compute_d_intr(e.text)
        D_intr_values.append(d["D_intr"])
        D_intr_rows.append({"id": e.id, **d})
    save_metrics(out_dir / "D_intr.csv", D_intr_rows)
    logger.info("Computed D_intr for %d excerpts", len(D_intr_rows))
    # 3-4) Ω(H) and ρ(C)
    contexts = [c.name for c in CONTEXTS]
    if surrogates == "ollama":
        models = [m.strip() for m in ollama_models.split(',') if m.strip()]
        if not models:
            logging.info("No --ollama-models provided; defaulting to ['gpt-oss'].")
            models = ["gpt-oss"]
        logger.info("Estimating Ω and ρ using strategy=%s models=%s host=%s k=%d contexts=%s", surrogates, models, ollama_host, k, contexts)
    else:
        models = None
        logger.info("Estimating Ω and ρ using heuristic strategy k=%d contexts=%s", k, contexts)
    omega_by_ctx, rho_by_ctx = omega_and_rho(texts, contexts, k=k, strategy=surrogates, models=models, ollama_host=ollama_host)
    save_metrics(out_dir / "Omega.csv", [{"context": c, "Omega": v} for c, v in omega_by_ctx.items()])
    save_metrics(out_dir / "Rho.csv", [{"context": c, "Rho": v} for c, v in rho_by_ctx.items()])
    logger.info("Computed Ω and ρ for %d contexts", len(contexts))
    # 5) D(M;H,C) for each context (using corpus-mean Ω, ρ)
    logger.info("Computing D_effective for %d excerpts across %d contexts...", len(corpus), len(contexts))
    D_rows = []
    for c in _tqdm(contexts, desc="D_effective contexts", total=len(contexts)):
        omega = omega_by_ctx[c]
        rho = rho_by_ctx[c]
        for i, e in enumerate(_tqdm(corpus, desc=f"{c} items", total=len(corpus), leave=False)):
            D_eff = effective_decodability(D_intr_values[i], omega, rho)
            D_rows.append({"id": e.id, "context": c, "D": D_eff})
    save_metrics(out_dir / "D_effective.csv", D_rows)
    logger.info("Computed D_effective (%d rows)", len(D_rows))
    # 6) Frontier k(H,C) (using pooled D across contexts)
    # Aggregate D per id by mean across contexts
    from collections import defaultdict
    D_by_id = defaultdict(list)
    for r in D_rows:
        D_by_id[r["id"]].append(r["D"])
    D_mean = [sum(D_by_id[e.id])/len(D_by_id[e.id]) for e in corpus]
    logger.info("Estimating frontier from %d paired values...", len(D_mean))
    frontier_pts = estimate_frontier(S_values, D_mean)
    save_metrics(out_dir / "Frontier.csv", [{"S_bin": s, "k95": k} for s, k in frontier_pts])
    logger.info("Estimated frontier with %d points", len(frontier_pts))
    plot_frontier_legacy(out_dir / "frontier.png", frontier_pts)
    logger.info("Frontier plot saved to %s", out_dir / "frontier.png")
    # 7) Analyses (H1 example)
    logger.info("Running analyses (e.g., Kendall tau)...")

    from semiconj.analysis import kendall_tau
    tau = kendall_tau(S_values, D_intr_values)
    logger.info("Kendall_tau_S_vs_Dintr = %.6f", tau)
    save_metrics(out_dir / "Analyses_kendall_tau.csv", [{"metric": "Kendall_tau_S_vs_Dintr", "value": tau}])
    plot_correlation_legacy(out_dir / "kendall_tau_correlation.png", S_values, D_intr_values, tau, "S(M) - Semiotic Complexity", "D_intr(M) - Intrinsic Decodability")
    logger.info("Kendall tau correlation plot saved to %s", out_dir / "kendall_tau_correlation.png")

    from semiconj.analysis import spearman_rho
    spearman = spearman_rho(S_values, D_intr_values)
    logger.info("spearman_rho_S_vs_Dintr = %.6f", spearman)
    save_metrics(out_dir / "Analyses_spearman.csv", [{"metric": "spearman_S_vs_Dintr", "value": spearman}])
    logger.info("Spearman  correlation plot saved to %s", out_dir / "kendall_tau_correlation.png")

    # 8) Generate all comprehensive plots automatically
    logger.info("Generating comprehensive plots for all CSV files...")
    generate_all_plots(out_dir)
    logger.info("All plots generated successfully")
    logger.info("Pipeline finished. Outputs written to %s", out_dir)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Semiotic Conjecture — Plan A pipeline\n"
            "Optional dependencies: numpy/scipy (analytics), matplotlib (plots), nltk (WordNet).\n"
            "Heuristic mode works fully offline; Ollama mode requires a running server."
        ),
        epilog=(
            "Install extras via requirements-sci.txt / requirements-plot.txt / requirements-nlp.txt.\n"
            "Example: python -m semiconj.cli run --input data/sample_corpus.csv --out out --surrogates heuristic"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("run", nargs='?', help="Run the full pipeline")
    ap.add_argument("--input", type=Path, required=True, help="Input CSV corpus path")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--K", type=int, default=12, help="Number of surrogate interpreters")
    ap.add_argument("--surrogates", choices=["ollama", "heuristic"], default="ollama", help="Surrogate interpreter type (heuristic requires no network)")
    ap.add_argument("--ollama-models", type=str, default="", help="Comma-separated Ollama model names (>=3 families recommended)")
    ap.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama server host URL")
    ap.add_argument("--figures-ollama-model", type=str, default="gpt-oss", help="If set, compute figures_score using this Ollama model (expects JSON {\"score\": <0..1>})")
    ap.add_argument("--embeddings-ollama-model", type=str, default="nomic-embed-text:v1.5", help="If set, compute sentence embeddings via this Ollama model (e.g., 'nomic-embed-text')")
    ap.add_argument("--nlp-ollama-model", type=str, default="gpt-oss", help="If set, perform tokenization, sentence splitting, and NER via this Ollama model (e.g., 'gpt-oss')")
    # Config overrides
    ap.add_argument("--seed", type=int, default=42, help="Global random seed (affects stochastic components)")
    ap.add_argument("--embedding-dim", type=int, default=128, help="Default embedding dimension for hashing/averaging")
    ap.add_argument("--figures-mul", type=float, default=1.0, help="Multiplier applied to figures_score (tuning)")
    ap.add_argument("--pos-tagger", choices=["naive", "spacy"], default="naive", help="POS tagging backend (spaCy requires en_core_web_sm)")
    ap.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args()
    # Logging
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format="%(levelname)s %(message)s")
    # Validate and apply config
    validate_config(DEFAULT_WEIGHTS, CONTEXTS)
    set_runtime_config(
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        figures_multiplier=args.figures_mul,
        pos_tagger=args.pos_tagger,
        figures_ollama_model=args.figures_ollama_model,
        figures_ollama_host=args.ollama_host,
        embeddings_ollama_model=args.embeddings_ollama_model,
        embeddings_ollama_host=args.ollama_host,
        nlp_ollama_model=args.nlp_ollama_model,
        nlp_ollama_host=args.ollama_host,
    )
    logger.info(
        "Runtime config applied: seed=%d embedding_dim=%d pos_tagger=%s figures_mul=%.3f models(figures=%s, embeddings=%s, nlp=%s) host=%s",
        args.seed, args.embedding_dim, args.pos_tagger, args.figures_mul, args.figures_ollama_model, args.embeddings_ollama_model or "", args.nlp_ollama_model, args.ollama_host,
    )
    run_pipeline(args.input, args.out, k=args.K, surrogates=args.surrogates, ollama_models=args.ollama_models, ollama_host=args.ollama_host)


if __name__ == "__main__":
    main()
