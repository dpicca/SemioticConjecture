Semiotic Conjecture — Plan A (Surrogate Communities)

Overview
- Implements a reproducible pipeline to estimate:
  - S(M): structural/semantic complexity metrics
  - D_intr(M): intrinsic decodability via semiotic triads and coherence
  - Ω(H): surrogate community agreement and convergence
  - ρ(C): contextual constraint strength
  - D(M;H,C) = Ω · ρ · D_intr
  - Frontier k(H,C) using a non-parametric 95th-percentile curve

Highlights
- Pure-Python with graceful fallbacks (no internet needed to run toy demo).
- Optional dependencies are used if available (NLTK WordNet, scikit-learn, SciPy, matplotlib).
- CLI to run the full pipeline on a CSV corpus and produce CSV outputs.

Corpus Format
- CSV with at least:
  - id: unique identifier
  - text: raw text excerpt (150–300 words recommended)
  - Optional metadata columns prefixed with `meta_` (e.g., `meta_domain`, `meta_register`).

Quickstart
1) Place your corpus at `data/corpus.csv` or use the provided sample at `data/sample_corpus.csv`.
2) Ensure Ollama is installed and running locally (default host `http://localhost:11434`). Pull at least 3 model families (e.g., `llama3`, `mistral`, `qwen`, `phi3`).
3) Run the pipeline using Ollama-based surrogate interpreters (default):
   - `python -m semiconj.cli run --input data/sample_corpus.csv --out out --ollama-models "llama3,mistral,qwen" --K 12`
   - Use `--surrogates heuristic` only for offline dry-runs without Ollama (not recommended).
4) Inspect outputs in `out/` (CSV summaries; plots if matplotlib present).

Notes on Dependencies
- For Ollama mode, you need a running Ollama server and local models. The client uses the HTTP API (fallback to `ollama run` CLI if requests is unavailable).
- Heuristic mode requires no network.
- Optional libs improve metrics and analyses:
  - nltk (with wordnet data), numpy, scipy, matplotlib
- Installation options:
  - Core: `pip install -r requirements.txt`
  - Sci extras (numpy/scipy): `pip install -r requirements-sci.txt`
  - Plot extras (matplotlib): `pip install -r requirements-plot.txt`
  - NLP extras (nltk): `pip install -r requirements-nlp.txt`
  - Dev tools: `pip install -r requirements-dev.txt`
- Conda: `conda env create -f environment.yml && conda activate semiconj`

Method Mapping
- S(M): MTLD, Yule’s K, n-gram entropy, POS entropy, sentence embedding dispersion, multi-domain coverage, mean WordNet senses per lemma (if available), figurative language heuristics, NER coverage, code-switching.
- D_intr(M): semiotic triad extraction (rule-based heuristics), structural precision, convergence of interpretants (via cosine), global coherence (referential cohesion + inter-sentence similarity), intrinsic interpretive entropy (penalty term).
- Ω(H): K surrogate interpreters (different seeds/heuristics) produce labels and summaries; compute Fleiss’ κ and summary cosine convergence.
- ρ(C): variance reduction under contexts C_open, C_medium, C_rigid relative to C_open.
- Frontier: non-parametric 95th-percentile curve of x = S·D per S bin, with bootstrap-ready hooks.

Limitations and Extensibility
- The demo uses heuristics when advanced models are unavailable. Hooks are provided to plug in stronger taggers, NLI, encoders, and EL.
- See `semiconj/metrics/*` and `semiconj/semiotic/parser.py` to extend.

License
- This project is licensed under the MIT License. See the LICENSE file for details.


Semiotic Conjecture — Plan A (Surrogate Communities)

Overview
- Implements a reproducible pipeline to estimate:
  - S(M): structural/semantic complexity metrics
  - D_intr(M): intrinsic decodability via semiotic triads and coherence
  - Ω(H): surrogate community agreement and convergence
  - ρ(C): contextual constraint strength
  - D(M;H,C) = Ω · ρ · D_intr
  - Frontier k(H,C) using a non-parametric 95th-percentile curve

Highlights
- Pure-Python with graceful fallbacks (no internet needed to run toy demo).
- Optional dependencies are used if available (NLTK WordNet, scikit-learn, SciPy, matplotlib).
- CLI to run the full pipeline on a CSV corpus and produce CSV outputs.

Corpus Format
- CSV with at least:
  - id: unique identifier
  - text: raw text excerpt (150–300 words recommended)
  - Optional metadata columns prefixed with `meta_` (e.g., `meta_domain`, `meta_register`).

Quickstart
1) Place your corpus at `data/corpus.csv` or use the provided sample at `data/sample_corpus.csv`.
2) Ensure Ollama is installed and running locally (default host `http://localhost:11434`). Pull at least 3 model families (e.g., `llama3`, `mistral`, `qwen`, `phi3`).
3) Run the pipeline using Ollama-based surrogate interpreters (default):
   - `python -m semiconj.cli run --input data/sample_corpus.csv --out out --ollama-models "llama3,mistral,qwen" --K 12`
   - Use `--surrogates heuristic` only for offline dry-runs without Ollama (not recommended).
4) Inspect outputs in `out/` (CSV summaries; plots if matplotlib present).

Notes on Dependencies
- For Ollama mode, you need a running Ollama server and local models. The client uses the HTTP API (fallback to `ollama run` CLI if requests is unavailable).
- Heuristic mode requires no network.
- Optional libs improve metrics and analyses:
  - nltk (with wordnet data), numpy, scipy, matplotlib
- Installation options:
  - Core: `pip install -r requirements.txt`
  - Sci extras (numpy/scipy): `pip install -r requirements-sci.txt`
  - Plot extras (matplotlib): `pip install -r requirements-plot.txt`
  - NLP extras (nltk): `pip install -r requirements-nlp.txt`
  - Dev tools: `pip install -r requirements-dev.txt`
- Conda: `conda env create -f environment.yml && conda activate semiconj`

Method Mapping
- S(M): MTLD, Yule’s K, n-gram entropy, POS entropy, sentence embedding dispersion, multi-domain coverage, mean WordNet senses per lemma (if available), figurative language heuristics, NER coverage, code-switching.
- D_intr(M): semiotic triad extraction (rule-based heuristics), structural precision, convergence of interpretants (via cosine), global coherence (referential cohesion + inter-sentence similarity), intrinsic interpretive entropy (penalty term).
- Ω(H): K surrogate interpreters (different seeds/heuristics) produce labels and summaries; compute Fleiss’ κ and summary cosine convergence.
- ρ(C): variance reduction under contexts C_open, C_medium, C_rigid relative to C_open.
- Frontier: non-parametric 95th-percentile curve of x = S·D per S bin, with bootstrap-ready hooks.

Semiotic Conjecture (Statement and Variables)
- Hypothesis: every LLM output is a sign; interactions are a semiotic game. The balance between semiotic amplitude and decodability is bounded by a frontier:
  - X(M;H,C) = S(M) · D(M;H,C)
  - S(M) · D(M;H,C) ≤ k(H,C)
- Variable definitions and ranges:
  - S(M) ∈ [0,1]: semiotic amplitude (codes/strata/intertextuality variety)
  - D_intr(M) ∈ [0,1]: intrinsic decodability in ideal conditions
  - Ω(H) ∈ [0,1]: interpretive homogeneity of the community
  - ρ(C) ∈ [0,1]: rigidity/constraint of the context
  - D(M;H,C) = Ω(H) · ρ(C) · D_intr(M)
  - k(H,C): empirical frontier estimated from observed S·D values in the ecosystem

Operationalization (what this repo computes)
- S(M) composite (normalized [0,1]) using default weights from semiconj/config.py:
  - semantics 0.30; intertextuality 0.25; figures 0.15; lexicon 0.15; pos 0.10; codeswitch 0.05
  - Implemented by: semiconj.cli.compute_S_components; aggregated with semiconj.decodability.aggregate_S
  - semantics: sentence embedding dispersion and domain coverage (avg) — functions: semiconj.metrics.embeddings.sentence_embedding_dispersion; semiconj.metrics.intertextuality.domain_coverage_score
  - intertextuality: NER-like coverage and domain coverage (avg) — functions: semiconj.metrics.intertextuality.ner_coverage; semiconj.metrics.intertextuality.domain_coverage_score
  - figures: metaphor/irony proxy patterns — function: semiconj.metrics.figures.figures_score
  - lexicon: MTLD (normalized), Yule’s K (1/(1+K)), senses per lemma (WordNet if available; normalized), bigram entropy — functions: semiconj.metrics.complexity.mtld; semiconj.metrics.complexity.yules_k; semiconj.metrics.complexity.senses_per_lemma; semiconj.metrics.entropy.ngram_entropy
  - pos: POS-tag distribution entropy (UPOS-normalized) — function: semiconj.metrics.complexity.pos_entropy
  - codeswitch: simple code-switching index — function: semiconj.metrics.codeswitch.codeswitch_index
- D_intr(M): semiotic parser identifies triads (sign, object, interpretant); components:
  - structural precision (triads per sentence), convergence (mean cosine among interpretants), coherence (1 − dispersion across sentences), interpretive entropy (penalty). Aggregated to D_intr ∈ [0,1]. Implemented by: semiconj.semiotic.parser.compute_d_intr
- Ω(H) and ρ(C):
  - Ω(H): Fleiss’ κ over categorical labels and cosine convergence of summaries, mapped to [0,1]. Implemented by: semiconj.surrogates.community.omega_and_rho (uses internal fleiss_kappa and cosine)
  - ρ(C): 1 − var_H(summaries|C)/var_H(summaries|C_open), clipped to [0,1]. Contexts used: C_open, C_medium, C_rigid. Implemented by: semiconj.surrogates.community.omega_and_rho
- Frontier k(H,C): non-parametric estimation via the 95th percentile of X = S·D within rank-based S bins. Implemented by: semiconj.frontier.estimate_frontier

Two Protocols for Measurement
- Protocol A (no human annotators; LLM/heuristic surrogates):
  - Generate or supply an LLM corpus; compute S(M).
  - Estimate D_intr(M) with the semiotic parser and coherence metrics.
  - Build surrogate communities (heuristic or Ollama-backed models) to estimate Ω(H).
  - Manipulate context (C_open, C_medium, C_rigid) to estimate ρ(C).
  - Compute D = Ω·ρ·D_intr, then estimate k(H,C) from x = S·D across items.
- Protocol B (with a small human panel, 4–8 annotators):
  - Train annotators; collect semiotic labels/summaries.
  - Compute agreement indices (Krippendorff’s α, Fleiss’ κ) and semantic convergence to estimate Ω(H).
  - Present the same texts under varying context constraints to estimate ρ(C).
  - Compute D and estimate k(H,C); validate with test–retest and bootstrap.

Outputs and Schemas
- The CLI writes the following CSVs to the output directory (see [Output schema](doc/docs/output_schema.md) for columns):
  - cleaned_input.csv, S_metrics.csv, D_intr.csv, Omega.csv, Rho.csv, D_effective.csv, Frontier.csv, Analyses.csv

Example Calculation
- If S = 0.72, D_intr = 0.65, Ω = 0.60, ρ = 0.75:
  - D = 0.60 × 0.75 × 0.65 = 0.2925
  - X = S · D ≈ 0.2106
  - Compare X to the estimated k(H,C) (e.g., 95th-percentile frontier) to assess proximity to the ecosystem frontier.

Stress Tests and Expectations
- Expect S↑ to correlate with greater interpretive dispersion (lower D) on extreme texts (e.g., experimental poetry vs technical manuals). The provided analyses include a Kendall’s tau check between S and D_intr.

Limitations and Extensibility
- The demo uses heuristics when advanced models are unavailable. Hooks are provided to plug in stronger taggers, NLI, encoders, and entity linking.
- See `semiconj/metrics/*` and `semiconj/semiotic/parser.py` to extend.

License
- This project is licensed under the MIT License. See the LICENSE file for details.
