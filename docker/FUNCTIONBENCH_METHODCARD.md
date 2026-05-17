# FunctionBench method card: PROTEA

> Paste this text into the FunctionBench submission form at
> https://functionbench.net/ when registering the method. Override any
> field the form prescribes (short id, image tag) with the values shown
> here unless the user has reason to deviate.

## Identification

| Field                | Value                                                       |
|----------------------|-------------------------------------------------------------|
| Method name          | PROTEA (PROtein functional Embedding-based Annotation)      |
| Short id             | `protea-knn-rrk`                                            |
| Version              | `v0.1.0`                                                    |
| DockerHub image      | `frapercan/protea-method:v0.1.0`                            |
| Source code          | https://github.com/frapercan/protea-method                  |
| License              | MIT                                                         |
| Contact              | Francisco Miguel Pérez Canales (frapercan1@gmail.com)       |

Pin the DockerHub digest at submission time (`docker pull` then
`docker image inspect ... --format '{{index .RepoDigests 0}}'`) and paste
the `sha256:...` into the form so the grader resolves to a single
immutable image.

## Authors and affiliation

* Francisco Miguel Pérez Canales (author and sole maintainer of PROTEA)
* Co-supervisors: David Orellana-Martín, Ana M. Rojas

Doctoral thesis project at the University of Seville. PROTEA is the
post-CAFA productisation of the team submission to CAFA 6 (team result,
rank 2). The author of this submission was the technical motor of that
team effort.

## Method description

PROTEA assigns Gene Ontology (GO) terms to query proteins by combining a
Protein Language Model (PLM) embedding step with a KNN retrieval over an
experimentally-annotated reference set, then re-ranks the propagated
candidates with per-aspect LightGBM rerankers.

**1. PLM embedding.** Each FASTA sequence is embedded with ESM-2 3B
(`esm2_t36_3B`, the LB.2 v226 champion configuration). The 2560-dim
mean-pooled residue embedding is the working representation. The image
also accepts ESM-2 650M (`esm2_t33_650M`) and ProstT5
(`prost_t5_xl_uniref50`); switching backends is a one-flag change at
inference time. Computed embeddings are cached on disk keyed by
`(backend_id, fasta_sha256)` so re-runs on the same FASTA skip the
forward pass.

**2. KNN retrieval.** For each query, the top-k=5 cosine-nearest
reference proteins are retrieved (configurable). The default numpy
chunked brute-force backend covers the FunctionBench reference scale;
a FAISS IVFFlat backend is available behind `--backend faiss` for very
large reference sets. Per the PROTEA hard constraint, no pgvector / no
ANN index that requires a database round-trip is used.

**3. Candidate aggregation and feature compute.** Annotations of the k
neighbours are unioned into a candidate GO term set. The feature
enricher computes per-(query, candidate) features: KNN distance
statistics, neighbour vote fraction, taxonomic lineage features, and
DAG-relation features. The leakage-fixed feature set (LB.2, 2026-05-17)
drops the anc2vec ancestor-embedding family and the PCA-of-embeddings
family that were found to leak label information at training time.

**4. Per-aspect rerank.** Three LightGBM boosters (one per GO aspect:
Molecular Function, Biological Process, Cellular Component) score the
candidates with `predict(raw_score=False)`. Selective rerank is applied
on the NK and LK eval cells; PK falls back to the KNN-baseline score
because the reranker did not consistently lift on the PK band during
LB.2 sweeps.

**5. GO ancestor propagation.** Each scored candidate is propagated up
the GO DAG using the OBO `is_a` and `part_of` edges. The maximum
descendant score is assigned to every ancestor (CAFA-style propagation,
`prop=fill`). The result is the union of leaf scores and propagated
ancestor scores per query.

## Validation results

Numbers below are from the LB.2 leakage-fixed multi-seed sweep on the
`bench-v1-K5-v226-lineage` benchmark (validation range v226, which falls
inside the LAFA eval band v226 to v230). Three seeds (42, 7, 137), six
cells (NK and LK across the three aspects), `cafaeval_fmax` with
`prop=fill, norm=cafa`.

| Slice                                              | Mean Fmax    | 95% CI half-width |
|----------------------------------------------------|--------------|-------------------|
| Selective avg (6 NK+LK rerank + 3 PK baseline)     | **0.6215**   | ±0.0014           |
| 6-cell NK+LK only avg (rerank applied)             | **0.6845**   | (see below)       |
| 9-cell all-baseline avg                            | 0.5818       | n/a               |
| Selective lift vs baseline                         | +0.0397      |                   |

Per-cell mean over the three seeds:

| Cell    | Mean Fmax | CI half | Baseline Fmax | Lift     |
|---------|-----------|---------|---------------|----------|
| nk-mfo  | 0.7065    | 0.0036  | 0.6447        | +0.0618  |
| nk-bpo  | 0.5596    | 0.0024  | 0.5333        | +0.0262  |
| nk-cco  | 0.7774    | 0.0048  | 0.7000        | +0.0774  |
| lk-mfo  | 0.6806    | 0.0060  | 0.5816        | +0.0991  |
| lk-bpo  | 0.6460    | 0.0032  | 0.5844        | +0.0615  |
| lk-cco  | 0.7367    | 0.0091  | 0.7053        | +0.0315  |

All six NK and LK cells show strictly positive lift across all three
seeds. The maximum CI half-width is 0.0091 (lk-cco), so the reranker
effect is robust to seed variation.

## Reproducibility

* **Source-tree pin.** The image is built from `develop` at commit
  `6fdc2e0` (`feat(LAFA-EMB.1): in-container PLM embedder for the LAFA
  submission image (#21)`), which is also the merge commit that
  surfaces PRs #19 (LAFA-CONTAINER.1) and #20 (F2C.6). Replace this
  hash with the actual build commit at release time.
* **Image digest.** Pin the `sha256:...` digest emitted by
  `docker push` and paste it into the FunctionBench form. Subsequent
  pulls of `:v0.1.0` resolve to the same digest.
* **Determinism.** With identical inputs and a fixed `--backend numpy`
  setting, the predictions TSV is bit-for-bit reproducible between
  runs. The FAISS backend exposes the IVFFlat training, which is
  data-order-dependent; prefer numpy for the reproducibility
  submission.
* **Bind-mount embeddings.** Supplying pre-computed parquet
  embeddings via `--query_embeds` / `--reference_embeds` short-circuits
  the PLM step entirely; the remaining pipeline is deterministic.

## Hardware

* CPU works end-to-end. The validation numbers above were produced on
  CPU using `--backend numpy`.
* For production-scale FunctionBench submissions, GPU (CUDA) inference
  is recommended via `--embed_device cuda`. The default image bundles
  a CPU torch wheel; GPU users rebuild from `v0.1.0-slim` adding the
  CUDA wheel of choice.
* Cold-start cost: ESM-2 3B downloads about 12 GB of HuggingFace
  weights on the first run. Mount a persistent HF cache to avoid
  paying it again.
* Image size: about 6 GB. Disk footprint after weights download: about
  18 GB.

## Limitations

* The aspect-separated KNN path landed with contract tests (PR #20),
  but the end-to-end regression against pinned v226 fixtures has not
  shipped yet; that lands with the FARM-EXP.5 fixture-freeze slice.
  Until then the numbers above stand as the validation reference, and
  any regression must be re-evaluated on the lab side rather than
  inside the container.
* The reranker is leakage-fixed (no anc2vec, no PCA features). Earlier
  sessions reported a 0.4562 average cafaeval number; that figure is
  pre-fix and is not comparable to the numbers in this card.
* The PK eval band is not currently re-ranked (selective fallback to
  the KNN baseline), since the reranker did not show a consistent lift
  on PK in the LB.2 sweep.
* The default `--backend numpy` is the safe choice up to a few hundred
  thousand reference proteins. Beyond that, switch to `--backend
  faiss` and accept the (small) IVFFlat training non-determinism.

## LAFA framing

PROTEA implements the LAFA standard container interface documented at
github.com/anphan0828/LAFA_container_guide. The image's entrypoint is
`python3 method_main.py` and consumes the five required LAFA flags
(`--query_file`, `--train_sequences`, `--annot_file`, `--graph`,
`--output_file`). Mount points follow the LAFA convention
(`/app/data:ro`, `/app/output:rw`). Output is a three-column TSV with
no header, all three aspects interleaved, optionally gzipped.
