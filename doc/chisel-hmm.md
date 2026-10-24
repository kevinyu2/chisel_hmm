# HMM bin clustering (HATCHet2-style) in CHISEL

CHISEL's calling step (`Caller.py`) first groups bins into clusters, then calls a copy-number state per
cluster in every cell, then phases the states along the genome. This fork can replace the first step,
k-medians on (RDR, mirrored BAF) of every cell, by the locality-aware Gaussian HMM of HATCHet2's
`cluster_bins` (Tanner et al., Genome Biology 2023). Everything after the clustering is unchanged.

## Using it

```
chisel ... --hmm [--hmmselect bic|silhouette] [--hmmmink 5] [--hmmmaxk 30] [--hmmrestarts 10]
              [--hmmtau 1e-6] [--hmmdecode map|viterbi] [--hmmtransmat diag|fixed|free]
              [--hmmbaf oriented|mirrored] [--hmmgroups CELL_GROUPS.tsv]
```

`--hmmk K` uses exactly K clusters. To redo only the calling on an existing run (from its
`combo/combo.tsv`), `chisel_calling` takes the same options, plus `--weighted` for the
length-weighted cloning that `chisel` uses with variable-length bins:
`chisel_calling RUN/combo/combo.tsv -x NEWRUN -P 3 -K 100 --weighted --hmm --hmmk 7`.

`calls/hmm_selection.tsv` gets the per-K table (criterion, log-likelihood, decoding score, states used,
stay probability, number of parameters, restart kept; `*` = chosen). Caller alone:
`python2.7 Caller.py combo.tsv --clustering hmm --hmm... --hmmreport FILE`.

To look at the states only, on any small table of RDR and BAF:

```
chisel_hmmstates TABLE [--mink 2 --maxk 6 | -K 3] [--select bic|silhouette] [...same options...]
chisel_hmmstates --demo [--demo-out toy.tsv]
```

TABLE has one line per bin and cell: a header with `CHR START END CELL RDR BAF` (optional
`A_COUNT B_COUNT COUNT`), or 6 columns without header, or CHISEL's `combo/combo.tsv`. It prints
`CHR START END STATE MAXPOST` per bin, and to stderr the model-selection table and the fitted mean BAF
and RDR of every state (per sample when there are at most 8). `--demo` runs a toy (2 chromosomes x 30
bins, 12 cells, a one-copy gain in half of the cells and an LOH deletion) and compares the states with
the truth.

## The model

* A bin is a vector of BAF and RDR for every sample (2 x samples values). A sample is a cell, or with
  `--hmmgroups` a pseudobulk group of cells (count-weighted RDR, pooled-allele BAF): the place to try
  fewer, less noisy samples.
* Bins are split into independent sequences at chromosome changes and at gaps between consecutive
  bins (arms with variable-length bins), `--hmmsplit chromosomes` for chromosomes only.
* K states, each a Gaussian with diagonal covariance (a mean and a variance per feature), shared by all
  sequences. Transitions: `diag` = stay with a learned probability, move to any other state with equal
  probability (HATCHet2's default), `fixed` = stay 1 - tau kept, `free` = a full learned matrix.
* For K = mink..maxk: `restarts` fits by Baum-Welch from k-means initial means (restart r uses seed
  SEED + r), the restart with the best decoding score kept; K chosen by BIC (lowest) or the silhouette
  of the decoded clusters on standardized features (highest).

The code follows hmmlearn's GaussianHMM as HATCHet2 calls it (initial variances = feature variances +
1e-3, variance prior 0.01, at most 10 EM iterations, tolerance 0.01; MAP restarts scored by the sum of
the largest posteriors) and is tested against hmmlearn and HATCHet2's `DiagGHMM` to 1e-8
(`tests/pytests/test_hmm.py`, Python 3 with hmmlearn). It is plain numpy so that it runs in CHISEL's
Python 2.7. Deliberate differences, each with an option to undo it:

* BIC uses the restart that was kept and the sequence lengths (HATCHet2: `model.bic(X)` on the last
  restart, without lengths). `--hmmbicparams tied` counts the tied transitions as 1 parameter instead of
  hmmlearn's K (K - 1).
* The tied stay probability is expected stays / expected transitions over all sequences; HATCHet2's
  `DiagGHMM` weights the states by the last sequence only (`--hmmtiedupdate hatchet`).
* k-means is a reimplementation of sklearn's (greedy k-means++, 10 runs): same algorithm, not the same
  random numbers.

## Things to know

* With cells as samples (1000 cells = 2000 features), BIC's penalty grows by about 2 x features x ln(bins)
  per cluster: small subclonal events are unlikely to earn a cluster (see the discussion of pseudobulk
  groups).
* On small inputs the learned stay probability can be far from sticky (0.90 on the 60-bin toy), so a
  single deviating bin can switch state; `--hmmtransmat fixed` keeps tau.
* The silhouette prefers few well-separated clusters: on the toy it merges the half-of-cells gain into
  the normal state.
* Bins where a cell has no SNP reads have BAF 0.5 in combo.tsv and are used as such.
