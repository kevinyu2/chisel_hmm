# Divisive cloning (Cloner --clustering divisive)

**Defaults (chisel, chisel_calling, chisel_cloning):** cells are grouped by divisive clustering on their
RDR and BAF *before* calling (`calls/cellgroups.tsv`, log `calls/cellgroups.log`; very noisy cells get
group `None`). With `--hmm`, bins are clustered on the groups' pooled reads (`--hmmgroups`, noisy cells
left out). The clones are those groups. Corrected copy numbers come from calling every clone's pooled
reads (pseudobulk): each clone becomes one pseudo-cell (reads and allele counts summed per bin) and is
called by Caller with the run's own bin clusters (`calls/pseudobulk_combo.tsv`, `_clusters.tsv`,
`_calls.tsv`); every cell gets its clone's call, noisy cells keep their own. A majority vote of noisy
single-cell calls can only be as good as the majority: an event most single cells miss (e.g. a 1-bin
deletion at a few SNP reads) is erased by the vote and kept by the pooled reads.

**Legacy switches:** `--nopseudobulk` keeps the divisive groups but clusters bins on every cell and
corrects by majority vote; `--cloning hamming` is CHISEL's original cloning (and no pseudobulk, which
needs the groups). Cloner.py alone: `--clustering divisive|hamming`, `--groups FILE`, `--nopseudobulk`,
`-P` (ploidy of the pseudobulk calling).


(See also [chisel-cloning.md](chisel-cloning.md) for the `chisel_cloning` command, which re-runs only the cloning.)

CHISEL's Cloner groups cells by the Hamming distance between their called copy-number states over the
whole genome and cuts the tree at `--maxdiff` (7% of the genome). Calls are made per bin cluster and
cell, so a cell that miscalls a large cluster differs from everyone, while clones that differ by a few
small events stay below the cut and merge. The divisive mode groups cells on their measured RDR and BAF
instead, top-down:

```
chisel_cloning calls/calls.tsv -x RUN --cloning divisive [--seed 12]   # re-do only the cloning of an existing run
chisel ... --cloning divisive          chisel_calling ... --cloning divisive
python2.7 Cloner.py calls/calls.tsv --clustering divisive [--seed 12] > clones/mapping.tsv
```
(Cloner rewrites the CORRECTED_HAP_CN column of calls.tsv in place; keep a copy to compare.)

* Features: every cell's RDR and BAF in every bin of `calls.tsv`, centred per bin; the RDR and BAF
  blocks scaled to total variance 1 and mixed `--rdrweight` : 1 - w (default 0.5); `--noisescale`
  expresses values in units of counting noise. All bins are used.
* Very noisy cells (robust noise level more than `--outlierz` 5 robust SDs above the typical cell) are
  set apart: clone None, own calls (`--assignoutliers` gives them the nearest clone).
* A group of cells is split in two when its own PCA has a PC beating a permutation noise floor by
  `--floormargin` 1.2x (every feature shuffled across the group's cells, `--perm` 20 times, floor at
  the `--permq` 90th percentile, about the 3rd largest of the 20: Horn's parallel analysis). The cut is
  2-means on the cells' scores on the signal PCs; both halves must have >= `--minsize` 14 cells (small
  outlier groups are set aside and the split retried, `--peel`). Groups that do not split are the
  clones (at most `--maxclones`).
* 2-means can cut through a clone (the middle one of three clones on a line, e.g. the same segment at
  1|1, 2|1 and 3|1), which then becomes two clusters. That oversplitting is accepted: the other way
  round, two clones in one cluster, is worse. Single-population data stays one cluster, as the
  permutation test finds no signal PC.
* Finally k-means with one centre per cluster, started at the clusters' means, reassigns the cells in
  the space of the signal PCs the splits used (their directions in feature space, overlaps counted
  once) (`--noreassign` to skip). These are the directions in which the clusters differ, also small
  differences that are not among the top PCs of all cells; noise in the other directions is left out.
  A cluster that loses all its cells is dropped.
* `--permq` and `--floormargin` are also options of `chisel`, `chisel_calling` and `chisel_cloning`
  (passed to the cell grouping and to Cloner). Lower values find smaller clones; raise them if a run
  overclones (the first defaults were 95 and 1.5).
* The corrected calls are each clone's majority state per bin, as in the default mode. The split tree
  is in the log.

The number of clones comes out of the splitting. Re-running PCA inside each group lets small
differences become the main signal once large ones are separated (nested, tree-like clones).
Code: `src/chisel/CellClusterer.py`; tests: `tests/pytests/test_cloning.py`.
