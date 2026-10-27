# Divisive cloning (Cloner --clustering divisive)

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
* Before every split, only the group's `--features` (default 200) most variable features are used,
  variance measured in units of counting noise (about 1 where the group's cells agree): with
  thousands of bins and few cells the noise floor of PCA grows with the number of features, so a
  difference over a few bins (e.g. two 0.5-Mb events in 100-kb bins) is hidden on all features but
  clear on the selected ones. Features are chosen again in every group (`--features 0`: all).
* Features are sums of `--scales` consecutive bins (default windows of 2 and 3 bins, divided by the square
  root of their length so pure noise keeps variance 1 in noise units; `--scales 1` for single bins): an
  event over a few bins whose single bins are each too weak adds up in the window covering it. The
  permutation test shuffles bins and rebuilds the windows, so it keeps their overlap structure.
* A group of cells is split in two when its own PCA has a PC beating a permutation noise floor by
  `--floormargin` 1.3x (every feature shuffled across the group's cells, `--perm` 20 times,
  `--permq` 95th percentile: Horn's parallel analysis), the cells projected on that direction have
  several modes (1-4 Gaussians by BIC) with a clear gap (Ashman's D >= `--ashman` 2, the cut placed in
  the widest gap), and both halves have >= `--minsize` 14 cells (small outlier groups are set aside
  and the split retried, `--peel`). Groups that do not split are the clones (at most `--maxclones`).
* Finally cells are reassigned to the nearest clone centre, in the features used by the splits
  (`--noreassign` to skip).
* The corrected calls are each clone's majority state per bin, as in the default mode. The split tree
  is in the log.

The number of clones comes out of the splitting. Re-running PCA inside each group lets small
differences become the main signal once large ones are separated (nested, tree-like clones), and
cutting at gaps between modes handles clones that differ by the same segment at several copy numbers.
Code: `src/chisel/CellClusterer.py`; tests: `tests/pytests/test_cloning.py`.
