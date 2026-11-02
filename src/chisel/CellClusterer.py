#!/usr/bin/env python2.7
"""Divisive (top-down) clustering of cells on their RDR and BAF, for CHISEL's Cloner
(Cloner.py --clustering divisive).

Features: every cell's RDR and BAF (B / (A + B)) in every bin, from the calls file, each bin centred
on its mean over cells (cells without SNP reads in a bin get the bin's mean BAF); optionally in
units of expected counting noise (Poisson on the cell's and the normal's reads for RDR, binomial on
the cell's SNP reads for BAF). The RDR block and the BAF block are scaled to total variance 1 and
weighted rdr_weight : 1 - rdr_weight, so within a block bins keep their own variance.

Starting from all cells, a group is split in two when both of these hold:
  * signal: PCA of the group's own cells has at least one PC whose variance beats a noise floor by
    floor_margin, the floor being the perm_q percentile of the top PC variance over perm matrices in
    which every feature's values are shuffled across the group's cells (Horn's parallel analysis:
    each feature keeps its own noise, shared structure such as a clone is destroyed);
  Then the cells are cut in two by 2-means on their scores on the signal PCs.
  * size: both halves have >= min_size cells. When 2-means cuts off fewer cells (outliers), those
    are set aside and the split retried (up to peel times); set-aside cells join the nearer half.
2-means may cut through a clone (e.g. the middle one of three clones on a line), which then ends up
in two clusters: oversplitting, which is preferred to two clones sharing a cluster.
Very noisy cells are set apart first (outlier_z): a cell's noise level is the median absolute
deviation of its values over all features, which a few event bins barely change; cells more than
outlier_z robust SDs above the typical level are left out of the splits, the noise floors and the
reassignment (label -1). Noisy cells inside a group would otherwise raise its noise floor and hide a
clone, and drag cluster centres.
Groups that do not split are the clusters. Finally k-means with one centre per cluster, started at
the clusters' means, reassigns the cells in the space of the signal PCs that the splits used (their
directions in feature space, together; corrects cells sent the wrong way by an early split). Those are
the directions in which clusters differ, including small differences that are not among the top PCs
of all cells, and leaving out the rest keeps the distances from being dominated by noise. Re-running PCA inside every group lets small differences become the main signal once
large ones are separated, which suits nested (tree-like) clones and clones that differ by the same
segment at different copy numbers.
"""
from __future__ import division, print_function

import numpy as np

# Defaults of the split test (shared by Cloner.py and the chisel drivers).
PERM_Q = 90.0        # noise floor = this percentile of the permuted top PC variance
FLOOR_MARGIN = 1.2   # a signal PC must beat the noise floor by this factor
MIN_CLONE = 30       # smallest cell group (CHISEL's own clones: 14, kept for --clustering hamming)


def read_features(path):
    """Calls file (CHISEL's calls.tsv) -> (cells, bins, dict of cells x bins arrays rdr, baf, a, b, count, norm)."""
    rows = {}
    cells, bins = set(), set()
    with open(path) as f:
        for line in f:
            if line[0] == '#' or len(line) < 2:
                continue
            p = line.split()
            b = (p[0], int(p[1]), int(p[2]))
            rows[(b, p[3])] = (float(p[4]), float(p[5]), float(p[6]), float(p[7]), float(p[8]), float(p[9]))
            cells.add(p[3])
            bins.add(b)
    digits = (lambda c: int(''.join(x for x in c if x.isdigit()) or 0))
    bins = sorted(bins, key=(lambda b: (digits(b[0]), b[1], b[2])))
    cells = sorted(cells)
    ci = {c: i for i, c in enumerate(cells)}
    bi = {b: j for j, b in enumerate(bins)}
    names = ('norm', 'count', 'rdr', 'a', 'b', 'baf')
    arr = {k: np.full((len(cells), len(bins)), np.nan) for k in names}
    for (b, c), vals in rows.items():
        i, j = ci[c], bi[b]
        for k, v in zip(names, vals):
            arr[k][i, j] = v
    return cells, bins, arr


def bin_noise(arr):
    """Expected counting noise (SD across cells) of every bin's RDR and BAF, for a typical cell -> (sdR, sdB).

    RDR: a cell's RDR in a bin is its read count over its expected count, times the bin's level, so its
    Poisson SD is RDR / sqrt(reads); with the bin's mean RDR r and the cells' mean reads m there,
    sdR = r / sqrt(m). The normal's reads scale every cell's RDR in a bin by the same factor, which the
    centring per bin removes, so they add nothing across cells. BAF: binomial, p (1 - p) / n for a cell
    with n SNP reads (p = the bin's pooled BAF); cells without SNP reads get the bin's mean (no spread),
    so sdB^2 = p (1 - p) * mean over cells of (1 / n, or 0 when n = 0). Overdispersion that is the same
    in every bin only rescales all weights and changes nothing."""
    with np.errstate(invalid='ignore', divide='ignore'):
        rbar = np.abs(np.nanmean(arr['rdr'], axis=0))
        reads = np.nanmean(arr['count'], axis=0)
        sdR = np.where(reads > 0, rbar / np.sqrt(np.where(reads > 0, reads, 1.0)), np.inf)
        n = arr['a'] + arr['b']
        p = np.clip(np.nansum(arr['b'], axis=0) / np.maximum(np.nansum(n, axis=0), 1.0), 0.01, 0.99)
        inv = np.where(n > 0, 1.0 / np.where(n > 0, n, 1.0), 0.0)
        sdB = np.sqrt(p * (1 - p) * np.nanmean(inv, axis=0))
    sdR = np.where(np.isfinite(sdR) & (sdR > 0), sdR, np.inf)
    sdB = np.where(np.isfinite(sdB) & (sdB > 0), sdB, np.inf)
    return sdR, sdB


def feature_matrix(arr, rdr_weight=0.5, noise_scale=False, bin_weight=True):
    """cells x (RDR bins + BAF bins), centred per bin.

    bin_weight (default): every bin divided by its expected counting noise (bin_noise), so every
    feature has noise SD ~1 and a bin counts by its signal-to-noise, (shift / noise)^2; RDR and BAF are
    then in the same units and mixed as sqrt(2 w) : sqrt(2 (1 - w)) (w = rdr_weight, 0.5: as they are).
    Otherwise (the first version): each block scaled to total variance 1 and weighted w : 1 - w, every
    bin keeping its own variance, so the noisiest bins (e.g. a few kb with < 1 read per cell) dominate.
    noise_scale: every value divided by its own cell's noise instead (no bin weighting)."""
    rdr, n = arr['rdr'], arr['a'] + arr['b']
    baf = np.where(n > 0, arr['b'] / np.where(n > 0, n, 1.0), np.nan)
    R = rdr - np.nanmean(rdr, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        B = baf - np.nanmean(baf, axis=0)
        if noise_scale:
            cnt, nrm = arr['count'], np.maximum(arr['norm'], 1.0)
            rsd = np.sqrt(rdr ** 2 * (1.0 / np.where(cnt > 0, cnt, np.nan) + 1.0 / nrm))
            p = np.clip(np.nansum(arr['b'], axis=0) / np.maximum(np.nansum(n, axis=0), 1.0), 0.01, 0.99)
            bsd = np.sqrt(p * (1 - p) / np.where(n > 0, n, np.nan))
            R, B = R / rsd, B / bsd
        elif bin_weight:
            sdR, sdB = bin_noise(arr)
            R, B = R / sdR, B / sdB
    R = np.nan_to_num(np.where(np.isfinite(R), R, 0.0))
    B = np.nan_to_num(np.where(np.isfinite(B), B, 0.0))
    if bin_weight and not noise_scale:
        return np.hstack([R * np.sqrt(2.0 * rdr_weight), B * np.sqrt(2.0 * (1.0 - rdr_weight))])
    tr = max((R ** 2).sum() / R.shape[0], 1e-12)
    tb = max((B ** 2).sum() / B.shape[0], 1e-12)
    return np.hstack([R * np.sqrt(rdr_weight / tr), B * np.sqrt((1.0 - rdr_weight) / tb)])


def kmeans(X, K, rng, n_init=20, iters=300):
    """Lloyd's k-means with k-means++ starts, best of n_init; labels 0..K-1."""
    n = X.shape[0]
    best, best_sse = None, np.inf
    for _ in range(n_init):
        C = [X[rng.randint(n)]]
        d2 = ((X - C[0]) ** 2).sum(1)
        for _ in range(1, K):
            C.append(X[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.randint(n)])
            d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
        C = np.array(C)
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
            lab = D.argmin(1)
            newC = np.array([X[lab == k].mean(0) if (lab == k).any() else X[D.min(1).argmax()] for k in range(K)])
            if np.allclose(newC, C):
                break
            C = newC
        sse = ((X - C[lab]) ** 2).sum()
        if sse < best_sse:
            best, best_sse = lab, sse
    return best


def signal_pcs(X, rng, n_perm=20, q=PERM_Q, margin=FLOOR_MARGIN, max_pcs=20):
    """Horn's parallel analysis -> (number of signal PCs, PC scores, PC variances, noise floor, PC directions
    in feature space: one per row)."""
    from scipy.sparse.linalg import svds
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(n - 1, 1)
    top = []
    cols = np.arange(Xc.shape[1])
    for _ in range(n_perm):
        idx = rng.rand(*Xc.shape).argsort(axis=0)        # an independent shuffle of every feature
        top.append(svds(Xc[idx, cols], k=1, return_singular_vectors=False)[0] ** 2 / max(n - 1, 1))
    floor = float(np.percentile(top, q)) * margin
    k = min(max_pcs, len(S))
    return int((var > floor).sum()), U[:, :k] * S[:k], var, floor, Vt[:k]


def noisy_cells(X, z=5.0):
    """Cells whose robust spread (median absolute deviation over features) is more than z robust SDs
    above the median cell's -> boolean mask."""
    spread = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    centre = np.median(spread)
    scale = 1.4826 * np.median(np.abs(spread - centre))
    if scale <= 0:
        return np.zeros(X.shape[0], dtype=bool)
    return spread > centre + z * scale


def divisive(X, rng, min_size=MIN_CLONE, n_perm=20, q=PERM_Q, margin=FLOOR_MARGIN, peel=5, max_clusters=30, reassign=True,
             outlier_z=5.0):
    """Top-down clustering of the rows of X -> (labels 0..K-1, -1 for noisy cells set apart; log lines)."""
    log = []
    noisy = noisy_cells(X, outlier_z) if outlier_z > 0 else np.zeros(X.shape[0], dtype=bool)
    if noisy.any():
        log.append('{} very noisy cells set apart (noise level > {:g} robust SDs above the typical cell)'.format(int(noisy.sum()), outlier_z))
    keep = np.flatnonzero(~noisy)
    labels, steps = _divisive(X[keep], rng, min_size, n_perm, q, margin, peel, max_clusters, reassign)
    full = np.full(X.shape[0], -1, dtype=int)
    full[keep] = labels
    return full, log + steps


def _divisive(X, rng, min_size, n_perm, q, margin, peel, max_clusters, reassign):
    leaves, queue, log, dirs = [], [(np.arange(X.shape[0]), 'root', 0)], [], []
    while queue:
        idx, name, depth = queue.pop(0)
        pad = '  ' * depth
        if len(idx) < 2 * min_size or len(leaves) + len(queue) + 1 >= max_clusters:
            why = '< {} cells'.format(2 * min_size) if len(idx) < 2 * min_size else 'at {} clusters'.format(max_clusters)
            log.append('{}{}: {} cells -> cluster ({})'.format(pad, name, len(idx), why))
            leaves.append(idx)
            continue
        core, peeled, verdict, split = idx, np.array([], dtype=int), None, None
        for _ in range(peel + 1):
            if len(core) < 2 * min_size:
                verdict = '{} cells left after setting aside {} outliers -> cluster'.format(len(core), len(peeled))
                break
            m, scores, var, floor, V = signal_pcs(X[core], rng, n_perm, q, margin)
            if m == 0:
                verdict = 'PC1 variance {:.3g} vs noise floor {:.3g}: no signal PC -> cluster'.format(var[0], floor)
                break
            lab = kmeans(scores[:, :m], 2, rng)
            sizes = np.bincount(lab, minlength=2)
            if sizes.min() < min_size:                  # outliers: set aside, retry on the rest
                small = int(sizes.argmin())
                peeled = np.concatenate([peeled, core[lab == small]])
                core = core[lab != small]
                continue
            split = (lab, m, var, floor, V[:m])
            break
        else:
            verdict = 'still cutting off < {} cells after {} retries -> cluster'.format(min_size, peel)
        note = ' ({} outlier cells set aside)'.format(len(peeled)) if len(peeled) else ''
        if split is None:
            log.append('{}{}: {} cells, {}{}'.format(pad, name, len(idx), verdict, note))
            leaves.append(idx)
            continue
        lab, m, var, floor, V = split
        dirs.append(V)
        halves = [core[lab == 0], core[lab == 1]]
        if len(peeled):
            cen = [X[h].mean(0) for h in halves]
            near = np.argmin([((X[peeled] - c) ** 2).sum(1) for c in cen], axis=0)
            halves = [np.concatenate([halves[k], peeled[near == k]]) for k in (0, 1)]
        log.append("{}{}: {} cells, {} signal PC(s) (PC1 {:.3g} vs floor {:.3g}) -> 2-means split {} / {}{}".format(
            pad, name, len(idx), m, var[0], floor, len(halves[0]), len(halves[1]), note))
        queue.append((halves[0], name + '.0', depth + 1))
        queue.append((halves[1], name + '.1', depth + 1))
    labels = np.empty(X.shape[0], dtype=int)
    for i, idx in enumerate(leaves):
        labels[idx] = i
    if reassign and len(leaves) > 1:
        labels, moves, rank = reassign_in(X, labels, dirs)
        log.append('reassignment: k-means from the {} cluster means in the {} directions of the signal PCs the splits '
                   'used: {} moves, {} clusters'.format(len(leaves), rank, moves, labels.max() + 1))
    return labels, log


def split_space(dirs, tol=1e-6):
    """Orthonormal basis (rows) of the span of the split directions (each an m x features array)."""
    D = np.vstack(dirs)
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[S > tol * S[0]]


def reassign_in(X, labels, dirs, iters=100):
    """k-means started at the clusters' means, in the span of the split directions (overlapping directions
    of nested splits counted once) -> (labels 0..K'-1, cells moved, dimensions). A cluster that loses all
    its cells is dropped."""
    Q = split_space(dirs)
    P = (X - X.mean(axis=0)).dot(Q.T)
    moves = 0
    for _ in range(iters):
        C = np.array([P[labels == j].mean(0) for j in range(labels.max() + 1)])
        new = ((P[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        moved = int((new != labels).sum())
        if not moved:
            break
        moves += moved
        labels = np.searchsorted(np.unique(new), new)          # drop emptied clusters
    return labels, moves, Q.shape[0]


def cell_groups(path, seed=0, min_size=MIN_CLONE, rdr_weight=0.5, noise_scale=False, bin_weight=True, n_perm=20, q=PERM_Q, margin=FLOOR_MARGIN,
                peel=5, max_clusters=30, reassign=True, outlier_z=5.0):
    """Divisive groups of the cells of a combo.tsv or calls.tsv -> (OrderedDict cell -> group name, log lines).
    Groups are Group1, Group2, ... by decreasing size; very noisy cells set apart get None."""
    from collections import OrderedDict
    cells, bins, arr = read_features(path)
    X = feature_matrix(arr, rdr_weight=rdr_weight, noise_scale=noise_scale, bin_weight=bin_weight)
    labels, log = divisive(X, np.random.RandomState(seed), min_size=min_size, n_perm=n_perm, q=q, margin=margin,
                           peel=peel, max_clusters=max_clusters, reassign=reassign, outlier_z=outlier_z)
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([], dtype=int)
    rank = {k: i + 1 for i, k in enumerate(np.argsort(-sizes, kind='mergesort'))}
    out = OrderedDict((e, 'Group{}'.format(rank[l]) if l >= 0 else 'None') for e, l in zip(cells, labels))
    head = ['{} cells, {} bins: {} groups ({}){}'.format(
        len(cells), len(bins), len(sizes), ', '.join(str(s) for s in sorted(sizes, reverse=True)),
        ', {} noisy cells set apart (None)'.format(int((labels < 0).sum())) if (labels < 0).any() else '')]
    return out, head + log


def add_split_arguments(parser, minclone=True):
    """The split-test options of the chisel drivers (passed on to CellClusterer.py and Cloner.py); minclone: also
    --minclone, the smallest cell group (drivers without their own -s)."""
    dv = parser.add_argument_group('divisive cloning', 'The test deciding whether a group of cells is split in two (by 2-means on its signal PCs; --cloning divisive)')
    dv.add_argument("--permq", type=float, required=False, default=PERM_Q, help="Noise floor = this percentile of the top PC variance of feature-shuffled copies of the group (default: {:g})".format(PERM_Q))
    dv.add_argument("--floormargin", type=float, required=False, default=FLOOR_MARGIN, help="A signal PC must beat the noise floor by this factor; lower finds smaller clones (default: {:g})".format(FLOOR_MARGIN))
    if minclone:
        dv.add_argument("--minclone", type=int, required=False, default=MIN_CLONE, help="Smallest cell group / clone: a group is split only into halves of at least this many cells, and Cloner keeps clones of at least this size (default: {}; CHISEL's original cloning used 14)".format(MIN_CLONE))
    return dv


def split_flags(args):
    """Command-line flags of add_split_arguments' options (args: parsed arguments or dict)."""
    get = (lambda k: args[k]) if isinstance(args, dict) else (lambda k: getattr(args, k))
    return " --permq {} --floormargin {}".format(get('permq'), get('floormargin'))


def main(argv=None):
    """CELL GROUP for every cell of a combo.tsv / calls.tsv (divisive clustering on RDR and BAF), to stdout."""
    import argparse
    import sys
    p = argparse.ArgumentParser(description="Divisive groups of cells from CHISEL's combo.tsv or calls.tsv (CELL GROUP to stdout; very noisy cells: None)")
    p.add_argument('INPUT', help='combo.tsv or calls.tsv')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-s', '--minsize', type=int, default=MIN_CLONE, help='minimum group size (default: {})'.format(MIN_CLONE))
    p.add_argument('--rdrweight', type=float, default=0.5)
    p.add_argument('--noisescale', action='store_true')
    p.add_argument('--rawbins', action='store_true', help='no weighting of bins by their expected noise (the first version)')
    p.add_argument('--perm', type=int, default=20)
    p.add_argument('--permq', type=float, default=PERM_Q)
    p.add_argument('--floormargin', type=float, default=FLOOR_MARGIN)
    p.add_argument('--peel', type=int, default=5)
    p.add_argument('--maxclones', type=int, default=30)
    p.add_argument('--noreassign', action='store_true')
    p.add_argument('--outlierz', type=float, default=5.0)
    a = p.parse_args(argv)
    groups, log = cell_groups(a.INPUT, seed=a.seed, min_size=a.minsize, rdr_weight=a.rdrweight, noise_scale=a.noisescale, bin_weight=not a.rawbins,
                              n_perm=a.perm, q=a.permq, margin=a.floormargin, peel=a.peel,
                              max_clusters=a.maxclones, reassign=not a.noreassign, outlier_z=a.outlierz)
    sys.stderr.write('Divisive cell groups: ' + '\n'.join(log) + '\n')
    sys.stdout.write('#CELL\tGROUP\n')
    for e, g in groups.items():
        sys.stdout.write('{}\t{}\n'.format(e, g))


if __name__ == '__main__':
    main()
