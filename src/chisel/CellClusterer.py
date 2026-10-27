#!/usr/bin/env python2.7
"""Divisive (top-down) clustering of cells on their RDR and BAF, for CHISEL's Cloner
(Cloner.py --clustering divisive).

Features: every cell's RDR and BAF (B / (A + B)) in every bin, from the calls file, each bin centred
on its mean over cells (cells without SNP reads in a bin get the bin's mean BAF); optionally in
units of expected counting noise (Poisson on the cell's and the normal's reads for RDR, binomial on
the cell's SNP reads for BAF). The RDR block and the BAF block are scaled to total variance 1 and
weighted rdr_weight : 1 - rdr_weight, so within a block bins keep their own variance.

Per-group feature selection (n_features > 0, needs the noise matrix Z: every value in units of its
expected counting noise, see noise_matrix): before every split, only the n_features features whose
variance in Z within the group is largest are used. A feature whose cells all agree has variance
about 1 in Z; the bins of an event that some of the group's cells carry vary more. With thousands of
features and few cells, the noise floor of the top PC grows with the number of features, so a
difference over a few bins (e.g. 1 Mb in 100-kb bins) is invisible to PCA on all features but clear
on the selected ones. The features are chosen again in every group, so nested differences each get
their turn. The permutation test stays valid: shuffling keeps every selected feature's variance.

Window features (scales): instead of single bins, the features are sums of 2, 3, ... consecutive
bins of a chromosome (sliding windows, every scale in `scales`; 1 = the bins themselves), divided by
sqrt(window length) so that pure noise keeps variance 1 in noise units. An event over a few bins,
whose single bins are each too weak to stand out among thousands of noise features, adds up in the
window that covers it. Overlapping windows share bins and therefore noise, so the permutation test
shuffles the bins (each bin's values across the group's cells) and rebuilds the selected windows from
them: the null keeps exactly the overlap structure of the real features.

Starting from all cells, a group is split in two when all of these hold:
  * signal: PCA of the group's own cells has at least one PC whose variance beats a noise floor by
    floor_margin, the floor being the perm_q percentile of the top PC variance over perm matrices in
    which every feature's values are shuffled across the group's cells (Horn's parallel analysis:
    each feature keeps its own noise, shared structure such as a clone is destroyed);
  * modes: the cells are projected on one line (the signal PC, or with several the line between
    the two centres of 2-means on the signal PCs) and 1 to 4 Gaussians are fitted to it, the number
    chosen by BIC; with more than one, the cut goes between the two neighbouring Gaussians that are
    furthest apart, which must be apart (Ashman's D = |mu1 - mu2| sqrt(2 / (s1^2 + s2^2)) >= ashman).
    Cutting between modes rather than by 2-means matters when clones lie on one line (the same
    segment at 1|1, 2|1, 3|1): 2-means then prefers to cut through the middle clone;
  * size: both halves have >= min_size cells. When 2-means cuts off fewer cells (outliers), those
    are set aside and the split retried (up to peel times); set-aside cells join the nearer half.
Very noisy cells are set apart first (outlier_z): a cell's noise level is the median absolute
deviation of its values over all features, which a few event bins barely change; cells more than
outlier_z robust SDs above the typical level are left out of the splits, the noise floors and the
reassignment (label -1). Noisy cells inside a group would otherwise raise its noise floor and hide a
clone, and drag cluster centres.
Groups that do not split are the clusters; finally every cell is reassigned to the nearest cluster
centre in the top PCs of all cells (with feature selection: of the features used by the accepted
splits), repeated until stable (corrects cells sent the wrong way by an
early split). Re-running PCA inside every group lets small differences become the main signal once
large ones are separated, which suits nested (tree-like) clones and clones that differ by the same
segment at different copy numbers.
"""
from __future__ import division, print_function

import numpy as np


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


def feature_matrix(arr, rdr_weight=0.5, noise_scale=False):
    """cells x (RDR bins + BAF bins), centred per bin, blocks scaled to total variance 1 and weighted."""
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
    R = np.nan_to_num(np.where(np.isfinite(R), R, 0.0))
    B = np.nan_to_num(np.where(np.isfinite(B), B, 0.0))
    tr = max((R ** 2).sum() / R.shape[0], 1e-12)
    tb = max((B ** 2).sum() / B.shape[0], 1e-12)
    return np.hstack([R * np.sqrt(rdr_weight / tr), B * np.sqrt((1.0 - rdr_weight) / tb)])


def noise_matrix(arr):
    """cells x (RDR bins + BAF bins), centred per bin, in units of expected counting noise (Poisson on the
    cell's and the normal's reads for RDR, binomial on the cell's SNP reads for BAF); missing -> 0."""
    rdr, n = arr['rdr'], arr['a'] + arr['b']
    with np.errstate(invalid='ignore', divide='ignore'):
        baf = np.where(n > 0, arr['b'] / np.where(n > 0, n, 1.0), np.nan)
        cnt, nrm = arr['count'], np.maximum(arr['norm'], 1.0)
        rsd = np.sqrt(rdr ** 2 * (1.0 / np.where(cnt > 0, cnt, np.nan) + 1.0 / nrm))
        p = np.clip(np.nansum(arr['b'], axis=0) / np.maximum(np.nansum(n, axis=0), 1.0), 0.01, 0.99)
        bsd = np.sqrt(p * (1 - p) / np.where(n > 0, n, np.nan))
        R = (rdr - np.nanmean(rdr, axis=0)) / rsd
        B = (baf - np.nanmean(baf, axis=0)) / bsd
    Z = np.hstack([R, B])
    return np.where(np.isfinite(Z), Z, 0.0)


def window_matrix(chroms, scales=(2, 3)):
    """Sliding windows of consecutive bins of the same chromosome, for every scale, as a sparse matrix
    (2 * bins) x features mapping [RDR bins | BAF bins] to [RDR windows | BAF windows]; entries
    1 / sqrt(window length). Returns (matrix, description of every feature as (block, scale, first bin))."""
    from scipy import sparse
    nb = len(chroms)
    rows, cols, vals, desc = [], [], [], []
    for block, off in (('RDR', 0), ('BAF', nb)):
        for k in sorted(set(int(x) for x in scales)):
            if k < 1:
                raise ValueError('window scales must be positive')
            for i in range(nb - k + 1):
                if chroms[i] != chroms[i + k - 1]:
                    continue                                  # a window never crosses chromosomes
                j = len(desc)
                for b in range(i, i + k):
                    rows.append(off + b)
                    cols.append(j)
                    vals.append(1.0 / np.sqrt(k))
                desc.append((block, k, i))
    W = sparse.csc_matrix((vals, (rows, cols)), shape=(2 * nb, len(desc)))
    return W, desc


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


def signal_pcs(X, rng, n_perm=20, q=95.0, margin=1.3, max_pcs=20, W=None):
    """Horn's parallel analysis -> (number of signal PCs, PC scores, PC variances, noise floor).
    With W (inputs x features, dense), the features are X W, and the permutations shuffle the inputs
    (columns of X) and rebuild the features, which keeps the structure W gives them under the null."""
    from scipy.sparse.linalg import svds
    if W is not None:
        Xin = X - X.mean(axis=0)
        n = Xin.shape[0]
        Xc = Xin.dot(W)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / max(n - 1, 1)
        top = []
        cols = np.arange(Xin.shape[1])
        for _ in range(n_perm):
            idx = rng.rand(*Xin.shape).argsort(axis=0)
            top.append(svds(Xin[idx, cols].dot(W), k=1, return_singular_vectors=False)[0] ** 2 / max(n - 1, 1))
        floor = float(np.percentile(top, q)) * margin
        k = min(max_pcs, len(S))
        return int((var > floor).sum()), U[:, :k] * S[:k], var, floor
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(n - 1, 1)
    top = []
    cols = np.arange(Xc.shape[1])
    for _ in range(n_perm):
        idx = rng.rand(*Xc.shape).argsort(axis=0)        # an independent shuffle of every feature
        top.append(svds(Xc[idx, cols], k=1, return_singular_vectors=False)[0] ** 2 / max(n - 1, 1))
    floor = float(np.percentile(top, q)) * margin
    k = min(max_pcs, len(S))
    return int((var > floor).sum()), U[:, :k] * S[:k], var, floor


def bimodal(y, min_d=2.0):
    """1-D: two Gaussians better than one by BIC, and apart (Ashman's D >= min_d)? -> (bool, BIC1 - BIC2, D)."""
    y = np.asarray(y, float)
    n = len(y)
    v = max(y.var(), 1e-12)
    ll1 = -0.5 * n * (np.log(2 * np.pi * v) + 1)
    lab = y > np.median(y)
    if lab.all() or not lab.any():
        return False, 0.0, 0.0
    w = np.array([1 - lab.mean(), lab.mean()])
    mu = np.array([y[~lab].mean(), y[lab].mean()])
    va = np.array([max(y[~lab].var(), 1e-6 * v), max(y[lab].var(), 1e-6 * v)])
    for _ in range(200):
        dens = w / np.sqrt(2 * np.pi * va) * np.exp(-0.5 * (y[:, None] - mu) ** 2 / va)
        r = dens / np.maximum(dens.sum(1, keepdims=True), 1e-300)
        nk = np.maximum(r.sum(0), 1e-9)
        w, mu = nk / n, (r * y[:, None]).sum(0) / nk
        va = np.maximum((r * (y[:, None] - mu) ** 2).sum(0) / nk, 1e-6 * v)
    dens = w / np.sqrt(2 * np.pi * va) * np.exp(-0.5 * (y[:, None] - mu) ** 2 / va)
    ll2 = np.log(np.maximum(dens.sum(1), 1e-300)).sum()
    delta = (-2 * ll1 + 2 * np.log(n)) - (-2 * ll2 + 5 * np.log(n))
    d = abs(mu[0] - mu[1]) * np.sqrt(2.0 / (va[0] + va[1]))
    return bool(delta > 0 and d >= min_d), float(delta), float(d)


def gmm1d(y, k, iters=300):
    """EM for a k-component 1-D Gaussian mixture, started at quantiles -> (weights, means, variances, loglik)."""
    n = len(y)
    v = max(y.var(), 1e-12)
    mu = np.percentile(y, np.linspace(100.0 / (2 * k), 100 - 100.0 / (2 * k), k))
    va = np.full(k, v / k ** 2)
    w = np.full(k, 1.0 / k)
    for _ in range(iters):
        logd = np.log(w) - 0.5 * np.log(2 * np.pi * va) - 0.5 * (y[:, None] - mu) ** 2 / va
        mx = logd.max(1, keepdims=True)
        r = np.exp(logd - mx)
        r /= r.sum(1, keepdims=True)
        nk = np.maximum(r.sum(0), 1e-9)
        w, mu = nk / n, (r * y[:, None]).sum(0) / nk
        va = np.maximum((r * (y[:, None] - mu) ** 2).sum(0) / nk, 1e-6 * v)
    logd = np.log(w) - 0.5 * np.log(2 * np.pi * va) - 0.5 * (y[:, None] - mu) ** 2 / va
    mx = logd.max(1, keepdims=True)
    ll = float((mx[:, 0] + np.log(np.exp(logd - mx).sum(1))).sum())
    return w, mu, va, ll, logd


def mode_split(y, ashman=2.0, kmax=4):
    """Split a 1-D sample between modes: Gaussian mixtures with 1..kmax components by BIC; if more than one,
    cut between the neighbouring pair with the largest Ashman's D. Returns (side 0/1 per value or None, info)."""
    y = np.asarray(y, float)
    n = len(y)
    best = None
    for k in range(1, kmax + 1):
        if n < 5 * k:
            break
        w, mu, va, ll, logd = gmm1d(y, k)
        bic = -2 * ll + (3 * k - 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, mu, va, logd)
    _, k, mu, va, logd = best
    if k == 1:
        return None, {'k': 1, 'D': 0.0}
    order = np.argsort(mu)
    mu, va = mu[order], va[order]
    d = [abs(mu[i + 1] - mu[i]) * np.sqrt(2.0 / (va[i] + va[i + 1])) for i in range(k - 1)]
    i = int(np.argmax(d))
    if d[i] < ashman:
        return None, {'k': k, 'D': float(d[i])}
    comp = np.argsort(order)[logd.argmax(1)]               # component of every value, in sorted order
    return (comp > i).astype(int), {'k': k, 'D': float(d[i])}


def noisy_cells(X, z=5.0):
    """Cells whose robust spread (median absolute deviation over features) is more than z robust SDs
    above the median cell's -> boolean mask."""
    spread = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    centre = np.median(spread)
    scale = 1.4826 * np.median(np.abs(spread - centre))
    if scale <= 0:
        return np.zeros(X.shape[0], dtype=bool)
    return spread > centre + z * scale


def divisive(X, rng, min_size=14, n_perm=20, q=95.0, margin=1.3, ashman=2.0, peel=5, max_clusters=30, reassign=True,
             outlier_z=5.0, Z=None, n_features=0, W=None):
    """Top-down clustering of the rows of X -> (labels 0..K-1, -1 for noisy cells set apart; log lines)."""
    log = []
    noisy = noisy_cells(X, outlier_z) if outlier_z > 0 else np.zeros(X.shape[0], dtype=bool)
    if noisy.any():
        log.append('{} very noisy cells set apart (noise level > {:g} robust SDs above the typical cell)'.format(int(noisy.sum()), outlier_z))
    keep = np.flatnonzero(~noisy)
    if n_features and Z is None:
        raise ValueError('feature selection needs the noise matrix Z')
    labels, steps = _divisive(X[keep], rng, min_size, n_perm, q, margin, ashman, peel, max_clusters, reassign,
                              None if Z is None else Z[keep], n_features, W)
    full = np.full(X.shape[0], -1, dtype=int)
    full[keep] = labels
    return full, log + steps


def _divisive(X, rng, min_size, n_perm, q, margin, ashman, peel, max_clusters, reassign, Z=None, n_features=0, W=None):
    leaves, queue, log, used = [], [(np.arange(X.shape[0]), 'root', 0)], [], set()
    nfeat = X.shape[1] if W is None else W.shape[1]
    select = n_features and 0 < n_features < nfeat
    ZW = None
    if select:
        ZW = Z if W is None else np.asarray(W.T.dot(Z.T).T)          # features in noise units, all cells

    def features(cells):
        """The group's n_features most variable features in noise units (all features without selection)."""
        if not select:
            return np.arange(nfeat), ''
        disp = ZW[cells].var(axis=0)
        f = np.sort(np.argsort(-disp)[:n_features])
        return f, ', top {} features (noise-unit variance {:.2g}-{:.2g}, median of all {:.2g})'.format(
            n_features, disp[f].min(), disp[f].max(), np.median(disp))

    def featurize(cells, feats):
        """Values of features feats for cells (X W[:, feats], or X columns without windows)."""
        if W is None:
            return X[np.ix_(cells, feats)]
        return np.asarray(W[:, feats].T.dot(X[cells].T).T)

    def test(cells, feats):
        if W is None:
            return signal_pcs(X[np.ix_(cells, feats)], rng, n_perm, q, margin)
        Wf = W[:, feats]
        support = np.unique(Wf.nonzero()[0])                          # the bins the selected windows use
        return signal_pcs(X[np.ix_(cells, support)], rng, n_perm, q, margin, W=Wf[support].toarray())
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
            feats, fnote = features(core)
            m, scores, var, floor = test(core, feats)
            if m == 0:
                verdict = 'PC1 variance {:.3g} vs noise floor {:.3g}: no signal PC{} -> cluster'.format(var[0], floor, fnote)
                break
            if m == 1:
                y = scores[:, 0]
            else:                                          # the line between the centres of 2-means on the signal PCs
                lab2 = kmeans(scores[:, :m], 2, rng)
                if min(np.bincount(lab2, minlength=2)) == 0:
                    y = scores[:, 0]
                else:
                    c0, c1 = scores[lab2 == 0, :m].mean(0), scores[lab2 == 1, :m].mean(0)
                    y = scores[:, :m].dot((c1 - c0) / max(np.linalg.norm(c1 - c0), 1e-12))
            lab, info = mode_split(y, ashman)
            if lab is None:
                verdict = ("{} signal PC(s) (PC1 {:.3g} vs floor {:.3g}{}) but {} -> cluster").format(
                    m, var[0], floor, fnote, 'one mode (BIC)' if info['k'] == 1 else
                    "{} modes, none apart (largest Ashman's D {:.2f})".format(info['k'], info['D']))
                break
            sizes = np.bincount(lab, minlength=2)
            if sizes.min() < min_size:                  # outliers: set aside, retry on the rest
                small = int(sizes.argmin())
                peeled = np.concatenate([peeled, core[lab == small]])
                core = core[lab != small]
                continue
            split = (lab, m, var, floor, info['k'], info['D'], feats, fnote)
            break
        else:
            verdict = 'still cutting off < {} cells after {} retries -> cluster'.format(min_size, peel)
        note = ' ({} outlier cells set aside)'.format(len(peeled)) if len(peeled) else ''
        if split is None:
            log.append('{}{}: {} cells, {}{}'.format(pad, name, len(idx), verdict, note))
            leaves.append(idx)
            continue
        lab, m, var, floor, nmodes, dd, feats, fnote = split
        used.update(feats.tolist())
        halves = [core[lab == 0], core[lab == 1]]
        if len(peeled):
            cen = [featurize(h, feats).mean(0) for h in halves]
            near = np.argmin([((featurize(peeled, feats) - c) ** 2).sum(1) for c in cen], axis=0)
            halves = [np.concatenate([halves[k], peeled[near == k]]) for k in (0, 1)]
        log.append("{}{}: {} cells, {} signal PC(s) (PC1 {:.3g} vs floor {:.3g}{}), {} modes, cut at the widest gap "
                   "(Ashman's D {:.2f}) -> split {} / {}{}".format(pad, name, len(idx), m, var[0], floor, fnote, nmodes, dd,
                                                                len(halves[0]), len(halves[1]), note))
        queue.append((halves[0], name + '.0', depth + 1))
        queue.append((halves[1], name + '.1', depth + 1))
    labels = np.empty(X.shape[0], dtype=int)
    for i, idx in enumerate(leaves):
        labels[idx] = i
    if reassign and len(leaves) > 1:
        cols = np.array(sorted(used)) if select and used else np.arange(nfeat)
        F = featurize(np.arange(X.shape[0]), cols)
        Xc = F - F.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        k = min(max(20, len(leaves)), len(S))
        P = U[:, :k] * S[:k]
        moves = 0
        for _ in range(100):
            C = np.array([P[labels == j].mean(0) for j in range(labels.max() + 1)])
            new = ((P[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
            moved = int((new != labels).sum())
            if not moved:
                break
            moves += moved
            labels = np.searchsorted(np.unique(new), new)          # drop emptied clusters
        log.append('reassignment to the nearest cluster centre (top {} PCs of {}): {} moves, {} clusters'.format(
            k, 'the {} features used by the splits'.format(len(cols)) if select and used else 'all features',
            moves, labels.max() + 1))
    return labels, log
