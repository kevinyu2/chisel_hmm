"""Tests for chisel.CellClusterer (divisive clustering of cells) and Cloner --clustering divisive.
Runs under Python 2.7 (Cloner end to end) and Python 3 (the clustering module)."""
from __future__ import division
import os
import sys
import tempfile
from collections import Counter

import numpy as np
import pytest

from chisel.CellClusterer import bimodal, mode_split, signal_pcs, divisive, kmeans, read_features, feature_matrix, noisy_cells, window_matrix

PY2 = sys.version_info[0] == 2


def ari(a, b):
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    comb = (lambda x: x * (x - 1) / 2.0)
    sij, sa, sb, n = comb(t).sum(), comb(t.sum(1)).sum(), comb(t.sum(0)).sum(), comb(len(a))
    exp = sa * sb / n
    return (sij - exp) / ((sa + sb) / 2.0 - exp)


def test_bimodal_separates_one_hump_from_two():
    rng = np.random.RandomState(0)
    assert not bimodal(rng.normal(0, 1, 400))[0]
    assert not bimodal(np.r_[rng.normal(0, 1, 300), rng.normal(0, 3, 100)])[0]     # heavy tails, one hump
    assert bimodal(np.r_[rng.normal(0, 1, 200), rng.normal(6, 1, 200)])[0]


def test_mode_split_cuts_between_modes():
    rng = np.random.RandomState(7)
    assert mode_split(rng.normal(0, 1, 300))[0] is None                            # one mode: no split
    y = np.r_[rng.normal(0, 1, 100), rng.normal(5, 1, 100), rng.normal(10, 1, 100)]
    side, info = mode_split(y)                                                       # three modes on a line
    assert info['k'] == 3 and sorted(np.bincount(side)) == [100, 200]                # cut between modes, not through one
    assert (side[:100] == side[0]).all() and (side[100:200] == side[100]).all() and (side[200:] == side[200]).all()


def test_permutation_test_finds_shared_structure_only():
    rng = np.random.RandomState(1)
    noise = rng.normal(0, 1, (150, 300))
    assert signal_pcs(noise, np.random.RandomState(2), n_perm=10)[0] == 0
    X = noise.copy()
    X[:50, :40] += 1.0                                   # a clone: 50 cells shifted in 40 features
    assert signal_pcs(X, np.random.RandomState(2), n_perm=10)[0] >= 1


def profiles(design, rng, n_feat=300, noise=1.0):
    """design: list of (cells, {feature slice: shift}) -> X, true labels."""
    rows, labels = [], []
    for g, (n, shifts) in enumerate(design):
        base = np.zeros(n_feat)
        for sl, v in shifts.items():
            base[slice(*sl)] += v
        rows.append(base + rng.normal(0, noise, (n, n_feat)))
        labels += [g] * n
    return np.vstack(rows), np.array(labels)


@pytest.mark.parametrize('name, design', [
    ('nested', [(150, {(0, 90): 1.2}), (150, {(100, 115): 1.0}), (80, {(150, 165): 1.0})]),      # one big, two small
    ('collinear', [(100, {}), (100, {(50, 70): 1.0}), (100, {(50, 70): 2.0})]),                 # 1|1, 2|1, 3|1-like
])
def test_divisive_recovers_clones(name, design):
    X, truth = profiles(design, np.random.RandomState(3))
    labels, log = divisive(X, np.random.RandomState(4), n_perm=10)
    assert labels.max() + 1 == 3, '\n'.join(log)
    # adjacent collinear clones are ~4.7 noise SDs apart on PC1: a few cells land with a neighbour
    assert ari(truth, labels) > 0.9, '\n'.join(log)


def test_noisy_cells_are_set_apart_and_do_not_hide_a_clone():
    rng = np.random.RandomState(8)
    X, truth = profiles([(120, {(0, 30): 1.2}), (60, {(100, 125): 1.3}), (120, {})], rng)
    X = np.vstack([X, rng.normal(0, 4, (25, X.shape[1]))])             # 25 very noisy cells
    assert noisy_cells(X).sum() == 25 and noisy_cells(X)[-25:].all()
    labels, log = divisive(X, np.random.RandomState(9), n_perm=10)
    assert (labels[-25:] == -1).all() and labels.max() + 1 == 3, '\n'.join(log)
    assert ari(truth, labels[:-25]) > 0.95


@pytest.mark.parametrize('seed', [11, 21])
def test_feature_selection_finds_a_small_difference_all_features_miss(seed):
    # 100 vs 300 cells differing in 16 of 2000 features: with thousands of features the noise floor of the
    # top PC hides it; the group's 200 most variable features (in noise units) show it
    X, truth = profiles([(100, {}), (300, {(0, 16): 1.5})], np.random.RandomState(seed), n_feat=2000)
    Z = X - X.mean(0)                                   # noise SD is 1: already in noise units
    lab_all, _ = divisive(X, np.random.RandomState(12), n_perm=10, margin=1.5, Z=Z, n_features=0)        # calibrated at 1.5
    lab_sel, log = divisive(X, np.random.RandomState(12), n_perm=10, margin=1.5, Z=Z, n_features=200)
    assert lab_all.max() + 1 == 1
    assert lab_sel.max() + 1 == 2 and ari(truth, lab_sel) > 0.95, '\n'.join(log)


def test_window_matrix():
    W, desc = window_matrix(['chr1'] * 4 + ['chr2'] * 3, (2, 3))
    A = W.toarray()
    assert A.shape == (14, 2 * ((3 + 2) + (2 + 1)))          # (RDR + BAF bins) x (2- and 3-windows per chromosome)
    for j, (block, k, i) in enumerate(desc):
        off = 0 if block == 'RDR' else 7
        assert np.flatnonzero(A[:, j]).tolist() == list(range(off + i, off + i + k))
        assert np.allclose(A[A[:, j] > 0, j], 1 / np.sqrt(k))
    assert not any(A[3, j] and A[4, j] for j in range(A.shape[1]))   # no window across chr1 / chr2


def test_windows_keep_pure_noise_unsplit():
    chroms = ['chr1'] * 500 + ['chr2'] * 500
    W = window_matrix(chroms, (2, 3))[0]
    for seed in (1, 2):
        X = np.random.RandomState(seed).normal(0, 1, (400, 2000))
        lab, log = divisive(X, np.random.RandomState(seed), n_perm=10, Z=X - X.mean(0), n_features=200, W=W)
        assert lab.max() + 1 == 1, '\n'.join(log)


@pytest.mark.parametrize('seed', [11, 21])
def test_windows_find_a_few_consecutive_bins_single_bins_miss(seed):
    # 100 vs 300 cells differing by 1.2 noise SDs in 5 consecutive bins (per cell ~2.7 SDs summed: a
    # 1:3 mixture that even an ideal cut splits with ~15% of cells on the wrong side, ARI ~0.6)
    chroms = ['chr1'] * 500 + ['chr2'] * 500
    W = window_matrix(chroms, (2, 3))[0]
    X, truth = profiles([(100, {}), (300, {(100, 105): 1.2})], np.random.RandomState(seed), n_feat=2000)
    Z = X - X.mean(0)
    single, _ = divisive(X, np.random.RandomState(12), n_perm=10, Z=Z, n_features=200)
    win, log = divisive(X, np.random.RandomState(12), n_perm=10, Z=Z, n_features=200, W=W)
    assert single.max() + 1 == 1
    assert win.max() + 1 == 2 and ari(truth, win) > 0.45, '\n'.join(log)


def write_calls(path, rng, noise=0.10):
    """calls.tsv: 2 chromosomes x 40 bins, clones A (60 cells, 15-bin gain), B and C (50 each, small events)."""
    groups = ['A'] * 60 + ['B'] * 50 + ['C'] * 50
    def cn(g, chrom, i):
        if g == 'A' and chrom == 'chr1' and 20 <= i < 35: return (1, 2)
        if g == 'B' and chrom == 'chr1' and 5 <= i < 8: return (1, 2)
        if g == 'B' and chrom == 'chr2' and 10 <= i < 12: return (0, 1)
        if g == 'C' and chrom == 'chr2' and 20 <= i < 23: return (1, 2)
        return (1, 1)
    truth = {}
    with open(path, 'w') as f:
        f.write('#CHR\tSTART\tEND\tCELL\tNORM_COUNT\tCOUNT\tRDR\tA_COUNT\tB_COUNT\tBAF\tCLUSTER\tHAP_CN\n')
        for chrom in ('chr1', 'chr2'):
            for i in range(40):
                for e, g in enumerate(groups):
                    a, b = cn(g, chrom, i)
                    cell = 'cell{:03d}'.format(e)
                    truth[(chrom, i, cell)] = (a, b)
                    norm = 400
                    count = rng.poisson(norm * (a + b) / 2.0)
                    snp = rng.poisson(10 * (a + b) / 2.0)
                    bb = rng.binomial(snp, b / float(a + b))
                    call = (a, b) if rng.rand() > noise else tuple(rng.randint(0, 3, 2))
                    f.write('\t'.join(map(str, [chrom, i * 1000000, (i + 1) * 1000000, cell, norm, count, count / float(norm),
                                                snp - bb, bb, bb / float(snp) if snp else 0.5, 0, '{}|{}'.format(*call)])) + '\n')
    return groups, truth


def test_features_from_calls():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'calls.tsv')
    groups, _ = write_calls(path, np.random.RandomState(5))
    cells, bins, arr = read_features(path)
    assert len(cells) == 160 and len(bins) == 80 and arr['rdr'].shape == (160, 80)
    X = feature_matrix(arr)
    assert X.shape == (160, 160) and np.isfinite(X).all()
    assert np.isclose((X[:, :80] ** 2).sum() / 160, 0.5) and np.isclose((X[:, 80:] ** 2).sum() / 160, 0.5)


@pytest.mark.skipif(not PY2, reason='Cloner.py is Python 2.7 code')
def test_cloner_divisive_end_to_end():
    from chisel.Cloner import main as cloner_main
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'calls.tsv')
    groups, truth = write_calls(path, np.random.RandomState(6))
    out = os.path.join(tmp, 'mapping.tsv')
    cloner_main(args=[path, '--clustering', 'divisive', '--perm', '10', '--seed', '1'], stdout_file=out)
    clone = {}
    with open(out) as f:
        assert f.readline().startswith('#CELL')
        for line in f:
            c, k, cl = line.split()
            clone[c] = cl
    labels = [clone['cell{:03d}'.format(e)] for e in range(len(groups))]
    assert len(set(labels)) == 3 and 'None' not in labels
    assert ari(groups, labels) > 0.98
    # corrected calls: every clone's majority state = its true state (raw calls had 10% noise)
    ok = []
    with open(path) as f:
        head = f.readline().split()
        assert head[-1] == 'CORRECTED_HAP_CN'
        for line in f:
            p = line.split()
            ok.append(tuple(map(int, p[12].split('|'))) == truth[(p[0], int(p[1]) // 1000000, p[3])])
    assert np.mean(ok) > 0.99


@pytest.mark.skipif(not PY2, reason='Cloner.py is Python 2.7 code')
def test_cloner_default_is_still_hamming():
    from chisel.Cloner import parse_args
    assert parse_args([__file__])['clustering'] == 'hamming'
