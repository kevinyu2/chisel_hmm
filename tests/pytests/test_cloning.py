"""Tests for chisel.CellClusterer (divisive clustering of cells) and Cloner --clustering divisive.
Runs under Python 2.7 (Cloner end to end) and Python 3 (the clustering module)."""
from __future__ import division
import os
import sys
import tempfile
from collections import Counter

import numpy as np
import pytest

from chisel.CellClusterer import bimodal, mode_split, signal_pcs, divisive, kmeans, read_features, feature_matrix, noisy_cells

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
    # the synthetic calls have no bin clusters (CLUSTER 0 everywhere): test the majority vote
    cloner_main(args=[path, '--clustering', 'divisive', '--perm', '10', '--seed', '1', '--nopseudobulk'], stdout_file=out)
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
def test_cloner_defaults_divisive_with_pseudobulk_and_legacy_switches():
    from chisel.Cloner import parse_args
    a = parse_args([__file__])
    assert a['clustering'] == 'divisive' and a['pseudobulk']
    assert not parse_args([__file__, '--nopseudobulk'])['pseudobulk']
    h = parse_args([__file__, '--clustering', 'hamming'])
    assert h['clustering'] == 'hamming' and not h['pseudobulk']          # hamming implies no pseudobulk


def test_cell_groups_command_and_read_groups_skip_noisy_cells():
    from chisel.CellClusterer import cell_groups
    from chisel.HMMClusterizer import read_groups
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'calls.tsv')
    groups, _ = write_calls(path, np.random.RandomState(13))
    g, log = cell_groups(path, n_perm=10)
    names = [g['cell{:03d}'.format(e)] for e in range(len(groups))]
    assert ari(groups, names) > 0.98 and set(names) == {'Group1', 'Group2', 'Group3'}
    gf = os.path.join(tmp, 'groups.tsv')
    with open(gf, 'w') as f:
        f.write('#CELL\tGROUP\n')
        for i, (e, k) in enumerate(g.items()):
            f.write('{}\t{}\n'.format(e, 'None' if i == 0 else k))     # one cell set apart
    rg = read_groups(gf, list(g.keys()))
    assert sum(len(ix) for _, ix in rg) == len(g) - 1 and 0 not in np.concatenate([ix for _, ix in rg])


def write_clone_calls(path, rng, weak=0.7):
    """calls.tsv with the true bin clusters; clone B has a 1-bin deletion that only (1 - weak) of its cells call."""
    groups = ['A'] * 60 + ['B'] * 50 + ['C'] * 50
    def cn(g, chrom, i):
        if g == 'A' and chrom == 'chr1' and 20 <= i < 35: return (1, 2)
        if g == 'B' and chrom == 'chr1' and 5 <= i < 8: return (1, 2)
        if g == 'B' and chrom == 'chr2' and i == 10: return (0, 1)
        if g == 'C' and chrom == 'chr2' and 20 <= i < 23: return (1, 2)
        return (1, 1)
    def bincl(chrom, i):
        for k, (c, lo, hi) in enumerate((('chr1', 20, 35), ('chr1', 5, 8), ('chr2', 10, 11), ('chr2', 20, 23))):
            if chrom == c and lo <= i < hi:
                return k + 1
        return 0
    depth = rng.lognormal(0, 0.15, len(groups))
    rows = []
    for chrom in ('chr1', 'chr2'):
        for i in range(40):
            for e, g in enumerate(groups):
                a, b = cn(g, chrom, i)
                cnt = rng.poisson(60 * depth[e] * (a + b) / 2.0)
                snp = rng.poisson(10 * (a + b) / 2.0)
                bb = rng.binomial(snp, b / float(a + b))
                call = (a, b)
                if g == 'B' and chrom == 'chr2' and i == 10 and rng.rand() < weak:
                    call = (1, 1)
                elif rng.rand() < 0.05:
                    call = tuple(rng.randint(0, 3, 2))
                rows.append([chrom, i * 1000000, (i + 1) * 1000000, 'cell{:03d}'.format(e), 4000, cnt, 0.0, snp - bb, bb,
                             bb / float(snp) if snp else 0.5, bincl(chrom, i), '{}|{}'.format(*call)])
    tot = {}
    for r in rows:
        tot[r[3]] = tot.get(r[3], 0) + r[5]
    with open(path, 'w') as f:
        f.write('#CHR\tSTART\tEND\tCELL\tNORM_COUNT\tCOUNT\tRDR\tA_COUNT\tB_COUNT\tBAF\tCLUSTER\tHAP_CN\n')
        for r in rows:
            r[6] = (r[5] / float(tot[r[3]])) / (4000.0 / (80 * 4000))
            f.write('\t'.join(map(str, r)) + '\n')
    return groups


@pytest.mark.skipif(not PY2, reason='Cloner.py is Python 2.7 code')
def test_pseudobulk_calls_keep_an_event_most_single_cells_miss():
    from chisel.Cloner import main as cloner_main
    out = {}
    for mode in ('pseudobulk', 'vote'):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'calls.tsv')
        groups = write_clone_calls(path, np.random.RandomState(4))
        args = [path, '--seed', '1', '-P', '3', '--perm', '10'] + (['--nopseudobulk'] if mode == 'vote' else [])
        cloner_main(args=args, stdout_file=os.path.join(tmp, 'mapping.tsv'))
        with open(path) as f:
            f.readline()
            rows = [l.split() for l in f]
        dele = [r[12] for r in rows if r[0] == 'chr2' and r[1] == '10000000' and groups[int(r[3][4:])] == 'B']
        raw = [r[11] for r in rows if r[0] == 'chr2' and r[1] == '10000000' and groups[int(r[3][4:])] == 'B']
        out[mode] = (np.mean([x == '0|1' for x in raw]), np.mean([x == '0|1' for x in dele]))
        if mode == 'pseudobulk':
            assert os.path.isfile(os.path.join(tmp, 'pseudobulk_calls.tsv'))
    assert out['pseudobulk'][0] < 0.5                    # most single cells miss the deletion
    assert out['vote'][1] == 0.0                         # so the majority vote erases it
    assert out['pseudobulk'][1] == 1.0                   # the clone's pooled reads keep it


@pytest.mark.skipif(not PY2, reason='Cloner.py is Python 2.7 code')
def test_cloner_takes_given_groups():
    from chisel.Cloner import main as cloner_main
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'calls.tsv')
    groups = write_clone_calls(path, np.random.RandomState(5))
    gf = os.path.join(tmp, 'groups.tsv')
    with open(gf, 'w') as f:
        f.write('#CELL\tGROUP\n')
        for e, g in enumerate(groups):
            f.write('cell{:03d}\t{}\n'.format(e, 'None' if e == 0 else 'G' + g))
    out = os.path.join(tmp, 'mapping.tsv')
    cloner_main(args=[path, '--groups', gf, '--nopseudobulk'], stdout_file=out)
    with open(out) as f:
        f.readline()
        clone = {l.split()[0]: l.split()[2] for l in f}
    assert clone['cell000'] == 'None' and len(set(v for v in clone.values() if v != 'None')) == 3
