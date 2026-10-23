"""Tests for chisel.HMMClusterizer (HATCHet2-style HMM bin clustering) and Caller --clustering hmm.

Runs under Python 2.7 (CHISEL's environment) and Python 3. The comparisons with hmmlearn and
HATCHet2's DiagGHMM run only where hmmlearn is installed (it does not support Python 2.7)."""
from __future__ import division
import itertools
import os
import tempfile

import numpy as np
import pytest

from chisel.HMMClusterizer import (GaussianHMM, BinData, forward_backward, viterbi, tied_transmat, kmeans,
                                   silhouette, standardize, pairwise_distances, select_model, make_samples,
                                   sequence_lengths, read_table, read_groups, demo_data, adjusted_rand,
                                   relabel_by_position, write_table)


def random_model(K, D, rng, transmat='free'):
    m = GaussianHMM(K, transmat=transmat)
    m.means = rng.normal(0, 1, (K, D))
    m.vars = rng.uniform(0.3, 2.0, (K, D))
    m.startprob = rng.dirichlet(np.ones(K))
    m.transmat = rng.dirichlet(np.ones(K), size=K)
    return m


def brute_force(m, X):
    """log P(X), best path and its log probability, and posteriors, by enumerating all paths."""
    T, K = X.shape[0], m.K
    logB = m.log_emissions(X)
    paths, logps = [], []
    for path in itertools.product(range(K), repeat=T):
        lp = np.log(m.startprob[path[0]]) + logB[0, path[0]]
        for t in range(1, T):
            lp += np.log(m.transmat[path[t - 1], path[t]]) + logB[t, path[t]]
        paths.append(path)
        logps.append(lp)
    logps = np.array(logps)
    mx = logps.max()
    total = mx + np.log(np.exp(logps - mx).sum())
    w = np.exp(logps - total)
    post = np.zeros((T, K))
    xi = np.zeros((K, K))
    for path, p in zip(paths, w):
        for t in range(T):
            post[t, path[t]] += p
        for t in range(1, T):
            xi[path[t - 1], path[t]] += p
    best = int(logps.argmax())
    return total, np.array(paths[best]), logps[best], post, xi


# ---------------------------------------------------------------------------------------------
# algorithms against brute force
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_forward_backward_and_viterbi_match_enumeration(seed):
    rng = np.random.RandomState(seed)
    m = random_model(3, 4, rng)
    X = rng.normal(0, 1, (5, 4))
    total, best_path, best_lp, post, xi = brute_force(m, X)
    ll, gamma, xis = forward_backward(m.log_emissions(X), m.startprob, m.transmat)
    assert np.isclose(ll, total)
    assert np.allclose(gamma, post)
    assert np.allclose(xis, xi)
    lp, path = viterbi(m.log_emissions(X), m.startprob, m.transmat)
    assert np.isclose(lp, best_lp) and list(path) == list(best_path)


def test_sequences_are_independent():
    rng = np.random.RandomState(3)
    m = random_model(2, 3, rng)
    X = rng.normal(0, 1, (7, 3))
    whole = m.score(X, [7])
    parts = m.score(X, [3, 4])
    assert np.isclose(parts, brute_force(m, X[:3])[0] + brute_force(m, X[3:])[0])
    assert not np.isclose(whole, parts)
    assert m.posteriors(X, [3, 4]).shape == (7, 2)


def test_scaled_recursion_survives_long_sticky_sequences():
    # 3000 bins, 400 features, tau = 1e-6: products would underflow without scaling
    rng = np.random.RandomState(4)
    X = np.vstack([rng.normal(0, 1, (1500, 400)), rng.normal(3, 1, (1500, 400))])
    m = GaussianHMM(2, tau=1e-6).init(X, rng)
    m.fit(X, [3000])
    ll, labels = m.decode(X, [3000], 'viterbi')
    assert np.isfinite(ll) and np.isfinite(m.score(X, [3000]))
    assert len(set(labels[:1500])) == 1 and len(set(labels[1500:])) == 1 and labels[0] != labels[-1]


def test_tied_transmat_and_its_update():
    A = tied_transmat(0.9, 4)
    assert np.allclose(A.sum(axis=1), 1) and np.allclose(np.diag(A), 0.9) and np.isclose(A[0, 1], 0.1 / 3)
    rng = np.random.RandomState(5)
    X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(5, 1, (30, 2)), rng.normal(0, 1, (40, 2))])
    m = GaussianHMM(2, transmat='diag', tau=0.2).init(X, rng)
    m.fit(X, [100])
    off = m.transmat[~np.eye(2, dtype=bool)]
    assert np.allclose(off, off[0])                                    # still tied
    assert 0.95 < m.stay < 1.0                                         # 2 changes in 99 steps
    f = GaussianHMM(2, transmat='fixed', tau=0.2).init(X, rng)
    f.fit(X, [100])
    assert np.allclose(np.diag(f.transmat), 0.8)                       # fixed: not learned


def test_em_increases_the_likelihood():
    rng = np.random.RandomState(6)
    X = np.vstack([rng.normal(0, 1, (40, 5)), rng.normal(2, 0.5, (40, 5))])
    m = GaussianHMM(2, n_iter=15, tol=-np.inf).init(X, rng)
    m.fit(X, [40, 40])
    h = np.array(m.history)
    assert (np.diff(h) > -1e-6 * np.abs(h[1:])).all()


def test_n_parameters_and_bic():
    m = GaussianHMM(4, transmat='diag')
    assert m.n_parameters(10) == 3 + 12 + 80                           # hmmlearn / HATCHet2 count
    assert m.n_parameters(10, 'tied') == 3 + 1 + 80
    assert GaussianHMM(4, transmat='fixed').n_parameters(10, 'tied') == 3 + 80
    assert GaussianHMM(4, transmat='free').n_parameters(10, 'tied') == 3 + 12 + 80
    rng = np.random.RandomState(7)
    X = rng.normal(0, 1, (20, 10))
    m.init(X, rng)
    assert np.isclose(m.bic(X, [10, 10]), -2 * m.score(X, [10, 10]) + 95 * np.log(20))


# ---------------------------------------------------------------------------------------------
# k-means, silhouette, model selection
# ---------------------------------------------------------------------------------------------
def test_kmeans_finds_separated_groups():
    rng = np.random.RandomState(8)
    X = np.vstack([rng.normal(c, 0.1, (20, 3)) for c in (0, 5, 10)])
    C = kmeans(X, 3, np.random.RandomState(0))
    assert sorted(np.round(C[:, 0]).astype(int).tolist()) == [0, 5, 10]


def test_silhouette_matches_definition():
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [9.0]])
    labels = [0, 0, 0, 1, 1, 2]
    Dm = np.abs(X - X.T)
    s = []
    for i in range(6):
        own = [j for j in range(6) if labels[j] == labels[i] and j != i]
        if not own:
            s.append(0.0)
            continue
        a = np.mean([Dm[i, j] for j in own])
        b = min(np.mean([Dm[i, j] for j in range(6) if labels[j] == c]) for c in set(labels) if c != labels[i])
        s.append((b - a) / max(a, b))
    assert np.isclose(silhouette(Dm, labels), np.mean(s))
    assert silhouette(Dm, [0] * 6) == -1.0


@pytest.mark.parametrize('decode', ['map', 'viterbi'])
def test_demo_states_are_recovered_with_bic(decode):
    data, truth = demo_data(0)
    X, names = make_samples(data)
    lengths = sequence_lengths(data.chrs, data.starts, data.ends)
    res = select_model(X, lengths, mink=2, maxk=5, restarts=3, criterion='bic', decode=decode)
    assert res['K'] == 3 and adjusted_rand(truth, res['labels']) == 1.0
    assert [r['K'] for r in res['table']] == [2, 3, 4, 5]


def test_silhouette_prefers_fewer_clusters_on_the_demo():
    # the gain is in half of the cells only, so it sits close to the normal state: silhouette
    # merges it (K = 2) although the K = 3 fit is exact
    data, truth = demo_data(0)
    X, names = make_samples(data)
    lengths = sequence_lengths(data.chrs, data.starts, data.ends)
    res = select_model(X, lengths, mink=2, maxk=5, restarts=3, criterion='silhouette')
    assert res['K'] == 2
    k3 = [r for r in res['table'] if r['K'] == 3][0]
    assert adjusted_rand(truth, k3['labels']) == 1.0 and k3['criterion'] < res['table'][0]['criterion']


def test_selection_is_reproducible_and_parallel_gives_the_same():
    data, truth = demo_data(1)
    X, _ = make_samples(data)
    lengths = sequence_lengths(data.chrs, data.starts, data.ends)
    a = select_model(X, lengths, mink=2, maxk=4, restarts=2, jobs=1)
    b = select_model(X, lengths, mink=2, maxk=4, restarts=2, jobs=2)
    assert [r['criterion'] for r in a['table']] == [r['criterion'] for r in b['table']]
    assert list(a['labels']) == list(b['labels'])


def test_locality_keeps_an_outlier_bin_in_its_segment():
    # bin 15 is inside the gain but its RDR is back to 1 in every cell: with sticky transitions
    # (tau 1e-6 kept fixed) it stays in the gain; with uniform transitions (no locality, a mixture
    # model) it leaves; the stay probability learned on this small toy (0.90) is not sticky enough
    data, truth = demo_data(2, noise_rdr=0.25, noise_baf=0.12)
    X, _ = make_samples(data)
    X[15, 1::2] = 1.0
    lengths = sequence_lengths(data.chrs, data.starts, data.ends)
    lab = (lambda **kw: select_model(X, lengths, mink=3, maxk=3, restarts=3, **kw)['labels'])
    sticky = lab(transmat='fixed', tau=1e-6)
    assert sticky[15] == sticky[14] == sticky[16]
    mixture = lab(transmat='fixed', tau=2.0 / 3.0)
    assert mixture[15] != mixture[14]
    learned = select_model(X, lengths, mink=3, maxk=3, restarts=3)
    assert learned['labels'][15] != learned['labels'][14] and learned['model'].stay < 0.95


# ---------------------------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------------------------
def test_sequence_lengths():
    chrs = ['chr1'] * 4 + ['chr2'] * 3
    starts = [0, 10, 30, 40, 0, 10, 20]
    ends = [10, 20, 40, 50, 10, 20, 30]
    assert sequence_lengths(chrs, starts, ends, 'gaps') == [2, 2, 3]
    assert sequence_lengths(chrs, starts, ends, 'chromosomes') == [4, 3]


def test_make_samples_cells_and_pseudobulk():
    rdr = np.array([[1.0, 2.0, 0.5], [1.0, 1.0, 1.0]])
    baf = np.array([[0.5, 0.75, 0.2], [0.5, 0.5, 0.5]])
    A = np.array([[5, 1, 8], [5, 5, 0]])
    B = np.array([[5, 3, 2], [5, 5, 0]])
    w = np.array([[100, 300, 100], [1, 1, 1]])
    d = BinData(['chr1', 'chr1'], [0, 10], [10, 20], ['a', 'b', 'c'], rdr, baf, A, B, w)
    X, names = make_samples(d)
    assert names == ['a', 'b', 'c'] and X.shape == (2, 6)
    assert np.allclose(X[0], [0.5, 1.0, 0.75, 2.0, 0.2, 0.5])          # BAF, RDR per cell
    Xm, _ = make_samples(d, baf='mirrored')
    assert np.allclose(Xm[0, 0::2], [0.5, 0.25, 0.2])
    groups = [('g1', np.array([0, 1])), ('g2', np.array([2]))]
    Xg, gnames = make_samples(d, groups=groups)
    assert gnames == ['g1', 'g2']
    assert np.isclose(Xg[0, 1], (1.0 * 100 + 2.0 * 300) / 400)         # count-weighted RDR
    assert np.isclose(Xg[0, 0], (5 + 3) / 14.0)                        # pooled BAF
    assert np.isclose(Xg[1, 2], 0.5)                                   # no SNP reads: 0.5


def test_read_table_formats_and_groups():
    data, truth = demo_data(0, ncells=3)
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'toy.tsv')
    write_table(data, p)
    d = read_table(p)
    assert d.nbins == 60 and d.cells == data.cells and np.allclose(d.rdr, data.rdr, atol=1e-5)
    # headerless 6 columns, cells and bins in any order
    q = os.path.join(tmp, 'plain.tsv')
    with open(p) as f, open(q, 'w') as g:
        lines = [l for l in f if not l.startswith('#')]
        g.writelines(lines[::-1])
    d2 = read_table(q)
    assert np.allclose(d2.rdr, d.rdr) and d2.chrs == d.chrs
    # combo.tsv (10 columns)
    c = os.path.join(tmp, 'combo.tsv')
    with open(c, 'w') as g:
        g.write('chr1\t0\t10\tx\t100\t120\t1.2\t3\t7\t0.7\n')
        g.write('chr1\t10\t20\tx\t100\t80\t0.8\t5\t5\t0.5\n')
    d3 = read_table(c)
    assert np.allclose(d3.rdr[:, 0], [1.2, 0.8]) and np.allclose(d3.B[:, 0], [7, 5]) and np.allclose(d3.w[:, 0], [120, 80])
    gp = os.path.join(tmp, 'groups.tsv')
    with open(gp, 'w') as g:
        g.write('cell00 A\ncell01 A\ncell02 B\n')
    groups = read_groups(gp, d.cells)
    assert [(n, list(i)) for n, i in groups] == [('A', [0, 1]), ('B', [2])]
    with open(gp, 'w') as g:
        g.write('cell00 A\n')
    with pytest.raises(ValueError):
        read_groups(gp, d.cells)


def test_relabel_by_position():
    assert list(relabel_by_position([4, 4, 1, 4, 7, 1])) == [0, 0, 1, 0, 2, 1]


# ---------------------------------------------------------------------------------------------
# equivalence with hmmlearn and HATCHet2 (Python 3 with hmmlearn only)
# ---------------------------------------------------------------------------------------------
def hatchet_diagghmm():
    """HATCHet2's DiagGHMM (utils/cluster_bins.py, master 84ebfba), unchanged."""
    from hmmlearn import hmm
    from scipy.special import logsumexp

    class DiagGHMM(hmm.GaussianHMM):
        def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice):
            super()._accumulate_sufficient_statistics(stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)
            if "t" in self.params:
                bothlattice = fwdlattice + bwdlattice
                loggamma = (bothlattice.T - logsumexp(bothlattice, axis=1)).T
                denoms = np.sum(np.exp(loggamma), axis=0)
                stats["denoms"] = denoms

        def _do_mstep(self, stats):
            super()._do_mstep(stats)
            if "t" in self.params:
                denoms = stats["denoms"]
                x = (self.transmat_.T * denoms).T
                num = np.sum(np.diag(x))
                denom = np.sum(x)
                stats["diag"] = num / denom
                self.transmat_ = self.form_transition_matrix(stats["diag"])

        def form_transition_matrix(self, diag):
            tol = 1e-10
            diag = np.clip(diag, tol, 1 - tol)
            offdiag = (1 - diag) / (self.n_components - 1)
            transmat_ = np.diag([diag - offdiag] * self.n_components)
            transmat_ += offdiag
            return transmat_
    return DiagGHMM


@pytest.mark.parametrize('transmat', ['diag', 'fixed', 'free'])
def test_same_fit_and_decoding_as_hmmlearn_and_hatchet(transmat):
    hmmlearn = pytest.importorskip('hmmlearn.hmm')
    data, truth = demo_data(3, noise_rdr=0.2, noise_baf=0.1)
    X, _ = make_samples(data)
    lengths = [20, 25, 15]                                               # several sequences, unequal
    K = 4
    ours = GaussianHMM(K, transmat=transmat, tau=1e-3, tied_update='hatchet').init(X, np.random.RandomState(0))
    params = {'smc': hmmlearn.GaussianHMM, 'smct': hmmlearn.GaussianHMM}
    if transmat == 'diag':
        theirs = hatchet_diagghmm()(n_components=K, init_params='', params='smct', covariance_type='diag')
    else:
        theirs = hmmlearn.GaussianHMM(n_components=K, init_params='', params='smc' if transmat == 'fixed' else 'smct',
                                      covariance_type='diag')
    theirs.startprob_ = ours.startprob.copy()
    theirs.transmat_ = ours.transmat.copy()
    theirs.means_ = ours.means.copy()
    theirs.covars_ = ours.vars.copy()
    ours.fit(X, lengths)
    theirs.fit(X, lengths)
    assert ours.n_iter_done == theirs.monitor_.iter
    assert np.allclose(ours.startprob, theirs.startprob_, atol=1e-8)
    assert np.allclose(ours.transmat, theirs.transmat_, atol=1e-8)
    assert np.allclose(ours.means, theirs.means_, atol=1e-8)
    assert np.allclose(ours.vars, theirs._covars_, atol=1e-8)
    assert np.isclose(ours.score(X, lengths), theirs.score(X, lengths))
    for alg in ('map', 'viterbi'):
        s1, l1 = ours.decode(X, lengths, alg)
        s2, l2 = theirs.decode(X, lengths, algorithm=alg)
        assert np.isclose(s1, s2) and list(l1) == list(l2)
    assert np.isclose(ours.bic(X, lengths), theirs.bic(X, lengths))   # hmmlearn's count, with lengths


def test_initial_variances_as_hmmlearn():
    hmmlearn = pytest.importorskip('hmmlearn.hmm')
    rng = np.random.RandomState(9)
    X = rng.normal(0, 1, (30, 4))
    theirs = hmmlearn.GaussianHMM(n_components=3, init_params='c', params='')
    theirs.means_ = np.zeros((3, 4))
    theirs.startprob_ = np.ones(3) / 3
    theirs.transmat_ = np.ones((3, 3)) / 3
    theirs._init(X)
    ours = GaussianHMM(3).init(X, rng, means=np.zeros((3, 4)))
    assert np.allclose(ours.vars, theirs._covars_)


def test_exact_and_hatchet_tied_updates_agree_on_one_sequence_up_to_the_last_bin():
    data, _ = demo_data(4)
    X, _ = make_samples(data)
    a = GaussianHMM(3, tau=1e-3, tied_update='exact').init(X, np.random.RandomState(0)).fit(X, [60])
    b = GaussianHMM(3, tau=1e-3, tied_update='hatchet').init(X, np.random.RandomState(0)).fit(X, [60])
    assert abs(a.stay - b.stay) < 0.02 and np.allclose(a.means, b.means, atol=0.05)


# ---------------------------------------------------------------------------------------------
# Caller --clustering hmm, end to end
# ---------------------------------------------------------------------------------------------
def write_combo(path, seed=0, ncells=16):
    """combo.tsv: 2 chromosomes x 24 bins. chr1 bins 8-15: +1 copy (2|1) in cells 0-7; chr2 bins
    6-11: LOH deletion (1|0) in every cell; elsewhere 1|1."""
    rng = np.random.RandomState(seed)
    truth = {}
    with open(path, 'w') as f:
        for c, chrom in enumerate(['chr1', 'chr2']):
            for i in range(24):
                for e in range(ncells):
                    cn = (1, 1)
                    if chrom == 'chr1' and 8 <= i < 16 and e < ncells // 2:
                        cn = (1, 2)
                    if chrom == 'chr2' and 6 <= i < 12:
                        cn = (0, 1)
                    truth[(chrom, i, e)] = cn
                    tot = sum(cn)
                    norm = 2000
                    count = rng.poisson(norm * tot / 2.0)
                    rdr = count / float(norm)
                    snp = 80
                    B = rng.binomial(snp, cn[1] / float(tot))
                    A = snp - B
                    f.write('\t'.join(map(str, [chrom, i * 1000000, (i + 1) * 1000000, 'cell{:02d}'.format(e), norm, count, rdr, A, B, B / float(snp)])) + '\n')
    return truth


@pytest.mark.skipif(__import__('sys').version_info[0] >= 3, reason='Caller.py is Python 2.7 code')
def test_caller_with_hmm_clustering_end_to_end():
    from chisel.Caller import main as caller_main
    tmp = tempfile.mkdtemp()
    combo = os.path.join(tmp, 'combo.tsv')
    truth = write_combo(combo)
    out = os.path.join(tmp, 'calls.tsv')
    report = os.path.join(tmp, 'hmm.tsv')
    caller_main(args=[combo, '-j', '2', '--seed', '1', '--clustering', 'hmm', '--hmmmink', '2', '--hmmmaxk', '5',
                      '--hmmrestarts', '3', '--hmmreport', report], stdout_file=out)
    clusters, totals = {}, {}
    with open(out) as f:
        header = f.readline().strip().split('\t')
        assert header[-2:] == ['CLUSTER', 'CN_STATE']
        for line in f:
            p = line.strip().split('\t')
            i = int(p[1]) // 1000000
            clusters[(p[0], i)] = int(p[10])
            totals[(p[0], i, int(p[3][4:]))] = sum(map(int, p[11].split('|')))
    # three clusters: normal, the gain, the deletion
    gain = set(clusters[('chr1', i)] for i in range(8, 16))
    dele = set(clusters[('chr2', i)] for i in range(6, 12))
    normal = set(clusters[k] for k in clusters if not (k[0] == 'chr1' and 8 <= k[1] < 16) and not (k[0] == 'chr2' and 6 <= k[1] < 12))
    assert len(gain) == len(dele) == len(normal) == 1 and len(gain | dele | normal) == 3
    # copy numbers: total 3 in gain cells over the gain, 1 over the deletion, 2 elsewhere
    agree = np.mean([totals[k] == sum(truth[k]) for k in totals])
    assert agree > 0.95
    with open(report) as f:
        rows = [l.split('\t') for l in f if not l.startswith('K')]
    assert [int(r[0]) for r in rows] == [2, 3, 4, 5] and sum(r[-1].strip() == '*' for r in rows) == 1
