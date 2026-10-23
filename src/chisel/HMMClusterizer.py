#!/usr/bin/env python2.7
"""Locality-aware clustering of bins with a Gaussian hidden Markov model, as in HATCHet2's
cluster_bins (Tanner et al., Genome Biology 2023; raphael-group/hatchet, utils/cluster_bins.py).

Every bin is one observation: for every sample (a cell, or a pseudobulk group of cells, see
make_samples) its BAF and RDR, so a bin is a vector of 2 x (number of samples) values. Bins are
ordered along the genome and split into sequences (chromosomes, and arms or other gaps between
bins). The hidden state of a bin is its cluster. Every state k emits a Gaussian with its own mean and
variance for every feature (diagonal covariance), shared by all sequences, so distant bins with the
same profile share a cluster; the transition matrix is "sticky" (stay 1 - tau, move tau / (K - 1)
to each other state), so neighbouring bins tend to share a cluster.

For every K in [mink, maxk], `restarts` models are fitted by Baum-Welch (EM) from k-means initial
means; the restart with the best decoding score is kept, and K is chosen by BIC (lowest) or by the
silhouette score of the decoded clusters (highest).

The implementation follows hmmlearn's GaussianHMM (which HATCHet2 uses) step by step so that it
runs in CHISEL's Python 2.7 environment without hmmlearn:
  * initial means: k-means (k-means++ seeding, 10 runs), one RandomState per restart index;
    initial variances: the variance of every feature + min_covar (1e-3), the same for all states;
    initial start probabilities uniform, initial transitions sticky with 1 - tau;
  * EM: at most n_iter (10) iterations, stopping when the log-likelihood gains < tol (0.01);
    variances use hmmlearn's inverse-gamma prior (covars_prior = 0.01 added to the numerator);
  * transitions: "diag" = one learned stay probability (HATCHet2's DiagGHMM), "fixed" = tau kept,
    "free" = a full learned matrix;
  * decoding: "map" (per-bin most probable state; hmmlearn scores it by the sum over bins of the
    largest posterior, which is what HATCHet2 uses to pick the restart) or "viterbi" (most probable
    path; scored by its log probability).
Differences from HATCHet2, on purpose (see the options to undo them):
  * BIC is computed on the restart that was kept and with the sequence lengths; HATCHet2 calls
    model.bic(X) on the last restart without lengths;
  * the tied stay probability is (expected stays) / (expected transitions) over all sequences;
    HATCHet2's DiagGHMM weights the states by the posterior mass of the last sequence only
    (tied_update="hatchet" reproduces it);
  * a state that receives no posterior mass keeps its previous mean and variance (hmmlearn would
    divide by zero).
"""
from __future__ import division, print_function

import sys
import os
import argparse
import multiprocessing as mp

import numpy as np


# --------------------------------------------------------------------------------------------
# logging and chromosome order: CHISEL's own when available (Python 2.7 inside the package), else
# a plain fallback (e.g. Python 3, where Utils.py does not import)
# --------------------------------------------------------------------------------------------
def _fallback_log(msg, level='STEP', lock=None):
    sys.stderr.write('[HMM] {}\n'.format(msg))


try:
    from Utils import log, orderchrs
except Exception:
    try:
        from chisel.Utils import log, orderchrs
    except Exception:
        log = _fallback_log

        def orderchrs(c):
            c = str(c)
            c = c[3:] if c.lower().startswith('chr') else c
            return int(c) if c.isdigit() else {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(c.upper(), 100)


TRANSMATS = ('diag', 'fixed', 'free')
DECODINGS = ('map', 'viterbi')
CRITERIA = ('bic', 'silhouette')
BAFS = ('oriented', 'mirrored')
SPLITS = ('gaps', 'chromosomes')


# ============================================================================================
# 1. Data: bins x cells, and the samples the HMM sees
# ============================================================================================
class BinData(object):
    """RDR and BAF of every bin in every cell.

    chrs, starts, ends: one entry per bin, sorted along the genome.
    cells: cell names. rdr, baf: bins x cells. Optional bins x cells arrays: A, B (allele counts,
    BAF = B / (A + B) as in CHISEL's combo.tsv) and w (read counts, the weights CHISEL uses to
    average RDRs).
    """

    def __init__(self, chrs, starts, ends, cells, rdr, baf, A=None, B=None, w=None):
        self.chrs = list(chrs)
        self.starts = np.asarray(starts, dtype=int)
        self.ends = np.asarray(ends, dtype=int)
        self.cells = list(cells)
        self.rdr = np.asarray(rdr, dtype=float)
        self.baf = np.asarray(baf, dtype=float)
        self.A = None if A is None else np.asarray(A, dtype=float)
        self.B = None if B is None else np.asarray(B, dtype=float)
        self.w = None if w is None else np.asarray(w, dtype=float)
        n, m = len(self.chrs), len(self.cells)
        for name in ('rdr', 'baf', 'A', 'B', 'w'):
            x = getattr(self, name)
            if x is not None and x.shape != (n, m):
                raise ValueError('{} has shape {}, expected {} bins x {} cells'.format(name, x.shape, n, m))

    @property
    def nbins(self):
        return len(self.chrs)


def from_caller(bins, cells, pos):
    """BinData from Caller.read_combo's dictionary, bins in the order of `pos`."""
    get = (lambda key: [[bins[b][e][key] for e in cells] for b in pos])
    AB = [[bins[b][e]['AB'] for e in cells] for b in pos]
    return BinData(chrs=[b[0] for b in pos], starts=[b[1] for b in pos], ends=[b[2] for b in pos],
                   cells=cells, rdr=get('RDR'), baf=get('BAF'),
                   A=[[ab[0] for ab in row] for row in AB], B=[[ab[1] for ab in row] for row in AB],
                   w=get('wRDR'))


def read_groups(path, cells):
    """CELL GROUP per line -> list of (group name, cell indices); every cell must be listed."""
    index = {e: i for i, e in enumerate(cells)}
    groups, order = {}, []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0].startswith('#'):
                continue
            if len(p) < 2:
                raise ValueError('{}: every line needs CELL GROUP: {}'.format(path, line.strip()))
            if p[0] not in index:
                continue
            if p[1] not in groups:
                groups[p[1]] = []
                order.append(p[1])
            groups[p[1]].append(index[p[0]])
    missing = len(cells) - sum(len(v) for v in groups.values())
    if missing:
        raise ValueError('{}: {} of {} cells have no group'.format(path, missing, len(cells)))
    return [(g, np.array(sorted(groups[g]))) for g in order]


def make_samples(data, groups=None, baf='oriented'):
    """The samples the HMM clusters on, and the feature matrix.

    groups=None: every cell is a sample. Otherwise a list of (name, cell indices): every group is
    one pseudobulk sample, its RDR the read-count-weighted mean of its cells' RDRs (plain mean
    without counts) and its BAF from the pooled allele counts (mean BAF without counts).
    baf: 'oriented' keeps the BAF as given (CHISEL's combo.tsv: B / (A + B) after Combiner's
    orientation, the analogue of HATCHet2's mhBAF); 'mirrored' uses min(BAF, 1 - BAF), as Caller's
    k-medians clustering does. Returns X (bins x 2 * samples: BAF, RDR of sample 1, BAF, RDR of
    sample 2, ...) and the sample names.
    """
    if baf not in BAFS:
        raise ValueError('baf must be one of {}'.format(BAFS))
    if groups is None:
        names = list(data.cells)
        rdr, bafs = data.rdr, data.baf
    else:
        names = [g for g, _ in groups]
        rdr = np.empty((data.nbins, len(groups)))
        bafs = np.empty((data.nbins, len(groups)))
        for j, (_, idx) in enumerate(groups):
            if data.w is not None:
                w = data.w[:, idx]
                tot = w.sum(axis=1)
                rdr[:, j] = np.where(tot > 0, (data.rdr[:, idx] * w).sum(axis=1) / np.maximum(tot, 1e-300), 0.0)
            else:
                rdr[:, j] = data.rdr[:, idx].mean(axis=1)
            if data.A is not None and data.B is not None:
                a, b = data.A[:, idx].sum(axis=1), data.B[:, idx].sum(axis=1)
                bafs[:, j] = np.where(a + b > 0, b / np.maximum(a + b, 1e-300), 0.5)
            else:
                bafs[:, j] = data.baf[:, idx].mean(axis=1)
    if baf == 'mirrored':
        bafs = np.minimum(bafs, 1.0 - bafs)
    X = np.empty((data.nbins, 2 * len(names)))
    X[:, 0::2] = bafs
    X[:, 1::2] = rdr
    return X, names


def sequence_lengths(chrs, starts, ends, split='gaps'):
    """Lengths of the runs of bins treated as independent sequences (bins sorted along the genome).

    'chromosomes': a new sequence at every chromosome. 'gaps': also wherever a bin does not start
    where the previous one ended (e.g. the centromere between arms when bins are made per arm, as
    variable-length bins are; HATCHet2 splits at the gap between arms the same way).
    """
    if split not in SPLITS:
        raise ValueError('split must be one of {}'.format(SPLITS))
    lengths, run = [], 0
    for i in range(len(chrs)):
        new = i > 0 and (chrs[i] != chrs[i - 1] or (split == 'gaps' and starts[i] != ends[i - 1]))
        if new:
            lengths.append(run)
            run = 0
        run += 1
    if run:
        lengths.append(run)
    return lengths


# ============================================================================================
# 2. The model
# ============================================================================================
def tied_transmat(diag, K):
    """Stay with probability diag, move to every other state with (1 - diag) / (K - 1)."""
    if K == 1:
        return np.ones((1, 1))
    off = (1.0 - diag) / (K - 1)
    return np.full((K, K), off) + np.eye(K) * (diag - off)


def _normalize_rows(a):
    s = a.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return a / s


def _split(n, lengths):
    out, i = [], 0
    for L in lengths:
        out.append((i, i + L))
        i += L
    if i != n:
        raise ValueError('sequence lengths sum to {} but there are {} bins'.format(i, n))
    return out


def forward_backward(logB, startprob, transmat):
    """Scaled forward-backward for one sequence.

    logB: T x K log emission densities. Returns (log-likelihood, posteriors gamma (T x K),
    expected transition counts xi summed over t (K x K)).
    """
    T, K = logB.shape
    m = logB.max(axis=1)
    b = np.exp(logB - m[:, None])
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = startprob * b[0]
    c[0] = max(a.sum(), 1e-300)
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = alpha[t - 1].dot(transmat) * b[t]
        c[t] = max(a.sum(), 1e-300)
        alpha[t] = a / c[t]
    loglik = float(np.log(c).sum() + m.sum())
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = transmat.dot(b[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    if T > 1:
        right = b[1:] * beta[1:] / c[1:, None]
        xi = transmat * alpha[:-1].T.dot(right)
    else:
        xi = np.zeros((K, K))
    return loglik, gamma, xi


def viterbi(logB, startprob, transmat):
    """Most probable state path of one sequence and its log probability."""
    T, K = logB.shape
    with np.errstate(divide='ignore'):
        logA = np.log(transmat)
        delta = np.log(startprob) + logB[0]
    back = np.zeros((T, K), dtype=int)
    cols = np.arange(K)
    for t in range(1, T):
        s = delta[:, None] + logA
        back[t] = s.argmax(axis=0)
        delta = s[back[t], cols] + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return float(delta.max()), path


class GaussianHMM(object):
    """Gaussian HMM with diagonal covariances (hmmlearn GaussianHMM semantics, see module docs)."""

    def __init__(self, K, transmat='diag', tau=1e-6, n_iter=10, tol=1e-2, covars_prior=1e-2,
                 min_covar=1e-3, tied_update='exact'):
        if transmat not in TRANSMATS:
            raise ValueError('transmat must be one of {}'.format(TRANSMATS))
        if tied_update not in ('exact', 'hatchet'):
            raise ValueError("tied_update must be 'exact' or 'hatchet'")
        if not 0.0 < tau < 1.0:
            raise ValueError('tau must be in (0, 1)')
        self.K = int(K)
        self.transmat_type = transmat
        self.tau = tau
        self.n_iter = n_iter
        self.tol = tol
        self.covars_prior = covars_prior
        self.min_covar = min_covar
        self.tied_update = tied_update
        self.startprob = self.transmat = self.means = self.vars = None
        self.n_iter_done = 0
        self.history = []

    # ---- parameters
    def init(self, X, rng, means=None):
        """hmmlearn's initialization with init_params='mc', start and transitions as HATCHet2."""
        K = self.K
        self.means = kmeans(X, K, rng) if means is None else np.array(means, dtype=float)
        var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        self.vars = np.tile(var + self.min_covar, (K, 1))
        self.startprob = np.full(K, 1.0 / K)
        self.transmat = tied_transmat(1.0 - self.tau, K)
        return self

    def get_params(self):
        return {'startprob': self.startprob, 'transmat': self.transmat, 'means': self.means,
                'vars': self.vars, 'K': self.K, 'transmat_type': self.transmat_type, 'tau': self.tau,
                'n_iter': self.n_iter, 'tol': self.tol, 'covars_prior': self.covars_prior,
                'min_covar': self.min_covar, 'tied_update': self.tied_update,
                'n_iter_done': self.n_iter_done}

    @classmethod
    def from_params(cls, p):
        m = cls(p['K'], transmat=p['transmat_type'], tau=p['tau'], n_iter=p['n_iter'], tol=p['tol'],
                covars_prior=p['covars_prior'], min_covar=p['min_covar'], tied_update=p['tied_update'])
        m.startprob, m.transmat, m.means, m.vars = p['startprob'], p['transmat'], p['means'], p['vars']
        m.n_iter_done = p.get('n_iter_done', 0)
        return m

    @property
    def stay(self):
        """Mean stay probability (the tied one for 'diag' and 'fixed')."""
        return float(np.mean(np.diag(self.transmat)))

    # ---- likelihoods
    def log_emissions(self, X):
        """N x K log Gaussian densities (diagonal covariance)."""
        prec = 1.0 / self.vars
        D = X.shape[1]
        const = -0.5 * (D * np.log(2 * np.pi) + np.log(self.vars).sum(axis=1) + (self.means ** 2 * prec).sum(axis=1))
        return const[None, :] - 0.5 * (X ** 2).dot(prec.T) + X.dot((self.means * prec).T)

    def _estep(self, X, lengths):
        logB = self.log_emissions(X)
        K, D = self.K, X.shape[1]
        st = {'start': np.zeros(K), 'trans': np.zeros((K, K)), 'post': np.zeros(K),
              'obs': np.zeros((K, D)), 'obs2': np.zeros((K, D)), 'last_post': np.zeros(K)}
        total = 0.0
        for i, j in _split(X.shape[0], lengths):
            ll, gamma, xi = forward_backward(logB[i:j], self.startprob, self.transmat)
            total += ll
            st['start'] += gamma[0]
            st['trans'] += xi
            st['post'] += gamma.sum(axis=0)
            st['obs'] += gamma.T.dot(X[i:j])
            st['obs2'] += gamma.T.dot(X[i:j] ** 2)
            st['last_post'] = gamma.sum(axis=0)     # HATCHet2's DiagGHMM keeps only the last sequence
        return total, st

    def _mstep(self, st):
        K = self.K
        self.startprob = _normalize_rows(np.maximum(st['start'], 0.0)[None, :])[0]
        if self.transmat_type == 'free':
            self.transmat = _normalize_rows(np.maximum(st['trans'], 0.0))
        elif self.transmat_type == 'diag' and K > 1:
            if self.tied_update == 'exact':
                tot = st['trans'].sum()
                diag = np.trace(st['trans']) / tot if tot > 0 else 1.0 - self.tau
            else:
                x = (_normalize_rows(np.maximum(st['trans'], 0.0)).T * st['last_post']).T
                diag = np.trace(x) / x.sum() if x.sum() > 0 else 1.0 - self.tau
            self.transmat = tied_transmat(float(np.clip(diag, 1e-10, 1.0 - 1e-10)), K)
        post = st['post'][:, None]
        ok = st['post'] > 1e-300
        means = np.where(ok[:, None], st['obs'] / np.maximum(post, 1e-300), self.means)
        c_n = st['obs2'] - 2 * means * st['obs'] + means ** 2 * post
        var = (self.covars_prior + c_n) / np.maximum(post, 1e-5)
        self.means = means
        self.vars = np.where(ok[:, None], var, self.vars)

    def fit(self, X, lengths):
        """Baum-Welch from the current parameters (call init first)."""
        self.history = []
        prev = None
        for it in range(self.n_iter):
            ll, st = self._estep(X, lengths)
            self._mstep(st)
            self.history.append(ll)
            self.n_iter_done = it + 1
            if prev is not None and ll - prev < self.tol:
                break
            prev = ll
        return self

    def score(self, X, lengths):
        """Log-likelihood log P(X | model), summed over sequences (forward algorithm)."""
        logB = self.log_emissions(X)
        return sum(forward_backward(logB[i:j], self.startprob, self.transmat)[0] for i, j in _split(X.shape[0], lengths))

    def posteriors(self, X, lengths):
        logB = self.log_emissions(X)
        return np.vstack([forward_backward(logB[i:j], self.startprob, self.transmat)[1] for i, j in _split(X.shape[0], lengths)])

    def decode(self, X, lengths, algorithm='map'):
        """(score, labels). map: per-bin argmax of the posteriors, score = sum of the maximum
        posteriors (hmmlearn); viterbi: most probable path, score = its log probability."""
        if algorithm == 'map':
            post = self.posteriors(X, lengths)
            return float(post.max(axis=1).sum()), post.argmax(axis=1)
        if algorithm == 'viterbi':
            logB = self.log_emissions(X)
            res = [viterbi(logB[i:j], self.startprob, self.transmat) for i, j in _split(X.shape[0], lengths)]
            return float(sum(r[0] for r in res)), np.concatenate([r[1] for r in res])
        raise ValueError('algorithm must be one of {}'.format(DECODINGS))

    def n_parameters(self, D, count='hmmlearn'):
        """Free parameters for BIC. 'hmmlearn' (HATCHet2): K - 1 start + K (K - 1) transitions +
        K D means + K D variances, whatever the transition type; 'tied': the transitions count as
        1 for 'diag', 0 for 'fixed', K (K - 1) for 'free'."""
        K = self.K
        if count == 'hmmlearn' or self.transmat_type == 'free':
            trans = K * (K - 1)
        else:
            trans = 1 if (self.transmat_type == 'diag' and K > 1) else 0
        return (K - 1) + trans + 2 * K * D

    def bic(self, X, lengths, count='hmmlearn'):
        return -2.0 * self.score(X, lengths) + self.n_parameters(X.shape[1], count) * np.log(X.shape[0])


# ============================================================================================
# 3. k-means (initial means) and silhouette (model selection)
# ============================================================================================
def _sqdist(X, C, xsq=None):
    xsq = (X ** 2).sum(axis=1) if xsq is None else xsq
    return np.maximum(xsq[:, None] - 2 * X.dot(C.T) + (C ** 2).sum(axis=1)[None, :], 0.0)


def kmeans(X, K, rng, n_init=10, max_iter=300, tol=1e-4):
    """Lloyd's k-means with greedy k-means++ seeding, best of n_init runs (as sklearn's KMeans,
    which hmmlearn uses; not bit-identical to it). Returns the K x D centers."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if K >= n:
        C = X[np.arange(K) % n].copy()
        return C
    xsq = (X ** 2).sum(axis=1)
    tol_abs = tol * float(np.mean(X.var(axis=0)))
    best, best_inertia = None, np.inf
    n_local = 2 + int(np.log(K))
    for _ in range(n_init):
        C = np.empty((K, X.shape[1]))
        first = rng.randint(n)
        C[0] = X[first]
        closest = _sqdist(X, C[:1], xsq)[:, 0]
        pot = closest.sum()
        for c in range(1, K):
            if pot <= 0:
                C[c] = X[rng.randint(n)]
                continue
            r = rng.random_sample(n_local) * pot
            cand = np.minimum(np.searchsorted(np.cumsum(closest), r), n - 1)
            dc = _sqdist(X[cand], X, (X[cand] ** 2).sum(axis=1))
            newc = np.minimum(closest[None, :], dc)
            pots = newc.sum(axis=1)
            k = int(pots.argmin())
            C[c], closest, pot = X[cand[k]], newc[k], pots[k]
        for _ in range(max_iter):
            d = _sqdist(X, C, xsq)
            lab = d.argmin(axis=1)
            newC = np.empty_like(C)
            far = np.argsort(-d[np.arange(n), lab])
            fi = 0
            for k in range(K):
                members = lab == k
                if members.any():
                    newC[k] = X[members].mean(axis=0)
                else:                                     # empty cluster: move it to a far point
                    newC[k] = X[far[fi]]
                    fi += 1
            shift = ((newC - C) ** 2).sum()
            C = newC
            if shift <= tol_abs:
                break
        d = _sqdist(X, C, xsq)
        inertia = d.min(axis=1).sum()
        if inertia < best_inertia:
            best, best_inertia = C.copy(), inertia
    return best


def standardize(X):
    """Every column to mean 0, variance 1 (sklearn's StandardScaler; constant columns kept at 0)."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def pairwise_distances(X):
    return np.sqrt(_sqdist(X, X))


def silhouette(Dm, labels):
    """Mean silhouette of the labels for a precomputed distance matrix (sklearn's definition:
    0 for bins alone in their cluster). Returns -1 when there are fewer than 2 or more than
    n - 1 clusters, where the silhouette is undefined (HATCHet2 would stop with an error)."""
    labels = np.asarray(labels)
    uniq, lab = np.unique(labels, return_inverse=True)
    n, k = len(labels), len(uniq)
    if not 2 <= k <= n - 1:
        return -1.0
    onehot = np.zeros((n, k))
    onehot[np.arange(n), lab] = 1.0
    sizes = onehot.sum(axis=0)
    sums = Dm.dot(onehot)                                # distance of every bin to every cluster
    own = sizes[lab]
    a = sums[np.arange(n), lab] / np.maximum(own - 1, 1)
    mean_other = sums / sizes[None, :]
    mean_other[np.arange(n), lab] = np.inf
    b = mean_other.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-300)
    s[own == 1] = 0.0
    return float(s.mean())


# ============================================================================================
# 4. Model selection
# ============================================================================================
_SHARED = {}


def _init_worker(X, lengths, Dm, opts):
    _SHARED['X'], _SHARED['lengths'], _SHARED['D'], _SHARED['opts'] = X, lengths, Dm, opts


def _fit_K(K):
    """All restarts for one K: keep the restart with the best decoding score, then score it."""
    X, lengths, Dm, o = _SHARED['X'], _SHARED['lengths'], _SHARED['D'], _SHARED['opts']
    best = None
    for s in range(o['restarts']):
        rng = np.random.RandomState(o['seed'] + s)          # HATCHet2: random_state = restart index
        model = GaussianHMM(K, transmat=o['transmat'], tau=o['tau'], n_iter=o['n_iter'], tol=o['tol'],
                            tied_update=o['tied_update']).init(X, rng)
        model.fit(X, lengths)
        dscore, labels = model.decode(X, lengths, o['decode'])
        if best is None or dscore > best[0]:                # strictly better, as HATCHet2
            best = (dscore, labels, model, s)
    dscore, labels, model, s = best
    loglik = model.score(X, lengths)
    if o['criterion'] == 'bic':
        crit = -2.0 * loglik + model.n_parameters(X.shape[1], o['bic_params']) * np.log(X.shape[0])
    else:
        crit = silhouette(Dm, labels)
    return {'K': K, 'criterion': float(crit), 'decode_score': dscore, 'loglik': float(loglik),
            'labels': np.asarray(labels), 'params': model.get_params(), 'restart': s,
            'states_used': int(len(np.unique(labels))), 'stay': model.stay,
            'n_params': model.n_parameters(X.shape[1], o['bic_params'])}


def select_model(X, lengths, mink=5, maxk=30, restarts=10, tau=1e-6, transmat='diag', decode='map',
                 criterion='bic', seed=0, jobs=1, n_iter=10, tol=1e-2, bic_params='hmmlearn',
                 tied_update='exact'):
    """Fit K = mink..maxk and choose K. Returns a dict with 'K', 'labels', 'model' (the chosen
    GaussianHMM), 'table' (one row per K, see _fit_K) and 'criterion'."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if criterion not in CRITERIA:
        raise ValueError('criterion must be one of {}'.format(CRITERIA))
    if decode not in DECODINGS:
        raise ValueError('decode must be one of {}'.format(DECODINGS))
    if not np.isfinite(X).all():
        raise ValueError('the features contain NaN or infinite values')
    mink, maxk = max(1, int(mink)), min(int(maxk), n)
    if mink > maxk:
        raise ValueError('mink ({}) is larger than maxk ({}, at most the number of bins {})'.format(mink, maxk, n))
    if criterion == 'silhouette' and maxk < 2:
        raise ValueError('the silhouette needs at least 2 clusters (maxk >= 2)')
    Dm = pairwise_distances(standardize(X)) if criterion == 'silhouette' else None
    opts = {'restarts': restarts, 'tau': tau, 'transmat': transmat, 'decode': decode, 'criterion': criterion,
            'seed': int(seed or 0), 'n_iter': n_iter, 'tol': tol, 'bic_params': bic_params, 'tied_update': tied_update}
    Ks = list(range(mink, maxk + 1))
    if jobs > 1 and len(Ks) > 1:
        pool = mp.Pool(processes=min(jobs, len(Ks)), initializer=_init_worker, initargs=(X, lengths, Dm, opts))
        try:
            rows = pool.map(_fit_K, Ks)
        finally:
            pool.close()
            pool.join()
    else:
        _init_worker(X, lengths, Dm, opts)
        rows = [_fit_K(K) for K in Ks]
    rows.sort(key=lambda r: r['K'])
    best = None
    for r in rows:                                          # first K that is strictly better wins
        better = best is None or (r['criterion'] < best['criterion'] if criterion == 'bic' else r['criterion'] > best['criterion'])
        if better:
            best = r
    return {'K': best['K'], 'labels': best['labels'], 'model': GaussianHMM.from_params(best['params']),
            'table': rows, 'criterion': criterion}


def format_table(result):
    head = ['K', result['criterion'].upper(), 'LOGLIK', 'DECODE_SCORE', 'STATES_USED', 'STAY', 'N_PARAMS', 'RESTART']
    lines = ['\t'.join(head)]
    for r in result['table']:
        mark = ' *' if r['K'] == result['K'] else ''
        lines.append('\t'.join([str(r['K']), '{:.6g}'.format(r['criterion']), '{:.6g}'.format(r['loglik']),
                                '{:.6g}'.format(r['decode_score']), str(r['states_used']), '{:.6g}'.format(r['stay']),
                                str(r['n_params']), str(r['restart'])]) + mark)
    return '\n'.join(lines)


def relabel_by_position(labels):
    """States renumbered 0, 1, 2, ... in order of first appearance along the genome."""
    mapping, out = {}, np.empty(len(labels), dtype=int)
    for i, l in enumerate(labels):
        if l not in mapping:
            mapping[l] = len(mapping)
        out[i] = mapping[l]
    return out


# ============================================================================================
# 5. Entry points: Caller (--clustering hmm) and chisel_hmmstates
# ============================================================================================
def hmm_clustering(bins, cells, pos, par):
    """Caller hook: cluster of every bin (list in the order of pos) with the HMM."""
    data = from_caller(bins, cells, pos)
    groups = read_groups(par['groups'], cells) if par.get('groups') else None
    X, names = make_samples(data, groups=groups, baf=par['baf'])
    lengths = sequence_lengths(data.chrs, data.starts, data.ends, par['split'])
    log('HMM clustering: {} bins in {} sequences, {} samples ({} features), K = {}..{}, {} restarts, {}'.format(
        data.nbins, len(lengths), len(names), X.shape[1], par['mink'], par['maxk'], par['restarts'], par['criterion']), level='INFO')
    res = select_model(X, lengths, mink=par['mink'], maxk=par['maxk'], restarts=par['restarts'], tau=par['tau'],
                       transmat=par['transmat'], decode=par['decode'], criterion=par['criterion'], seed=par['seed'],
                       jobs=par['j'], n_iter=par['n_iter'], bic_params=par['bic_params'], tied_update=par['tied_update'])
    log('HMM model selection (* = chosen):\n' + format_table(res), level='INFO')
    if par.get('report'):
        with open(par['report'], 'w') as f:
            f.write(format_table(res).replace(' *', '\t*') + '\n')
    labels = relabel_by_position(res['labels'])
    log('HMM chose K = {} ({} states used by bins), stay probability {:.6g}'.format(
        res['K'], len(set(labels)), res['model'].stay), level='INFO')
    return [int(l) for l in labels]


def read_table(path):
    """Long table, one line per bin and cell. With a header (CHR START END CELL RDR BAF and,
    optionally, A_COUNT B_COUNT COUNT; any order, case-insensitive), or without one: 6 columns
    CHR START END CELL RDR BAF, or CHISEL's 10-column combo.tsv."""
    rows, header = [], None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if header is None and not rows and (p[0].startswith('#') or p[0].upper() in ('CHR', 'CHROM')):
                header = [x.lstrip('#').upper() for x in p]
                continue
            if p[0].startswith('#'):
                continue
            rows.append(p)
    if not rows:
        raise ValueError('{}: no data'.format(path))
    if header is not None:
        col = {h: i for i, h in enumerate(header)}
        need = ['CHR', 'START', 'END', 'CELL', 'RDR', 'BAF']
        missing = [h for h in need if h not in col]
        if missing:
            raise ValueError('{}: header lacks {}'.format(path, ', '.join(missing)))
        ia, ib, iw = col.get('A_COUNT', col.get('A')), col.get('B_COUNT', col.get('B')), col.get('COUNT')
        get = (lambda p: (p[col['CHR']], int(p[col['START']]), int(p[col['END']]), p[col['CELL']], float(p[col['RDR']]),
                          float(p[col['BAF']]), None if ia is None else float(p[ia]), None if ib is None else float(p[ib]),
                          None if iw is None else float(p[iw])))
    elif len(rows[0]) == 10:
        get = (lambda p: (p[0], int(p[1]), int(p[2]), p[3], float(p[6]), float(p[9]), float(p[7]), float(p[8]), float(p[5])))
    elif len(rows[0]) == 6:
        get = (lambda p: (p[0], int(p[1]), int(p[2]), p[3], float(p[4]), float(p[5]), None, None, None))
    else:
        raise ValueError('{}: without a header, lines need 6 columns (CHR START END CELL RDR BAF) or the 10 of combo.tsv'.format(path))
    recs = [get(p) for p in rows]
    bins = sorted(set(r[:3] for r in recs), key=lambda b: (orderchrs(b[0]), b[1], b[2]))
    cells = sorted(set(r[3] for r in recs))
    bi = {b: i for i, b in enumerate(bins)}
    ci = {e: j for j, e in enumerate(cells)}
    shape = (len(bins), len(cells))
    rdr, baf = np.full(shape, np.nan), np.full(shape, np.nan)
    has_ab, has_w = recs[0][6] is not None and recs[0][7] is not None, recs[0][8] is not None
    A = B = w = None
    if has_ab:
        A, B = np.zeros(shape), np.zeros(shape)
    if has_w:
        w = np.zeros(shape)
    seen = np.zeros(shape, dtype=bool)
    for r in recs:
        i, j = bi[r[:3]], ci[r[3]]
        if seen[i, j]:
            raise ValueError('{}: cell {} appears twice in bin {}:{}-{}'.format(path, r[3], *r[:3]))
        seen[i, j] = True
        rdr[i, j], baf[i, j] = r[4], r[5]
        if has_ab:
            A[i, j], B[i, j] = r[6], r[7]
        if has_w:
            w[i, j] = r[8]
    if not seen.all():
        raise ValueError('{}: {} bin-cell pairs are missing; every cell needs every bin'.format(path, int((~seen).sum())))
    return BinData([b[0] for b in bins], [b[1] for b in bins], [b[2] for b in bins], cells, rdr, baf, A, B, w)


def demo_data(seed=0, ncells=12, noise_rdr=0.08, noise_baf=0.05):
    """Toy data: 2 chromosomes x 30 bins, 12 cells. chr1 bins 10-19: gain of one copy (RDR 1.5,
    BAF 2/3) in cells 0-5; chr2 bins 5-14: LOH deletion in every cell (RDR 0.5, BAF 1, i.e. only the
    B allele left, B / (A + B) = 1); everything else diploid balanced (RDR 1, BAF 0.5).
    Returns (BinData, true state per bin: 0 normal, 1 gain in half of the cells, 2 deletion)."""
    rng = np.random.RandomState(seed)
    chrs = ['chr1'] * 30 + ['chr2'] * 30
    starts = [i * 1000000 for i in range(30)] * 2
    ends = [s + 1000000 for s in starts]
    truth = np.zeros(60, dtype=int)
    truth[10:20] = 1
    truth[35:45] = 2
    rdr = np.ones((60, ncells))
    baf = np.full((60, ncells), 0.5)
    half = np.arange(ncells) < ncells // 2
    rdr[10:20][:, half], baf[10:20][:, half] = 1.5, 2.0 / 3.0
    rdr[35:45], baf[35:45] = 0.5, 1.0
    rdr = rdr + rng.normal(0, noise_rdr, rdr.shape)
    baf = np.clip(baf + rng.normal(0, noise_baf, baf.shape), 0, 1)
    cells = ['cell{:02d}'.format(i) for i in range(ncells)]
    return BinData(chrs, starts, ends, cells, rdr, baf), truth


def write_table(data, path):
    with open(path, 'w') as f:
        f.write('#CHR\tSTART\tEND\tCELL\tRDR\tBAF\n')
        for i in range(data.nbins):
            for j, e in enumerate(data.cells):
                f.write('{}\t{}\t{}\t{}\t{:.6g}\t{:.6g}\n'.format(data.chrs[i], data.starts[i], data.ends[i], e, data.rdr[i, j], data.baf[i, j]))


def adjusted_rand(a, b):
    """Adjusted Rand index of two labelings (1 = identical partitions)."""
    a, b = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb = (lambda x: x * (x - 1) / 2.0)
    sij, sa, sb, n = comb(table).sum(), comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum(), comb(len(a))
    expected = sa * sb / n if n else 0.0
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == expected else float((sij - expected) / (mx - expected))


def parse_args(args=None):
    p = argparse.ArgumentParser(description='States of bins from an HMM on RDR and BAF (HATCHet2-style locality-aware clustering). '
                                            'Prints one line per bin: CHR START END STATE MAXPOST (posterior of the state).')
    p.add_argument('INPUT', nargs='?', default=None, help='Table of RDR and BAF per bin and cell: header CHR START END CELL RDR BAF '
                   '[A_COUNT B_COUNT COUNT], or 6 columns without header, or CHISEL combo.tsv')
    p.add_argument('--demo', action='store_true', help='Run on a toy dataset (2 chromosomes, 12 cells, a gain in half of the cells and an LOH deletion)')
    p.add_argument('--demo-out', default=None, help='With --demo: also write the toy table here (an input example to edit)')
    p.add_argument('-o', '--output', default=None, help='Per-bin states (default: stdout)')
    p.add_argument('--report', default=None, help='Per-K model-selection table (default: only printed to stderr)')
    p.add_argument('-K', '--exactk', type=int, default=0, help='Fit only this K (default: choose in [mink, maxk])')
    p.add_argument('--mink', type=int, default=5, help='Smallest K (default: 5, as HATCHet2)')
    p.add_argument('--maxk', type=int, default=30, help='Largest K (default: 30, as HATCHet2)')
    p.add_argument('--select', default='bic', choices=CRITERIA, help='Choose K by BIC (lowest) or silhouette (highest) (default: bic)')
    p.add_argument('--restarts', type=int, default=10, help='Restarts per K (default: 10)')
    p.add_argument('--tau', type=float, default=1e-6, help='Initial probability of moving to another state (default: 1e-6)')
    p.add_argument('--transmat', default='diag', choices=TRANSMATS, help='diag: one learned stay probability; fixed: tau kept; free: full matrix (default: diag)')
    p.add_argument('--decode', default='map', choices=DECODINGS, help='map: most probable state per bin; viterbi: most probable path (default: map)')
    p.add_argument('--baf', default='oriented', choices=BAFS, help='BAF as given, or mirrored to min(BAF, 1 - BAF) (default: oriented)')
    p.add_argument('--split', default='gaps', choices=SPLITS, help='Independent sequences: per chromosome and at gaps between bins, or per chromosome (default: gaps)')
    p.add_argument('--groups', default=None, help='CELL GROUP per line: cluster on pseudobulk groups instead of single cells')
    p.add_argument('--bicparams', default='hmmlearn', choices=('hmmlearn', 'tied'), help='Parameter count for BIC (default: hmmlearn, as HATCHet2)')
    p.add_argument('--tiedupdate', default='exact', choices=('exact', 'hatchet'), help="Stay-probability update for --transmat diag (default: exact; 'hatchet' reproduces HATCHet2's last-sequence weighting)")
    p.add_argument('--iter', type=int, default=10, help='Maximum EM iterations (default: 10)')
    p.add_argument('--seed', type=int, default=0, help='Restart r uses random seed SEED + r (default: 0)')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Parallel jobs over K (default: 1)')
    a = p.parse_args(args)
    if not a.demo and a.INPUT is None:
        p.error('give an INPUT table or --demo')
    if a.exactk:
        a.mink = a.maxk = a.exactk
    return a


def main(args=None):
    a = parse_args(args)
    if a.demo:
        data, truth = demo_data(a.seed)
        if a.demo_out:
            write_table(data, a.demo_out)
            _fallback_log('toy table written to {}'.format(a.demo_out))
        if a.mink == 5 and a.maxk == 30 and not a.exactk:
            a.mink, a.maxk = 1, 6                                 # the toy has 3 states
    else:
        data, truth = read_table(a.INPUT), None
    groups = read_groups(a.groups, data.cells) if a.groups else None
    X, names = make_samples(data, groups=groups, baf=a.baf)
    lengths = sequence_lengths(data.chrs, data.starts, data.ends, a.split)
    _fallback_log('{} bins in {} sequences x {} samples ({} features); K = {}..{}, {} restarts, {}, {}, {}'.format(
        data.nbins, len(lengths), len(names), X.shape[1], a.mink, min(a.maxk, data.nbins), a.restarts, a.select, a.transmat, a.decode))
    res = select_model(X, lengths, mink=a.mink, maxk=a.maxk, restarts=a.restarts, tau=a.tau, transmat=a.transmat,
                       decode=a.decode, criterion=a.select, seed=a.seed, jobs=a.jobs, n_iter=a.iter,
                       bic_params=a.bicparams, tied_update=a.tiedupdate)
    sys.stderr.write('\nModel selection (* = chosen):\n' + format_table(res) + '\n')
    if a.report:
        with open(a.report, 'w') as f:
            f.write(format_table(res).replace(' *', '\t*') + '\n')
    model, labels = res['model'], relabel_by_position(res['labels'])
    post = model.posteriors(X, lengths)
    maxpost = post[np.arange(len(labels)), res['labels']]

    # state summary: bins, Mb, mean over samples of each state's fitted BAF and RDR
    sys.stderr.write('\nStates (fitted means averaged over samples; per sample below when <= 8 samples):\n')
    sys.stderr.write('STATE\tBINS\tMB\tMEAN_BAF\tMEAN_RDR\n')
    order = []
    for l in res['labels']:
        if l not in order:
            order.append(l)
    for new, old in enumerate(order):
        sel = res['labels'] == old
        mb = (data.ends[sel] - data.starts[sel]).sum() / 1e6
        sys.stderr.write('{}\t{}\t{:.3g}\t{:.3f}\t{:.3f}\n'.format(new, sel.sum(), mb, model.means[old, 0::2].mean(), model.means[old, 1::2].mean()))
    if len(names) <= 8:
        sys.stderr.write('STATE\t' + '\t'.join('{}:BAF\t{}:RDR'.format(s, s) for s in names) + '\n')
        for new, old in enumerate(order):
            sys.stderr.write('{}\t'.format(new) + '\t'.join('{:.3f}'.format(v) for v in model.means[old]) + '\n')
    if truth is not None:
        sys.stderr.write('\nDemo truth (0 normal, 1 gain in cells 0-5, 2 LOH deletion) vs states:\n')
        for t in sorted(set(truth)):
            counts = {s: int(((truth == t) & (labels == s)).sum()) for s in sorted(set(labels))}
            sys.stderr.write('  truth {}: '.format(t) + ', '.join('state {} x{}'.format(s, c) for s, c in counts.items() if c) + '\n')
        sys.stderr.write('  adjusted Rand index: {:.3f}\n'.format(adjusted_rand(truth, labels)))

    out = open(a.output, 'w') if a.output else sys.stdout
    try:
        out.write('#CHR\tSTART\tEND\tSTATE\tMAXPOST\n')
        for i in range(data.nbins):
            out.write('{}\t{}\t{}\t{}\t{:.4f}\n'.format(data.chrs[i], data.starts[i], data.ends[i], labels[i], maxpost[i]))
    finally:
        if a.output:
            out.close()


if __name__ == '__main__':
    main()
