#!/usr/bin/env python2.7
"""One HMM over every cell and bin, emitting a (RDR, BAF) pair per cell-bin (chisel_cellhmm).

Every (cell, bin) is one observation of two numbers, its RDR and its BAF. Every cell has its own latent Markov
chain along its bins, over one state space of K states shared by all cells: cells are laid end to end (each cell
split into runs at chromosomes and gaps between bins, as in HMMClusterizer), and every run starts again from the
start distribution, so the "transition" from the last bin of one cell to the first bin of the next does not depend
on the last state. Every state has one emission distribution of (RDR, BAF), so all cell-bins together learn the K
states; the cells are independent given those.

--means learned (default): K states with learned means (RDR mean mu_k, BAF mean nu_k, the same in every cell),
    initialised by constrained k-means: the K centers are picked from the integer grid a|b (a + b <= maxcn)
    -> (RDR (a + b) / ploidy, BAF b / (a + b)) to best fit a sample of the cell-bins (greedy picks, then Lloyd
    steps in which every cluster moves to the best free grid point; distance = negative log-likelihood under the
    counting noise below). Baum-Welch then moves the means freely, so a systematic shift (e.g. from the normal
    reference) is absorbed by the state instead of forcing another one, and data with more structure than grid
    states can split into nearby states. Every state is then labelled with the copy number a|b of the grid point
    that best fits the cell-bins decoded to it (several states may share a label).
--means grid: the states ARE the grid points a|b, with RDR mean gamma_c (a + b) (gamma_c = 1 / the cell's mean copy
    number, learned per cell, started at median RDR / ploidy) and BAF mean b / (a + b); there is no K.

Transitions: homogeneous, fixed a priori (stay with probability --stay, move to each other state with
(1 - stay) / (K - 1)), and start probabilities uniform 1 / K; --fit-transitions learns one tied stay probability and
the start probabilities instead.

Emission variances follow the counting noise of the bin, as HMMClusterizer's counting noise, under the state's mean m:
    RDR: tau2_r + phi_r (T_N / (T_c N_b)) m + m^2 / N_b      (the cell's Poisson reads, the normal's reads)
    BAF: tau2_b + phi_b m (1 - m) / n_cb                       (binomial SNP reads; n_cb = A + B)
phi >= 1 (overdispersion) and tau2 >= 1e-6 (extra variance) are fitted per cell (--noise percell) or shared by all
cells (--noise shared) by weighted least squares of the squared residuals, as in HMMClusterizer. BAF means are kept
in [eps, 1 - eps]. --baf binomial uses the binomial likelihood of the B reads instead of the Gaussian BAF.
The normal's read noise is shared by all cells in a bin, and is treated here as independent per cell.

EM: Baum-Welch on all runs at once; the runs of all cells cover the same bins, so the forward-backward recursion runs
over bins with all cells (in chunks) as a batch. M-step: state means (one inverse-variance weighted step per iteration,
the variance depends on the mean), phi and tau2 (and gamma_c, transitions as above). The best parameters are kept.
The report also gives BIC (not used to choose anything yet).

  chisel_cellhmm combo/combo.tsv -K 12 -o calls.tsv --cells cells.tsv --report cellhmm.tsv
  chisel_cellhmm --demo -K 8                # simulated counts with planted events, scored against the truth

calls.tsv has the columns of CHISEL's Caller (CLUSTER = state, CN_STATE = the state's a|b label), so the evaluation
reads it as per-cell calls (there is no clones/mapping.tsv).
"""
from __future__ import division, print_function

import sys
import time
import argparse

import numpy as np

try:
    from HMMClusterizer import read_table, count_samples, sequence_lengths, tied_transmat, BinData, _split, orderchrs
except Exception:
    from chisel.HMMClusterizer import read_table, count_samples, sequence_lengths, tied_transmat, BinData, _split, orderchrs


def _log(msg):
    sys.stderr.write('[cellhmm {}] {}\n'.format(time.strftime('%H:%M:%S'), msg))
    sys.stderr.flush()


# ============================================================================================
# 1. Grid of copy numbers, and data
# ============================================================================================
def make_grid(maxcn):
    """All a|b with a + b <= maxcn, ordered by total copy number then a."""
    return [(a, t - a) for t in range(maxcn + 1) for a in range(t + 1)]


class CellData(object):
    """Bins x cells arrays the model reads, from HMMClusterizer's BinData (e.g. read_combo of combo.tsv).

    rdr: RDR = (C / N_b) / (T_c / T_N); ar, cr: its counting-variance factors (the cell's reads, the normal's
    reads), vr: bins with normal reads; B, n: B reads and SNP reads (A + B), vb: n > 0; baf: B / n."""

    def __init__(self, data):
        X, names, cn = count_samples(data, groups=None, baf='oriented')
        self.data = data
        self.chrs, self.starts, self.ends, self.cells = data.chrs, data.starts, data.ends, list(data.cells)
        self.rdr = np.ascontiguousarray(X[:, 1::2])
        self.ar = np.ascontiguousarray(cn.a[:, 1::2])
        self.cr = np.ascontiguousarray(cn.c[:, 1::2])
        self.vr = np.ascontiguousarray(cn.valid[:, 1::2]).astype(float)
        self.B = np.asarray(data.B, dtype=float)
        self.n = np.asarray(data.A, dtype=float) + self.B
        self.vb = (self.n > 0).astype(float)
        self.baf = np.where(self.n > 0, self.B / np.maximum(self.n, 1.0), 0.5)

    @property
    def shape(self):
        return self.rdr.shape


def read_combo(path):
    """BinData from CHISEL's combo.tsv (CHR START END CELL NORM_COUNT COUNT RDR A_COUNT B_COUNT BAF, with or without
    a # header) into bins x cells arrays with pandas, which holds 2000 cells x thousands of bins in memory where
    HMMClusterizer.read_table (one Python tuple per line) does not. Without pandas: read_table."""
    try:
        import pandas as pd
    except ImportError:
        return read_table(path)
    with open(path) as f:
        first = f.readline().split()
    if not first:
        raise ValueError('{}: empty'.format(path))
    head = first[0].startswith('#') or first[0].upper() in ('CHR', 'CHROM')
    if head:
        names = [x.lstrip('#').upper() for x in first]
        if 'A_COUNT' not in names or 'NORM_COUNT' not in names:
            return read_table(path)
        df = pd.read_csv(path, sep='\t', skiprows=1, header=None, names=names)
    else:
        if len(first) != 10:
            return read_table(path)
        names = ['CHR', 'START', 'END', 'CELL', 'NORM_COUNT', 'COUNT', 'RDR', 'A_COUNT', 'B_COUNT', 'BAF']
        df = pd.read_csv(path, sep='\t', header=None, names=names)
    df['CHR'] = df['CHR'].astype(str)
    df['CELL'] = df['CELL'].astype(str)
    bins = df[['CHR', 'START', 'END']].drop_duplicates()
    order = sorted(range(len(bins)), key=lambda i: (orderchrs(bins['CHR'].iat[i]), int(bins['START'].iat[i]), int(bins['END'].iat[i])))
    bins = bins.iloc[order].reset_index(drop=True)
    cells = sorted(df['CELL'].unique())
    key = (lambda t: t['CHR'] + ':' + t['START'].astype(str) + '-' + t['END'].astype(str))   # old pandas: no MultiIndex.from_frame
    ri = pd.Series(np.arange(len(bins)), index=key(bins).values).reindex(key(df).values).values
    ci = pd.Series(np.arange(len(cells)), index=cells).reindex(df['CELL']).values
    shape = (len(bins), len(cells))
    if len(df) != shape[0] * shape[1] or len(set(zip(ri.tolist(), ci.tolist()))) != len(df):
        raise ValueError('{}: every cell needs every bin exactly once ({} lines, {} bins x {} cells)'.format(path, len(df), shape[0], shape[1]))
    arr = {}
    for k in ('RDR', 'BAF', 'A_COUNT', 'B_COUNT', 'COUNT'):
        x = np.empty(shape)
        x[ri, ci] = df[k].values.astype(float)
        arr[k] = x
    norm = np.zeros(shape[0])
    norm[ri] = df['NORM_COUNT'].values.astype(float)
    return BinData(bins['CHR'].tolist(), bins['START'].values, bins['END'].values, cells, arr['RDR'], arr['BAF'],
                   A=arr['A_COUNT'], B=arr['B_COUNT'], w=arr['COUNT'], norm=norm)


# ============================================================================================
# 2. Batched forward-backward and Viterbi (L bins x C cells x K states)
# ============================================================================================
def forward_backward_batch(logB, startprob, transmat):
    """Scaled forward-backward for C sequences of the same length L at once. Returns (log-likelihood of every
    sequence (C), posteriors (L x C x K), expected transition counts summed over bins and sequences (K x K))."""
    L, C, K = logB.shape
    m = logB.max(axis=2)
    b = np.exp(logB - m[:, :, None])
    alpha = np.empty_like(b)
    c = np.empty((L, C))
    a = startprob[None, :] * b[0]
    c[0] = np.maximum(a.sum(axis=1), 1e-300)
    alpha[0] = a / c[0][:, None]
    for t in range(1, L):
        a = alpha[t - 1].dot(transmat) * b[t]
        c[t] = np.maximum(a.sum(axis=1), 1e-300)
        alpha[t] = a / c[t][:, None]
    loglik = np.log(c).sum(axis=0) + m.sum(axis=0)
    beta = np.empty_like(b)
    beta[-1] = 1.0
    for t in range(L - 2, -1, -1):
        beta[t] = (b[t + 1] * beta[t + 1] / c[t + 1][:, None]).dot(transmat.T)
    if L > 1:
        right = b[1:] * beta[1:] / c[1:, :, None]
        xi = transmat * alpha[:-1].reshape(-1, K).T.dot(right.reshape(-1, K))
    else:
        xi = np.zeros((K, K))
    gamma = alpha
    gamma *= beta
    gamma /= np.maximum(gamma.sum(axis=2, keepdims=True), 1e-300)
    return loglik, gamma, xi


def viterbi_batch(logB, startprob, transmat):
    """Most probable state path of C sequences at once: (log probability (C), paths L x C)."""
    L, C, K = logB.shape
    with np.errstate(divide='ignore'):
        logA = np.log(transmat)
        delta = np.log(startprob)[None, :] + logB[0]
    back = np.zeros((L, C, K), dtype=np.int16)
    for t in range(1, L):
        s = delta[:, :, None] + logA[None, :, :]
        back[t] = s.argmax(axis=1)
        delta = s.max(axis=1) + logB[t]
    path = np.empty((L, C), dtype=int)
    path[-1] = delta.argmax(axis=1)
    rows = np.arange(C)
    for t in range(L - 1, 0, -1):
        path[t - 1] = back[t, rows, path[t]]
    return delta.max(axis=1), path


# ============================================================================================
# 3. Constrained k-means: K centers from the copy-number grid
# ============================================================================================
def grid_costs(d, rows, cols, grid_r, grid_f, tau2=1e-3):
    """Negative log-likelihood (N sampled cell-bins x G grid points) of every sampled cell-bin under every grid point,
    with the counting noise at phi = 1 plus tau2; BAF only where the cell-bin has SNP reads."""
    r, ar, cr, vr = d.rdr[rows, cols], d.ar[rows, cols], d.cr[rows, cols], d.vr[rows, cols]
    f, n, vb = d.baf[rows, cols], d.n[rows, cols], d.vb[rows, cols]
    mr = np.maximum(grid_r, 1e-3)[None, :]
    v = tau2 + ar[:, None] * mr + cr[:, None] * mr * mr
    D = 0.5 * (np.log(v) + (r[:, None] - grid_r[None, :]) ** 2 / v) * vr[:, None]
    vf = tau2 + grid_f[None, :] * (1.0 - grid_f[None, :]) / np.maximum(n, 1.0)[:, None]
    D += 0.5 * (np.log(vf) + (f[:, None] - grid_f[None, :]) ** 2 / vf) * vb[:, None]
    return D


def constrained_kmeans(D, K, max_iter=50):
    """K distinct columns of the cost matrix D (points x grid points) that minimise the summed cost of every point to
    its nearest chosen column: greedy picks (the best single column, then the one that lowers the total most), then
    Lloyd steps: assign every point to its nearest chosen column, move every cluster (largest first) to the free grid
    point with the lowest summed cost of its points. Returns the chosen grid indices (in pick order)."""
    G = D.shape[1]
    if K > G:
        raise ValueError('K = {} states but only {} grid points: raise --maxcn'.format(K, G))
    sel = [int(D.sum(axis=0).argmin())]
    cur = D[:, sel[0]].copy()
    while len(sel) < K:
        gain = np.maximum(cur[:, None] - D, 0.0).sum(axis=0)
        gain[sel] = -1.0
        g = int(gain.argmax())
        sel.append(g)
        cur = np.minimum(cur, D[:, g])
    for _ in range(max_iter):
        lab = D[:, sel].argmin(axis=1)
        sizes = np.bincount(lab, minlength=K)
        new, taken = list(sel), set()
        for k in np.argsort(-sizes, kind='mergesort'):
            if sizes[k] == 0:                                # an empty cluster keeps its grid point if still free
                continue
            cost = D[lab == k].sum(axis=0)
            cost[list(taken)] = np.inf
            new[k] = int(cost.argmin())
            taken.add(new[k])
        for k in range(K):
            if sizes[k] == 0:
                if new[k] in taken:
                    free = [g for g in np.argsort(D.sum(axis=0)) if g not in taken]
                    new[k] = int(free[0])
                taken.add(new[k])
        if new == sel:
            break
        sel = new
    return sel


# ============================================================================================
# 4. The model
# ============================================================================================
def _solve_dispersion(S, prev_t, prev_p, min_tau2, min_phi):
    """tau2, phi from the weighted least-squares sums S = (S00, S01, S11, T0, T1) of r^2 - sc on [1, sa]
    (arrays, one entry per cell), with tau2 >= min_tau2 and phi >= min_phi, as HMMClusterizer."""
    S00, S01, S11, T0, T1 = S
    with np.errstate(divide='ignore', invalid='ignore'):
        det = S00 * S11 - S01 * S01
        ok = (det > 1e-12 * np.maximum(S00 * S11, 1e-300)) & (S11 > 0)
        t = np.where(ok, (S11 * T0 - S01 * T1) / np.where(ok, det, 1.0), min_tau2)
        p = np.where(ok, (S00 * T1 - S01 * T0) / np.where(ok, det, 1.0), min_phi - 1.0)
        low = p < min_phi
        p = np.where(low, min_phi, p)
        t = np.where(low, (T0 - p * S01) / np.where(S00 > 0, S00, 1.0), t)
        low = t < min_tau2
        t = np.where(low, min_tau2, t)
        p = np.where(low, np.maximum((T1 - t * S01) / np.where(S11 > 0, S11, 1.0), min_phi), p)
    keep = ~(S00 > 0)
    return np.where(keep, prev_t, t), np.where(keep, prev_p, p)


class CellHMM(object):
    """See the module docstring. Parameters: mu_r, nu (K, learned means) or gamma (C, grid means); tau2_r, phi_r,
    tau2_b, phi_b (C each, equal across cells with noise='shared'); startprob (K) and stay."""

    def __init__(self, K=None, means='learned', maxcn=6, ploidy=2.0, baf='gaussian', baf_eps=0.02, stay=0.99,
                 fit_transitions=False, n_iter=30, tol=1e-5, noise='percell', decode='map', fit_scale=True,
                 min_tau2=1e-6, min_phi=1.0, chunk_mb=64, init_sample=200000, seed=0):
        for name, v, ok in (('means', means, ('learned', 'grid')), ('baf', baf, ('gaussian', 'binomial')),
                            ('noise', noise, ('percell', 'shared')), ('decode', decode, ('map', 'viterbi'))):
            if v not in ok:
                raise ValueError('{} must be one of {}'.format(name, ok))
        if not 0.0 < stay < 1.0:
            raise ValueError('stay must be in (0, 1)')
        self.grid = np.array(make_grid(maxcn))
        gt = self.grid.sum(axis=1).astype(float)
        self.grid_r = gt / ploidy
        self.grid_f = np.clip(np.where(gt > 0, self.grid[:, 1] / np.maximum(gt, 1.0), 0.5), baf_eps, 1.0 - baf_eps)
        self.means = means
        if means == 'learned':
            if not K or K < 1:
                raise ValueError('--means learned needs the number of states K >= 1')
            if K > len(self.grid):
                raise ValueError('K = {} states but only {} grid points a|b with a + b <= {}: raise --maxcn'.format(K, len(self.grid), maxcn))
            self.K = int(K)
        else:
            self.K = len(self.grid)
        self.ploidy, self.baf_model, self.baf_eps, self.noise, self.decode_alg = ploidy, baf, baf_eps, noise, decode
        self.stay0, self.fit_transitions, self.n_iter, self.tol, self.fit_scale = stay, fit_transitions, n_iter, tol, fit_scale
        self.min_tau2, self.min_phi, self.chunk_mb = min_tau2, min_phi, chunk_mb
        self.init_sample, self.seed = init_sample, seed
        self.history = []

    # ---- parameters
    def init(self, d):
        C, K = d.shape[1], self.K
        self.startprob = np.full(K, 1.0 / K)
        self.stay = float(self.stay0)
        self.tau2_r, self.phi_r = np.full(C, 1e-4), np.ones(C)
        self.tau2_b, self.phi_b = np.full(C, 1e-4), np.ones(C)
        if self.means == 'grid':
            self.state_cn = self.grid.copy()
            self.init_cn = self.grid.copy()
            g = np.empty(C)
            for j in range(C):                              # most of a genome at the base ploidy: median RDR / ploidy
                r = d.rdr[d.vr[:, j] > 0, j]
                g[j] = np.median(r) / self.ploidy if len(r) and np.median(r) > 0 else 1.0 / self.ploidy
            self.gamma = g
            self.mu_r, self.nu = self.grid_r.copy(), self.grid_f.copy()
            return self
        # learned means: constrained k-means on a sample of the cell-bins with normal reads
        rng = np.random.RandomState(self.seed)
        rows, cols = np.nonzero(d.vr > 0)
        if len(rows) > self.init_sample:
            pick = rng.choice(len(rows), self.init_sample, replace=False)
            rows, cols = rows[pick], cols[pick]
        self.sample = (rows, cols)
        self.sample_costs = grid_costs(d, rows, cols, self.grid_r, self.grid_f)
        sel = constrained_kmeans(self.sample_costs, K)
        self.init_cn = self.grid[sel]
        self.state_cn = self.init_cn.copy()
        self.mu_r, self.nu = self.grid_r[sel].copy(), self.grid_f[sel].copy()
        self.gamma = np.full(C, np.nan)
        _log('constrained k-means on {} cell-bins: initial states {}'.format(
            len(rows), ' '.join('{}|{}'.format(a, b) for a, b in self.init_cn)))
        return self

    @property
    def transmat(self):
        return tied_transmat(self.stay, self.K)

    def get_params(self):
        return dict((k, np.copy(getattr(self, k))) for k in ('startprob', 'stay', 'gamma', 'mu_r', 'nu', 'tau2_r', 'phi_r', 'tau2_b', 'phi_b'))

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, np.copy(v) if k != 'stay' else float(v))

    def n_parameters(self, C):
        means = 2 * self.K if self.means == 'learned' else (C if self.fit_scale else 0)
        disp = (2 if self.baf_model == 'gaussian' else 1) * 2 * (C if self.noise == 'percell' else 1)
        trans = (self.K - 1 + 1) if self.fit_transitions else 0
        return means + disp + trans

    # ---- emissions of bins i:j, cells c0:c1 (L x Cc x K); with parts=True also what the M-step needs
    def emissions(self, d, i, j, c0, c1, parts=False):
        cs = slice(c0, c1)
        if self.means == 'grid':
            mu = self.gamma[cs][None, :, None] * self.grid.sum(axis=1)[None, None, :]
            zero = self.grid.sum(axis=1) == 0
        else:
            mu = np.broadcast_to(self.mu_r[None, None, :], (1, c1 - c0, self.K))
            zero = np.zeros(self.K, dtype=bool)
        mr = np.maximum(mu, 1e-3)
        r = d.rdr[i:j, cs][:, :, None]
        sa_r = d.ar[i:j, cs][:, :, None] * mr
        sc_r = d.cr[i:j, cs][:, :, None] * mr * mr
        var_r = self.tau2_r[cs][None, :, None] + self.phi_r[cs][None, :, None] * sa_r + sc_r
        vr = d.vr[i:j, cs][:, :, None]
        res_r = r - mu
        ll = -0.5 * (np.log(2 * np.pi * var_r) + res_r * res_r / var_r) * vr
        vb = d.vb[i:j, cs][:, :, None]
        nu = self.nu[None, None, :]
        if self.baf_model == 'gaussian':
            n = np.maximum(d.n[i:j, cs], 1.0)[:, :, None]
            sa_b = nu * (1.0 - nu) / n
            var_b = self.tau2_b[cs][None, :, None] + self.phi_b[cs][None, :, None] * sa_b
            f = d.baf[i:j, cs][:, :, None]
            res_b = f - nu
            llb = -0.5 * (np.log(2 * np.pi * var_b) + res_b * res_b / var_b)
            if zero.any():
                llb[:, :, zero] = 0.0                        # 0|0 (grid means): BAF uniform on [0, 1]
        else:
            Bc, n = d.B[i:j, cs][:, :, None], d.n[i:j, cs][:, :, None]
            llb = Bc * np.log(nu) + (n - Bc) * np.log(1.0 - nu)
            if zero.any():
                llb[:, :, zero] = (n * np.log(0.5))[:, :, 0][:, :, None]
        ll = ll + llb * vb
        if not parts:
            return ll
        out = {'ll': ll, 'r': r, 'res_r': res_r, 'var_r': var_r, 'sa_r': sa_r, 'sc_r': sc_r, 'vr': vr,
               'vb': vb * (~zero)[None, None, :]}
        if self.baf_model == 'gaussian':
            out.update({'f': f, 'res_b': res_b, 'var_b': var_b, 'sa_b': sa_b})
        else:
            out.update({'Bc': Bc, 'n': n})
        return out

    def _chunks(self, d, lengths):
        C = d.shape[1]
        for i, j in _split(d.shape[0], lengths):
            step = max(1, min(C, int(self.chunk_mb * 2 ** 20 // (8 * self.K * (j - i)))))
            for c0 in range(0, C, step):
                yield i, j, c0, min(C, c0 + step)

    # ---- E-step statistics
    def estep(self, d, lengths):
        C, K = d.shape[1], self.K
        st = {'ll': np.zeros(C), 'start': np.zeros(K), 'trans': np.zeros((K, K)), 'occ': np.zeros((C, K)),
              'gnum': np.zeros(C), 'gden': np.zeros(C), 'Sr': np.zeros((5, C)), 'Sb': np.zeros((5, C)),
              'rnum': np.zeros(K), 'rden': np.zeros(K), 'bnum': np.zeros(K), 'bden': np.zeros(K)}
        A = self.transmat
        tot = self.grid.sum(axis=1).astype(float)
        for i, j, c0, c1 in self._chunks(d, lengths):
            p = self.emissions(d, i, j, c0, c1, parts=True)
            ll, gam, xi = forward_backward_batch(p['ll'], self.startprob, A)
            st['ll'][c0:c1] += ll
            st['start'] += gam[0].sum(axis=0)
            st['trans'] += xi
            st['occ'][c0:c1] += gam.sum(axis=0)
            G = gam * p['vr'] / p['var_r']                  # inverse-variance weights of the RDRs
            if self.means == 'grid':                        # gamma_c: weighted least squares of RDR on a + b
                st['gnum'][c0:c1] += (G * p['r'] * tot).sum(axis=(0, 2))
                st['gden'][c0:c1] += (G * tot * tot).sum(axis=(0, 2))
            else:                                           # state means: inverse-variance weighted averages
                st['rnum'] += (G * p['r']).sum(axis=(0, 1))
                st['rden'] += G.sum(axis=(0, 1))
                gb = gam * p['vb']
                if self.baf_model == 'gaussian':
                    Gb = gb / p['var_b']
                    st['bnum'] += (Gb * p['f']).sum(axis=(0, 1))
                    st['bden'] += Gb.sum(axis=(0, 1))
                else:                                       # binomial: pooled B / (A + B)
                    st['bnum'] += (gb * p['Bc']).sum(axis=(0, 1))
                    st['bden'] += (gb * p['n']).sum(axis=(0, 1))
            st['Sr'][:, c0:c1] += self._disp_sums(gam * p['vr'], p['var_r'], p['res_r'] ** 2 - p['sc_r'], p['sa_r'])
            if self.baf_model == 'gaussian':
                st['Sb'][:, c0:c1] += self._disp_sums(gam * p['vb'], p['var_b'], p['res_b'] ** 2, p['sa_b'])
        return st

    @staticmethod
    def _disp_sums(g, var, y, sa):
        w = g / np.maximum(var, 1e-300) ** 2
        return np.array([w.sum(axis=(0, 2)), (w * sa).sum(axis=(0, 2)), (w * sa * sa).sum(axis=(0, 2)),
                         (w * y).sum(axis=(0, 2)), (w * y * sa).sum(axis=(0, 2))])

    def mstep(self, st):
        if self.fit_transitions:
            self.startprob = (st['start'] + 1.0) / (st['start'].sum() + self.K)
            tot = st['trans'].sum()
            if tot > 0 and self.K > 1:
                self.stay = float(np.clip(np.trace(st['trans']) / tot, 1e-6, 1.0 - 1e-9))
        if self.means == 'grid':
            if self.fit_scale:
                ok = st['gden'] > 0
                self.gamma = np.maximum(np.where(ok, st['gnum'] / np.where(ok, st['gden'], 1.0), self.gamma), 1e-3)
        else:                                               # a state without posterior mass keeps its means
            ok = st['rden'] > 0
            self.mu_r = np.maximum(np.where(ok, st['rnum'] / np.where(ok, st['rden'], 1.0), self.mu_r), 1e-3)
            ok = st['bden'] > 0
            self.nu = np.clip(np.where(ok, st['bnum'] / np.where(ok, st['bden'], 1.0), self.nu), self.baf_eps, 1.0 - self.baf_eps)
        Sr, Sb = st['Sr'], st['Sb']
        if self.noise == 'shared':
            Sr = np.repeat(Sr.sum(axis=1, keepdims=True), Sr.shape[1], axis=1)
            Sb = np.repeat(Sb.sum(axis=1, keepdims=True), Sb.shape[1], axis=1)
        self.tau2_r, self.phi_r = _solve_dispersion(Sr, self.tau2_r, self.phi_r, self.min_tau2, self.min_phi)
        if self.baf_model == 'gaussian':
            self.tau2_b, self.phi_b = _solve_dispersion(Sb, self.tau2_b, self.phi_b, self.min_tau2, self.min_phi)

    def fit(self, d, lengths):
        nobs = float(np.prod(d.shape))
        best, prev = None, None
        self.history = []
        for it in range(self.n_iter):
            t0 = time.time()
            st = self.estep(d, lengths)
            ll = float(st['ll'].sum())
            self.history.append(ll)
            if best is None or ll > best[0]:
                best = (ll, self.get_params(), st)
            means = ('median ploidy {:.3g}'.format(np.median(1.0 / self.gamma)) if self.means == 'grid' else
                     'RDR means ' + ' '.join('{:.2f}'.format(v) for v in self.mu_r))
            _log('EM iteration {}: log-likelihood {:.8g} ({:.5f} per cell-bin), {}, median phi_r {:.3g}, {:.1f} s'.format(
                it + 1, ll, ll / nobs, means, np.median(self.phi_r), time.time() - t0))
            if prev is not None and (ll - prev) / nobs < self.tol:
                break
            prev = ll
            self.mstep(st)
        self.loglik, params, self.last_stats = best
        self.set_params(params)
        self.bic = -2.0 * self.loglik + self.n_parameters(d.shape[1]) * np.log(nobs)
        return self

    def decode(self, d, lengths):
        """States (bins x cells) and the posterior of the decoded state; with learned means, also labels every state
        with the grid point a|b that best fits the sampled cell-bins decoded to it (state_cn)."""
        labels = np.empty(d.shape, dtype=np.int16)
        maxpost = np.empty(d.shape, dtype=np.float32)
        A = self.transmat
        for i, j, c0, c1 in self._chunks(d, lengths):
            logB = self.emissions(d, i, j, c0, c1)
            _, gam, _ = forward_backward_batch(logB, self.startprob, A)
            path = viterbi_batch(logB, self.startprob, A)[1] if self.decode_alg == 'viterbi' else gam.argmax(axis=2)
            labels[i:j, c0:c1] = path
            maxpost[i:j, c0:c1] = gam[np.arange(j - i)[:, None], np.arange(c1 - c0)[None, :], path]
        if self.means == 'learned':
            rows, cols = self.sample
            lab = labels[rows, cols]
            cn = self.init_cn.copy()
            for k in range(self.K):
                sel = lab == k
                if sel.any():
                    cn[k] = self.grid[int(self.sample_costs[sel].sum(axis=0).argmin())]
            self.state_cn = cn
        return labels, maxpost


# ============================================================================================
# 5. Output
# ============================================================================================
def state_names(model):
    return ['{}|{}'.format(a, b) for a, b in model.state_cn]


def write_calls(path, d, model, labels):
    """CHISEL Caller's calls.tsv columns: CHR START END CELL NORM_COUNT COUNT RDR A_COUNT B_COUNT BAF CLUSTER CN_STATE."""
    data, names = d.data, state_names(model)
    with open(path, 'w') as f:
        f.write('#CHR\tSTART\tEND\tCELL\tNORM_COUNT\tCOUNT\tRDR\tA_COUNT\tB_COUNT\tBAF\tCLUSTER\tCN_STATE\n')
        for i in range(len(d.chrs)):
            pre = '{}\t{}\t{}\t'.format(d.chrs[i], d.starts[i], d.ends[i])
            norm = int(data.norm[i])
            f.write(''.join('{}{}\t{}\t{}\t{:g}\t{}\t{}\t{:g}\t{}\t{}\n'.format(
                pre, e, norm, int(data.w[i, j]), data.rdr[i, j], int(data.A[i, j]), int(data.B[i, j]), data.baf[i, j],
                labels[i, j], names[labels[i, j]]) for j, e in enumerate(d.cells)))


def write_cells(path, d, model, labels):
    """One line per cell: gamma and ploidy (grid means; NA with learned means), mean copy number of its calls,
    log-likelihood, noise, and the fraction of bins not called 1|1."""
    st = model.last_stats
    tot = model.state_cn.sum(axis=1)
    normal = (model.state_cn[:, 0] == 1) & (model.state_cn[:, 1] == 1)
    fmt = (lambda v: 'NA' if not np.isfinite(v) else '{:.5g}'.format(v))
    with open(path, 'w') as f:
        f.write('#CELL\tGAMMA\tPLOIDY\tMEAN_CN\tLOGLIK\tPHI_RDR\tTAU2_RDR\tPHI_BAF\tTAU2_BAF\tFRAC_ABERRANT\n')
        for j, e in enumerate(d.cells):
            f.write('{}\t{}\t{}\t{:.4g}\t{:.8g}\t{:.4g}\t{:.4g}\t{:.4g}\t{:.4g}\t{:.4g}\n'.format(
                e, fmt(model.gamma[j]), fmt(1.0 / model.gamma[j]), tot[labels[:, j]].mean(), st['ll'][j], model.phi_r[j],
                model.tau2_r[j], model.phi_b[j], model.tau2_b[j], float((~normal[labels[:, j]]).mean())))


def format_report(model, labels, nobs):
    lines = ['# means {}, K = {}, BAF {}, noise {}, transitions {} (stay {:.6g})'.format(
                 model.means, model.K, model.baf_model, model.noise, 'fitted' if model.fit_transitions else 'fixed', model.stay),
             '# log-likelihood per EM iteration: ' + ' '.join('{:.8g}'.format(v) for v in model.history),
             '# chosen log-likelihood {:.8g}, parameters {}, cell-bins {}, BIC {:.8g}'.format(
                 model.loglik, model.n_parameters(labels.shape[1]), int(nobs), model.bic),
             'STATE\tCN\tINIT_CN\tMEAN_RDR\tMEAN_BAF\tSTART_PROB\tFRAC_CELL_BINS']
    frac = np.bincount(labels.ravel().astype(int), minlength=model.K) / float(labels.size)
    names = state_names(model)
    for k in range(model.K):
        rdr = '{:.4g}'.format(model.mu_r[k]) if model.means == 'learned' else 'gamma*{}'.format(int(model.state_cn[k].sum()))
        lines.append('{}\t{}\t{}|{}\t{}\t{:.4g}\t{:.4g}\t{:.4g}'.format(
            k, names[k], model.init_cn[k][0], model.init_cn[k][1], rdr, model.nu[k], model.startprob[k], frac[k]))
    return '\n'.join(lines)


# ============================================================================================
# 6. Simulated counts (--demo)
# ============================================================================================
def demo_counts(seed=0, ncells=300, reads=20.0, snpreads=4.0, normal_reads=3000.0, switch=0.0, err=0.01, shift=0.0):
    """Two chromosomes (800 + 400 bins of 30 kb), 4 clones (50%, 25%, 15%, 10% of the cells), events of
    5, 10, 20, 50 and 100 bins (gains 2|1 / 1|2, losses 1|0 / 0|1, LOH 2|0) planted in one clone or the clonal
    trunk. Per cell: about `reads` reads and `snpreads` SNP reads per bin at 1|1, depth varying by cell
    (log-normal, sd 0.3), a GC-like bias per bin shared with the normal. switch: per-bin probability of a phasing
    switch error (the B allele of the phasing flips from that bin on, in every cell); err: base error of BAF;
    shift: the cells' bias differs from the normal's by a factor exp(shift * z_b), z_b a smooth standard-normal
    profile along the genome (a systematic mismatch with the normal reference).
    Returns (BinData, true CN as (bins x cells x 2), list of events (dict), phasing flips)."""
    rng = np.random.RandomState(seed)
    chrs = ['chr1'] * 800 + ['chr2'] * 400
    starts = np.concatenate([np.arange(800), np.arange(400)]) * 30000
    ends = starts + 30000
    nb = len(chrs)
    clone = rng.choice(4, size=ncells, p=[0.5, 0.25, 0.15, 0.10])
    cna = np.ones((nb, ncells), dtype=int)
    cnb = np.ones((nb, ncells), dtype=int)
    kinds = [(2, 1), (1, 0), (1, 2), (0, 1), (2, 0)]
    events, pos = [], {'chr1': 20, 'chr2': 820}
    for size in (5, 10, 20, 50, 100):
        for o in ['trunk', 0, 1, 2, 3]:
            ch = 'chr1' if pos['chr1'] + size < 790 else 'chr2'
            s = pos[ch]
            if s + size >= (790 if ch == 'chr1' else 1190):
                continue
            a, b = kinds[len(events) % len(kinds)]
            carriers = np.ones(ncells, dtype=bool) if o == 'trunk' else clone == o
            cna[s:s + size][:, carriers] = a
            cnb[s:s + size][:, carriers] = b
            events.append({'start': s, 'end': s + size, 'size': size, 'cn': (a, b), 'owner': o, 'carriers': carriers})
            pos[ch] = s + size + 8
    bias = np.exp(rng.normal(0, 0.15, nb))
    z = np.convolve(rng.normal(0, 1, nb + 60), np.ones(60) / np.sqrt(60), mode='valid')[:nb]
    cellbias = bias * np.exp(shift * z)
    depth = np.exp(rng.normal(0, 0.3, ncells))
    tot = cna + cnb
    C = rng.poisson(reads / 2.0 * cellbias[:, None] * depth[None, :] * tot)
    N = rng.poisson(normal_reads * bias)
    n = rng.poisson(snpreads / 2.0 * depth[None, :] * tot)
    flip = (np.cumsum(rng.random_sample(nb) < switch) % 2).astype(bool)
    pb = np.where(flip[:, None], cna, cnb)
    p = np.where(tot > 0, pb / np.maximum(tot, 1).astype(float), 0.5)
    p = p * (1 - err) + (1 - p) * err
    B = rng.binomial(n, p)
    A = n - B
    TC, TN = C.sum(axis=0).astype(float), float(N.sum())
    rdr = np.where(N[:, None] > 0, C / np.maximum(N, 1)[:, None] / (TC / TN)[None, :], 0.0)
    baf = np.where(n > 0, B / np.maximum(n, 1).astype(float), 0.0)
    cells = ['cell{:04d}'.format(j) for j in range(ncells)]
    data = BinData(chrs, starts, ends, cells, rdr, baf, A=A, B=B, w=C, norm=N)
    return data, np.stack([cna, cnb], axis=2), events, flip


def score_demo(model, labels, truth, events, flip):
    """Per event: % of carrier cell-bins whose call shows the event at the allele-specific level (as
    eval_detection.py: gains called >= the gained allele's CN, losses <=, alleles sorted per call); exact
    allele-specific accuracy; calls off 1|1 among non-event cell-bins."""
    st = model.state_cn
    ca, cb = st[labels, 0], st[labels, 1]
    hi, lo = np.maximum(ca, cb), np.minimum(ca, cb)
    th, tl = truth.max(axis=2), truth.min(axis=2)
    exact = (hi == th) & (lo == tl)
    normal = (truth[:, :, 0] == 1) & (truth[:, :, 1] == 1)
    print('\nAllele-specific accuracy over all cell-bins: {:.2%}; non-event cell-bins called off 1|1: {:.2%}'.format(
        exact.mean(), (~exact[normal]).mean()))
    print('Phasing switch errors: {}'.format(int(np.abs(np.diff(flip.astype(int))).sum())))
    print('\nEVENT\tBINS\tCN\tCARRIERS\tDETECTED\tEXACT')
    rows = []
    for k, ev in enumerate(events):
        s, e, car = ev['start'], ev['end'], ev['carriers']
        a, b = ev['cn']
        h, l = max(a, b), min(a, b)
        H, Lw = hi[s:e][:, car], lo[s:e][:, car]
        det = (H >= h).mean() if a + b > 2 else (Lw <= l).mean()
        rows.append((ev['size'], det))
        print('{}\t{}\t{}|{}\t{}\t{:.0%}\t{:.0%}'.format(k, ev['size'], a, b, int(car.sum()), det, exact[s:e][:, car].mean()))
    print('\nMean detection by event size: ' + ', '.join('{} bins {:.0%}'.format(sz, np.mean([d for z, d in rows if z == sz]))
                                                        for sz in sorted(set(z for z, _ in rows))))


# ============================================================================================
# 7. Command line
# ============================================================================================
def parse_args(args=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='See the module docstring of chisel/CellHMM.py.')
    p.add_argument('INPUT', nargs='?', default=None, help='combo.tsv (or a table with CHR START END CELL RDR BAF A_COUNT B_COUNT COUNT NORM_COUNT)')
    p.add_argument('-K', '--states', type=int, default=None, help='Number of states (needed with --means learned)')
    p.add_argument('-o', '--output', default='calls.tsv', help='Calls in CHISEL calls.tsv format (default: calls.tsv)')
    p.add_argument('--cells', default=None, help='Per-cell noise, mean copy number (default: none)')
    p.add_argument('--report', default=None, help='EM history, BIC and per-state summary (default: printed to stderr)')
    p.add_argument('--means', default='learned', choices=('learned', 'grid'), help='learned: K states with learned means, started from the copy-number grid; grid: the states are the grid points a|b (default: learned)')
    p.add_argument('--maxcn', type=int, default=6, help='Largest total copy number of the grid (default: 6)')
    p.add_argument('--ploidy', type=float, default=2.0, help='RDR of a|b on the grid = (a + b) / ploidy (default: 2)')
    p.add_argument('--baf', default='gaussian', choices=('gaussian', 'binomial'), help='BAF likelihood: Gaussian with counting variance, or binomial of the B reads (default: gaussian)')
    p.add_argument('--baf-eps', type=float, default=0.02, help='BAF means kept in [eps, 1 - eps] (default: 0.02)')
    p.add_argument('--stay', type=float, default=0.99, help='Probability of staying in a state from one bin to the next in a cell (default: 0.99)')
    p.add_argument('--fit-transitions', action='store_true', help='Learn the stay probability and start probabilities (default: fixed stay, uniform start)')
    p.add_argument('--noise', default='percell', choices=('percell', 'shared'), help='Overdispersion and extra variance per cell or shared (default: percell)')
    p.add_argument('--fixed-scale', action='store_true', help='--means grid: keep every cell\'s gamma at median RDR / ploidy')
    p.add_argument('--decode', default='map', choices=('map', 'viterbi'), help='Decoding (default: map)')
    p.add_argument('--split', default='gaps', choices=('gaps', 'chromosomes'), help='Runs split at chromosomes and gaps between bins, or chromosomes only (default: gaps)')
    p.add_argument('--iter', type=int, default=30, help='Maximum EM iterations (default: 30)')
    p.add_argument('--tol', type=float, default=1e-5, help='Stop when the log-likelihood gains less than this per cell-bin (default: 1e-5)')
    p.add_argument('--init-sample', type=int, default=200000, help='Cell-bins sampled for the constrained k-means (default: 200000)')
    p.add_argument('--seed', type=int, default=0, help='Seed of the k-means sample and of --demo (default: 0)')
    p.add_argument('--chunk-mb', type=float, default=64, help='Size of one bins x cells x states array in a batch (default: 64)')
    p.add_argument('--demo', action='store_true', help='Simulated counts with planted events, scored against the truth')
    p.add_argument('--demo-cells', type=int, default=300, help='--demo: cells (default: 300)')
    p.add_argument('--demo-snpreads', type=float, default=4.0, help='--demo: SNP reads per cell per bin at 1|1 (default: 4)')
    p.add_argument('--demo-reads', type=float, default=20.0, help='--demo: reads per cell per bin at 1|1 (default: 20)')
    p.add_argument('--demo-switch', type=float, default=0.0, help='--demo: per-bin phasing switch-error probability (default: 0)')
    p.add_argument('--demo-err', type=float, default=0.01, help='--demo: probability that a SNP read shows the other allele (default: 0.01)')
    p.add_argument('--demo-shift', type=float, default=0.0, help='--demo: sd of a smooth log-depth mismatch between the cells and the normal (default: 0)')
    a = p.parse_args(args)
    if not a.demo and a.INPUT is None:
        p.error('give an INPUT combo.tsv or --demo')
    if a.means == 'learned' and not a.states:
        p.error('--means learned needs -K / --states')
    return a


def main(args=None):
    a = parse_args(args)
    if a.demo:
        data, truth, events, flip = demo_counts(seed=a.seed, ncells=a.demo_cells, reads=a.demo_reads, snpreads=a.demo_snpreads,
                                                switch=a.demo_switch, err=a.demo_err, shift=a.demo_shift)
    else:
        _log('reading {}'.format(a.INPUT))
        data = read_combo(a.INPUT)
    d = CellData(data)
    lengths = sequence_lengths(d.chrs, d.starts, d.ends, a.split)
    model = CellHMM(K=a.states, means=a.means, maxcn=a.maxcn, ploidy=a.ploidy, baf=a.baf, baf_eps=a.baf_eps, stay=a.stay,
                    fit_transitions=a.fit_transitions, n_iter=a.iter, tol=a.tol, noise=a.noise, decode=a.decode,
                    fit_scale=not a.fixed_scale, chunk_mb=a.chunk_mb, init_sample=a.init_sample, seed=a.seed)
    _log('{} bins in {} runs x {} cells; {} states ({} means, grid a + b <= {}); BAF {}; stay {} ({})'.format(
        d.shape[0], len(lengths), d.shape[1], model.K, a.means, a.maxcn, a.baf, a.stay, 'fitted' if a.fit_transitions else 'fixed'))
    model.init(d).fit(d, lengths)
    labels, maxpost = model.decode(d, lengths)
    rep = format_report(model, labels, np.prod(d.shape))
    if a.report:
        with open(a.report, 'w') as f:
            f.write(rep + '\n')
    sys.stderr.write(rep + '\n')
    if a.demo:
        score_demo(model, labels, truth, events, flip)
        return model, labels
    write_calls(a.output, d, model, labels)
    _log('calls written to {}'.format(a.output))
    if a.cells:
        write_cells(a.cells, d, model, labels)
    return model, labels


if __name__ == '__main__':
    main()
