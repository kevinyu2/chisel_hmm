#!/usr/bin/env python3
"""
Score CHISEL calls (calls.tsv) against the truth written by simulate.py.

Accuracy is base-pair weighted over cells x genome: for every bin, the part of
it covered by each true segment is scored separately. Levels:
  total           a+b equal
  allele          {a, b} equal (unordered, i.e. allele-specific)
Also scored inside every focal event for the cells of the clone that carries it,
and clone recovery (adjusted Rand index of CHISEL clones vs true populations).

Usage: evaluate.py SIMDIR NAME:calls.tsv[:mapping.tsv] [...]
"""
import sys, os
from collections import defaultdict, Counter
from math import comb

FOCAL = [('A', 'chr1', 15500000, 16500000, 'A: chr1 focal amp 3|1'),
         ('B', 'chr1', 8500000, 9500000, 'B: chr1 focal cnLOH 2|0'),
         ('B', 'chr2', 11000000, 12000000, 'B: chr2 focal del 1|0')]


def read_truth(simdir):
    truth = defaultdict(list)
    with open(os.path.join(simdir, 'truth.tsv')) as f:
        for l in f:
            if l[0] != '#':
                p, c, s, e, a, b = l.split()
                truth[p].append((c, int(s), int(e), int(a), int(b)))
    pop = {}
    with open(os.path.join(simdir, 'cells.tsv')) as f:
        for l in f:
            if l[0] != '#':
                bc, p = l.split()
                pop[bc] = p
    return truth, pop


def read_calls(f, column):
    calls = defaultdict(list)
    with open(f) as i:
        head = None
        for l in i:
            p = l.rstrip('\n').split('\t')
            if l[0] == '#':
                head = [h.lstrip('#') for h in p]
                continue
            r = dict(zip(head, p))
            col = column if column in r else 'CN_STATE' if 'CN_STATE' in r else 'HAP_CN'
            a, b = map(int, r[col].split('|'))
            calls[r['CELL']].append((r['CHR'], int(r['START']), int(r['END']), a, b))
    return calls


def overlaps(call, segs):
    c, s, e = call[:3]
    for tc, ts, te, ta, tb in segs:
        if tc == c and ts < e and s < te:
            yield min(e, te) - max(s, ts), (ta, tb)


def score(calls, truth, pop, region=None, cells=None):
    tot = acc_t = acc_a = 0
    for cell, bins in calls.items():
        if cell not in pop or (cells is not None and pop[cell] not in cells):
            continue
        segs = truth[pop[cell]]
        if region is not None:
            rc, rs, re_ = region
            segs = [(c, max(s, rs), min(e, re_), a, b) for c, s, e, a, b in segs if c == rc and s < re_ and rs < e]
        for c, s, e, a, b in bins:
            for w, (ta, tb) in overlaps((c, s, e), segs):
                tot += w
                acc_t += w * (a + b == ta + tb)
                acc_a += w * (sorted((a, b)) == sorted((ta, tb)))
    return (acc_t / tot, acc_a / tot) if tot else (float('nan'), float('nan'))


def ari(x, y):
    n = len(x)
    pairs = lambda cnt: sum(comb(v, 2) for v in cnt.values())
    sxy = pairs(Counter(zip(x, y)))
    sx, sy = pairs(Counter(x)), pairs(Counter(y))
    exp = sx * sy / comb(n, 2)
    mx = (sx + sy) / 2.0
    return (sxy - exp) / (mx - exp) if mx != exp else 1.0


def main():
    simdir = sys.argv[1]
    truth, pop = read_truth(simdir)
    rows = []
    for arg in sys.argv[2:]:
        name, fcalls, *rest = arg.split(':')
        nbins = len(set((c, s, e) for c, s, e, a, b in next(iter(read_calls(fcalls, 'CN_STATE').values()))))
        for column in ['CN_STATE', 'CORRECTED_HAP_CN']:
            calls = read_calls(fcalls, column)
            r = [name, nbins, 'raw' if column == 'CN_STATE' else 'clone-corrected', len(calls)]
            r += list(score(calls, truth, pop))
            for p, c, s, e, label in FOCAL:
                r.append(score(calls, truth, pop, (c, s, e), {p})[1])
            rows.append(r)
        if rest and os.path.isfile(rest[0]):
            m = {}
            with open(rest[0]) as f:
                for l in f:
                    if l[0] != '#':
                        cell, clu, clone = l.split()
                        m[cell] = clone
            cells = [c for c in m if c in pop]
            noisy = sum(m[c] == 'None' for c in cells)
            rows[-1].append('ARI={:.3f} ({} clones, {} unassigned)'.format(
                ari([pop[c] for c in cells], [m[c] for c in cells]), len(set(m.values()) - {'None'}), noisy))
    head = ['run', 'bins', 'calls', 'cells', 'total', 'allele'] + [f[4] for f in FOCAL] + ['clones']
    print('\t'.join(head))
    for r in rows:
        print('\t'.join('{:.3f}'.format(v) if isinstance(v, float) else str(v) for v in r))


if __name__ == '__main__':
    main()
