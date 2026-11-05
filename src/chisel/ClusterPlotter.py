#!/usr/bin/env python2.7
"""Bin clusters (HMM or k-medians) of calls/calls.tsv along the genome, on the BAF CHISEL actually used.

calls.tsv has, for every bin and cell, RDR, read counts, A/B counts (in the orientation chosen by Combiner:
blocks aligned per bin on the counts pooled over all cells) and the bin's CLUSTER. Per bin this plots the
POOLED values (RDR = count-weighted mean of the cells' RDRs; BAF = sum B / sum(A + B)) coloured by cluster:

  top      RDR of every bin, a segment from START to END
  middle   BAF of every bin as in combo.tsv (--mirror: min(BAF, 1-BAF))
  strip    the cluster of every bin
  right    RDR vs BAF, one dot per bin (size ~ bin length), cluster means labelled

and writes <prefix>.clusters.tsv (per cluster: bins, Mb, mean pooled RDR/BAF, spread).

  chisel_clusterplot calls/calls.tsv --region 13
  chisel_clusterplot calls/calls.tsv --region 13:19500000-115169878 --mirror
  chisel_clusterplot calls/calls.tsv --region all -o genome
"""
from __future__ import print_function

import argparse
import re
import sys

import numpy as np


def natkey(c):
    m = re.match(r'^(chr)?(\d+)$', c)
    return (0, int(m.group(2)), '') if m else (1, 0, c)


def parse_regions(s):
    if s == 'all':
        return None
    out = []
    for item in s.split(','):
        c, _, r = item.partition(':')
        a, _, b = r.replace(',', '').partition('-')
        out.append((c, int(float(a)) if a else 0, int(float(b)) if b else 10 ** 12))
    return out


def read_bins(path, regions):
    agg = {}
    with open(path) as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 11:
                p = line.split()
            if len(p) < 11:
                continue
            c, s, e = p[0], int(p[1]), int(p[2])
            if regions is not None and not any(c == rc and s < re_ and e > rs for rc, rs, re_ in regions):
                continue
            cnt, rdr, a, b = int(p[5]), float(p[6]), int(p[7]), int(p[8])
            t = agg.setdefault((c, s, e), [0.0, 0.0, 0, 0, p[10]])
            t[0] += rdr * cnt
            t[1] += cnt
            t[2] += a
            t[3] += b
    bins = []
    for (c, s, e), t in agg.items():
        rdr = t[0] / t[1] if t[1] > 0 else float('nan')
        baf = t[3] / float(t[2] + t[3]) if t[2] + t[3] > 0 else float('nan')
        bins.append((c, s, e, rdr, baf, t[2] + t[3], t[4]))
    bins.sort(key=(lambda x: (natkey(x[0]), x[1])))
    return bins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("INPUT", nargs='?', default='calls/calls.tsv')
    ap.add_argument("--region", default='all', help="all, or comma-separated CHR or CHR:START-END items (default: all)")
    ap.add_argument("--mirror", action='store_true', default=False, help="Show min(BAF, 1-BAF) (default: BAF as in combo.tsv)")
    ap.add_argument("-o", "--out", default=None, help="Output prefix (default: clusters_<region>)")
    ap.add_argument("-f", "--figformat", default='png', choices=['png', 'pdf'])
    a = ap.parse_args(argv)

    regions = parse_regions(a.region)
    bins = read_bins(a.INPUT, regions)
    if not bins:
        raise ValueError("No bins in the requested region")
    prefix = a.out or 'clusters_' + re.sub(r'[^A-Za-z0-9]+', '_', a.region)

    # x: cumulative position, with a gap between chromosomes / regions
    order = []
    for b in bins:
        if b[0] not in order:
            order.append(b[0])
    lo = dict((c, min(b[1] for b in bins if b[0] == c)) for c in order)
    hi = dict((c, max(b[2] for b in bins if b[0] == c)) for c in order)
    off, x0 = {}, float(lo[order[0]])
    for c in order:
        off[c] = x0 - lo[c]
        x0 += (hi[c] - lo[c]) + 0.01 * sum(hi[k] - lo[k] for k in order)
    X = lambda c, v: (v + off[c]) / 1e6

    clusters = sorted(set(b[6] for b in bins), key=(lambda k: int(k) if k.lstrip('-').isdigit() else k))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20' if len(clusters) > 10 else 'tab10')
    col = dict((k, cmap(i % cmap.N)) for i, k in enumerate(clusters))

    fy = (lambda v: min(v, 1.0 - v) if a.mirror else v)
    from matplotlib import gridspec
    fig = plt.figure(figsize=(17, 6.2))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3.2, 1], height_ratios=[3, 3, 0.35], hspace=0.12, wspace=0.12)
    axr, axb, axs = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
    axv = fig.add_subplot(gs[:, 1])
    for b in bins:
        c, s, e, rdr, baf, n, k = b
        xs, xe = X(c, s), X(c, e)
        if rdr == rdr:
            axr.plot([xs, xe], [rdr, rdr], color=col[k], lw=3, solid_capstyle='butt')
        if baf == baf:
            axb.plot([xs, xe], [fy(baf)] * 2, color=col[k], lw=3, solid_capstyle='butt')
        axs.add_patch(plt.Rectangle((xs, 0), xe - xs, 1, color=col[k], lw=0))
    xmax = max(X(c, hi[c]) for c in order)
    xmin = min(X(c, lo[c]) for c in order)
    for ax in (axr, axb, axs):
        ax.set_xlim(xmin, xmax)
    axr.set_ylabel('pooled RDR')
    axb.set_ylabel('pooled ' + ('min(BAF, 1-BAF)' if a.mirror else 'BAF (as combined)'))
    axb.set_ylim(-0.02, 0.52 if a.mirror else 1.02)
    axb.axhline(0.5, color='k', lw=0.5, ls=':')
    axr.axhline(1.0, color='k', lw=0.5, ls=':')
    axr.set_xticks([])
    axb.set_xticks([])
    axs.set_ylim(0, 1)
    axs.set_yticks([])
    axs.set_xlabel('position (Mb)' if len(order) == 1 else 'chromosomes: ' + ' '.join(order))
    if len(order) > 1:
        axs.set_xticks([(X(c, lo[c]) + X(c, hi[c])) / 2.0 for c in order])
        axs.set_xticklabels([c.replace('chr', '') for c in order], fontsize=7)
        for c in order[1:]:
            for ax in (axr, axb):
                ax.axvline(X(c, lo[c]) - 0.005 * xmax, color='gray', lw=0.4)

    rows = []
    for k in clusters:
        sel = [b for b in bins if b[6] == k and b[3] == b[3] and b[4] == b[4]]
        if not sel:
            continue
        w = np.array([b[2] - b[1] for b in sel], dtype=float)
        r = np.array([b[3] for b in sel])
        f = np.array([fy(b[4]) for b in sel])
        mr, mf = np.average(r, weights=w), np.average(f, weights=w)
        axv.scatter(f, r, s=np.clip(w / 1e5, 8, 150), color=col[k], edgecolor='k', linewidths=0.3, alpha=0.8)
        axv.annotate(str(k), (float(np.median(f)), mr), fontsize=9, weight='bold', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col[k], alpha=0.9))
        rows.append([k, len(sel), w.sum() / 1e6, mr, mf, r.std(), f.std()])
    axv.set_xlabel('pooled ' + ('min(BAF, 1-BAF)' if a.mirror else 'BAF'))
    axv.set_ylabel('pooled RDR')
    fig.suptitle('{}  bins by cluster ({} bins, {} clusters)'.format(a.region, len(bins), len(clusters)), fontsize=11)
    fig.savefig('{}.{}'.format(prefix, a.figformat), dpi=150, bbox_inches='tight')
    with open(prefix + '.clusters.tsv', 'w') as o:
        o.write('CLUSTER\tBINS\tMB\tMEAN_RDR\tMEAN_BAF\tSD_RDR\tSD_BAF\n')
        for r in rows:
            o.write('\t'.join([str(r[0])] + ['{:.4g}'.format(v) for v in r[1:]]) + '\n')
    print("cluster  bins     Mb  mean RDR  mean {}  (SD within cluster: RDR, BAF)".format('mBAF' if a.mirror else 'BAF'))
    for r in rows:
        print("{:>7} {:>5} {:>6.1f} {:>9.3f} {:>9.3f}   ({:.3f}, {:.3f})".format(r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    print("wrote {0}.{1} and {0}.clusters.tsv".format(prefix, a.figformat))


if __name__ == '__main__':
    try:
        main()
    except ValueError as e:
        sys.stderr.write("ERROR: {}\n".format(e))
        sys.exit(1)
