#!/usr/bin/env python2.7
"""Per-cell RDR and BAF of ONE bin, to see how many reads per cell it takes to see a signal.

Reads calls/calls.tsv (12 columns, with the per-cell CLUSTER and CN_STATE) or combo/combo.tsv (10 columns)
and, for the chosen bin, writes

  <prefix>.tsv   one row per cell: RDR, read counts, A/B counts, BAF, mirrored BAF (+ CLUSTER, CN_STATE)
  <prefix>.png   (1) RDR vs BAF, one dot per cell; (2) BAF vs SNP-covering reads A+B with the binomial
                 funnel pooled_BAF +- 2 sqrt(p(1-p)/n); (3) RDR vs read count with the Poisson funnel
                 pooled_RDR +- 2 pooled_RDR/sqrt(count)
  stdout         pooled values and how many cells carry any SNP read / are far from the pooled value

Cells with no SNP-covering read in the bin have a BAF of 0.5 only as a placeholder: they are not drawn in
the BAF panels and are counted separately.

Examples:
  chisel_cellbin calls/calls.tsv --at chr13:50000000
  chisel_cellbin calls/calls.tsv --bin chr13:48000000-49500000 --mirror
  chisel_cellbin calls/calls.tsv --list chr13        # bins of a chromosome, with their cell coverage
"""
from __future__ import print_function

import argparse
import math
import os
import sys

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("INPUT", nargs='?', default='calls/calls.tsv', type=str, help="calls/calls.tsv or combo/combo.tsv (default: calls/calls.tsv)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--at", type=str, default=None, help="The bin containing this position, CHR:POS")
    g.add_argument("--bin", type=str, default=None, help="The bin overlapping this region, CHR:START-END (must overlap exactly one bin)")
    g.add_argument("--index", type=int, default=None, help="The i-th bin (0-based) of --list's chromosome ordering, with --chr")
    g.add_argument("--list", type=str, default=None, metavar="CHR", help="List the bins of this chromosome (name, size, mean SNP reads per cell) and exit")
    p.add_argument("--chr", type=str, default=None, help="Chromosome for --index")
    p.add_argument("-o", "--out", type=str, default=None, help="Output prefix (default: cellbin_<chr>_<start>_<end> in the current directory)")
    p.add_argument("--mirror", action='store_true', default=False, help="Plot min(BAF, 1-BAF) instead of BAF (default: BAF as in the input)")
    p.add_argument("--minsnpreads", type=int, default=1, help="Draw cells with at least this many SNP-covering reads in the BAF panels (default: 1)")
    p.add_argument("--color", type=str, default='auto', choices=['auto', 'cn', 'cluster', 'none'], help="Colour cells by CN_STATE, by bin CLUSTER, or not (default: auto = CN_STATE if the input has it)")
    p.add_argument("--maxcolors", type=int, default=8, help="Colour only the most common categories, the rest gray (default: 8)")
    p.add_argument("-f", "--figformat", type=str, default='png', choices=['png', 'pdf'], help="Figure format (default: png)")
    p.add_argument("--nofig", action='store_true', default=False, help="Only write the table")
    return p.parse_args(argv)


def parse_region(s):
    c, _, r = s.partition(':')
    if not c or not r:
        raise ValueError("Expected CHR:START-END or CHR:POS, got '{}'".format(s))
    r = r.replace(',', '')
    a, _, b = r.partition('-')
    a = int(float(a))
    return c, a, (int(float(b)) if b else a)


def read_rows(path, keep):
    """Yield the split rows whose (chr, start, end) satisfies keep(chr, start, end)."""
    with open(path, 'r') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 10:
                p = line.split()
            if len(p) < 10:
                continue
            if keep(p[0], int(p[1]), int(p[2])):
                yield p


def list_bins(path, chrom):
    bins = {}
    for p in read_rows(path, lambda c, s, e: c == chrom):
        b = (int(p[1]), int(p[2]))
        t = bins.setdefault(b, [0, 0, 0])
        t[0] += 1
        t[1] += int(p[7]) + int(p[8])
        t[2] += int(int(p[7]) + int(p[8]) > 0)
    print("index\tchr\tstart\tend\tkb\tcells\tmean_snp_reads_per_cell\tcells_with_snp_read")
    for i, (b, t) in enumerate(sorted(bins.items())):
        print("{}\t{}\t{}\t{}\t{:.0f}\t{}\t{:.2f}\t{}".format(i, chrom, b[0], b[1], (b[1] - b[0]) / 1000.0, t[0], t[1] / float(t[0]), t[2]))


def select_bin(path, args):
    if args.at:
        c, pos, _ = parse_region(args.at)
        keep = (lambda cc, s, e: cc == c and s <= pos < e)
    elif args.bin:
        c, a, b = parse_region(args.bin)
        keep = (lambda cc, s, e: cc == c and s < b and e > a)
    else:
        if args.index is None or not args.chr:
            raise ValueError("Give one of --at, --bin, or --index with --chr (or --list CHR)")
        starts = sorted(set((int(p[1]), int(p[2])) for p in read_rows(path, lambda cc, s, e: cc == args.chr)))
        if not 0 <= args.index < len(starts):
            raise ValueError("--index must be in [0, {})".format(len(starts)))
        bs, be = starts[args.index]
        keep = (lambda cc, s, e: cc == args.chr and s == bs and e == be)
    rows = list(read_rows(path, keep))
    bins = sorted(set((p[0], int(p[1]), int(p[2])) for p in rows), key=(lambda x: (x[1], x[2])))
    if not bins:
        raise ValueError("No bin matches the request")
    if len(bins) > 1:
        raise ValueError("The region overlaps {} bins ({}); choose one with --at, or --list".format(
            len(bins), ', '.join('{}:{}-{}'.format(*b) for b in bins[:6]) + (' ...' if len(bins) > 6 else '')))
    return bins[0], rows


def table(rows):
    cols = ['CELL', 'NORM_COUNT', 'COUNT', 'RDR', 'A_COUNT', 'B_COUNT', 'BAF', 'MIRRORED_BAF', 'SNP_READS']
    extra = []
    if len(rows[0]) >= 12:
        extra = ['CLUSTER', 'CN_STATE']
    if len(rows[0]) >= 13:
        extra.append('CORRECTED_CN_STATE')
    out = []
    for p in rows:
        a, b = int(p[7]), int(p[8])
        baf = float(p[9])
        out.append([p[3], int(p[4]), int(p[5]), float(p[6]), a, b, baf, min(baf, 1.0 - baf), a + b] + p[10:10 + len(extra)])
    return cols + extra, out


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.INPUT):
        raise ValueError("Input file does not exist: {}".format(args.INPUT))
    if args.list:
        list_bins(args.INPUT, args.list)
        return
    (chrom, start, end), rows = select_bin(args.INPUT, args)
    cols, data = table(rows)
    prefix = args.out or "cellbin_{}_{}_{}".format(chrom, start, end)
    with open(prefix + '.tsv', 'w') as f:
        f.write('#CHR\tSTART\tEND\t' + '\t'.join(cols) + '\n')
        for r in data:
            f.write('\t'.join([chrom, str(start), str(end)] + [str(x) for x in r]) + '\n')

    ix = dict((c, i) for i, c in enumerate(cols))
    n = len(data)
    A = np.array([r[ix['A_COUNT']] for r in data], dtype=float)
    B = np.array([r[ix['B_COUNT']] for r in data], dtype=float)
    N = np.array([r[ix['NORM_COUNT']] for r in data], dtype=float)
    C = np.array([r[ix['COUNT']] for r in data], dtype=float)
    R = np.array([r[ix['RDR']] for r in data], dtype=float)
    baf = np.array([r[ix['MIRRORED_BAF' if args.mirror else 'BAF']] for r in data], dtype=float)
    S = A + B
    has = S >= max(1, args.minsnpreads)

    pooled_baf = B.sum() / S.sum() if S.sum() > 0 else float('nan')
    if args.mirror:
        pooled_baf = min(pooled_baf, 1.0 - pooled_baf)
    # RDR is count-based: the pooled RDR is the read-count-weighted mean of the cells' RDRs, as in the HMM
    pooled_rdr = float((R * C).sum() / C.sum()) if C.sum() > 0 else float('nan')

    print("bin {}:{}-{} ({:.0f} kb), {} cells".format(chrom, start, end, (end - start) / 1000.0, n))
    print("  SNP-covering reads per cell: mean {:.2f}, median {:.0f}; cells with >=1: {} ({:.0f}%), with >={}: {}".format(
        S.mean(), np.median(S), int((S > 0).sum()), 100.0 * (S > 0).mean(), max(1, args.minsnpreads), int(has.sum())))
    print("  read starts per cell: mean {:.2f}, median {:.0f}; cells with 0: {}".format(C.mean(), np.median(C), int((C == 0).sum())))
    print("  pooled {}BAF {:.4f} (from {:.0f} SNP reads), pooled RDR {:.4f}".format('mirrored ' if args.mirror else '', pooled_baf, S.sum(), pooled_rdr))
    if has.any() and pooled_baf == pooled_baf:
        sd = np.sqrt(max(pooled_baf * (1 - pooled_baf), 1e-9) / S[has])
        z = np.abs(baf[has] - pooled_baf) / sd
        print("  cells with BAF within 2 binomial SD of the pooled BAF: {:.0f}% (expected ~95% if cells share one state)".format(100.0 * (z <= 2).mean()))
        for k in (1, 5, 10, 20, 50, 100):
            print("    SD of one cell's BAF with {:>3} SNP reads at the pooled BAF: {:.3f}".format(k, math.sqrt(max(pooled_baf * (1 - pooled_baf), 0) / k)))

    if args.nofig:
        print("wrote {}.tsv".format(prefix))
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # colours
    mode = args.color
    if mode == 'auto':
        mode = 'cn' if 'CN_STATE' in ix else 'none'
    if mode == 'cn' and 'CN_STATE' not in ix or mode == 'cluster' and 'CLUSTER' not in ix:
        mode = 'none'
    labels = np.array([str(r[ix['CN_STATE' if mode == 'cn' else 'CLUSTER']]) for r in data]) if mode != 'none' else np.array(['all'] * n)
    uniq, cnt = np.unique(labels, return_counts=True)
    top = [u for u, _ in sorted(zip(uniq, cnt), key=(lambda x: -x[1]))[:args.maxcolors]]
    palette = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#EE8866', '#44BB99']
    col = dict((u, palette[i % len(palette)]) for i, u in enumerate(top))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    blabel = 'min(BAF, 1-BAF)' if args.mirror else 'BAF'

    def draw(a, x, y, sel):
        for u in top + ['__other']:
            m = sel & ((labels == u) if u != '__other' else ~np.isin(labels, top))
            if m.any():
                a.scatter(x[m], y[m], s=14, alpha=0.6, linewidths=0, color=col.get(u, '#999999'),
                          label=(u if mode != 'none' else None) if u != '__other' else 'other')

    draw(ax[0], R, baf, has)
    ax[0].axvline(pooled_rdr, color='k', lw=0.8, ls='--')
    ax[0].axhline(pooled_baf, color='k', lw=0.8, ls='--')
    ax[0].set_xlabel('RDR (per cell)')
    ax[0].set_ylabel(blabel)
    ax[0].set_title('{} cells with >= {} SNP read(s); {} without'.format(int(has.sum()), max(1, args.minsnpreads), int((~has).sum())), fontsize=10)
    if mode != 'none':
        ax[0].legend(title='CN_STATE' if mode == 'cn' else 'CLUSTER', fontsize=7, loc='best')

    draw(ax[1], S, baf, has)
    if has.any():
        xs = np.arange(1, max(2, int(S.max())) + 1)
        for k in (1, 2):
            half = k * np.sqrt(max(pooled_baf * (1 - pooled_baf), 0) / xs)
            ax[1].plot(xs, np.clip(pooled_baf + half, 0, 1), color='k', lw=0.8, ls=':' if k == 2 else '-')
            ax[1].plot(xs, np.clip(pooled_baf - half, 0, 1), color='k', lw=0.8, ls=':' if k == 2 else '-')
    ax[1].axhline(pooled_baf, color='k', lw=0.8, ls='--')
    ax[1].set_xlabel('SNP-covering reads in the bin (A + B)')
    ax[1].set_ylabel(blabel)
    ax[1].set_title('binomial funnel: pooled +- 1, 2 SD', fontsize=10)

    allc = np.ones(n, dtype=bool)
    draw(ax[2], C, R, allc)
    xs = np.arange(1, max(2, int(C.max())) + 1)
    for k in (1, 2):
        half = k * pooled_rdr / np.sqrt(xs)
        ax[2].plot(xs, pooled_rdr + half, color='k', lw=0.8, ls=':' if k == 2 else '-')
        ax[2].plot(xs, np.maximum(pooled_rdr - half, 0), color='k', lw=0.8, ls=':' if k == 2 else '-')
    ax[2].axhline(pooled_rdr, color='k', lw=0.8, ls='--')
    ax[2].set_xlabel('read starts in the bin (per cell)')
    ax[2].set_ylabel('RDR')
    ax[2].set_title('Poisson funnel: pooled +- 1, 2 SD', fontsize=10)

    fig.suptitle('{}:{}-{} ({:.0f} kb), {} cells, pooled RDR {:.3f}, pooled {} {:.3f}'.format(
        chrom, start, end, (end - start) / 1000.0, n, pooled_rdr, blabel, pooled_baf), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('{}.{}'.format(prefix, args.figformat), dpi=150)
    print("wrote {0}.tsv and {0}.{1}".format(prefix, args.figformat))


if __name__ == '__main__':
    try:
        main()
    except ValueError as e:
        sys.stderr.write("ERROR: {}\n".format(e))
        sys.exit(1)
