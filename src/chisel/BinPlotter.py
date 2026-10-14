#!/usr/bin/env python2.7
"""
Diagnostic plots of variable-length bins (output of Binner.py).

Page 1: distribution of bin sizes and of SNP-covering reads per cell, by deciding criterion.
Then one page per chromosome with four aligned tracks:
  1. heterozygous SNP density (SNPs per 100 kb);
  2. how bins fill: SNP-covering reads per cell accumulated window by window inside each
     bin, which resets at every boundary; the dashed line is the threshold, so a bin
     closes where its curve first crosses it (unless another criterion is limiting);
  3. size of every bin, coloured by the criterion that decided its boundary, hatched when
     it absorbed a trailing run or never met the thresholds;
  4. pseudobulk RDR of every bin (tumor/normal read starts, normalised by their totals).

Usage: BinPlotter.py BINS.tsv WINDOWS.tsv[.gz] -o bins.pdf
"""

import os, sys
import argparse
import gzip
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

COLORS = {'snp_reads' : '#1f77b4', 'snps' : '#9467bd', 'tumor_reads' : '#2ca02c', 'normal_reads' : '#ff7f0e'}
OTHER = '#7f7f7f'


def order(c):
    d = ''.join(x for x in c if x.isdigit())
    return (int(d) if d else (23 if 'X' in c else (24 if 'Y' in c else 25)), c)


def read_table(f):
    meta, rows, head = {}, [], None
    op = gzip.open if f.endswith('.gz') else open
    with op(f, 'rt' if sys.version_info[0] >= 3 else 'r') as i:
        for l in i:
            l = l.rstrip('\n')
            if l.startswith('##'):
                k, v = l[2:].split('=', 1)
                try:
                    meta[k] = float(v)
                except ValueError:
                    meta[k] = v
            elif l.startswith('#'):
                head = l[1:].split('\t')
            elif l:
                rows.append(dict(zip(head, l.split('\t'))))
    return meta, rows


def color(limit):
    return COLORS.get(limit.split('+')[0], OTHER)


def main(args=None):
    parser = argparse.ArgumentParser(description="Diagnostic plots of variable-length bins computed by Binner.py")
    parser.add_argument("BINS", type=str, help="Bins file written by Binner.py")
    parser.add_argument("WINDOWS", type=str, help="Per-window trace written by Binner.py --windows")
    parser.add_argument("-o", "--output", type=str, default='bins.pdf', help="Output PDF (default: bins.pdf)")
    parser.add_argument("-c", "--chromosomes", type=str, default=None, help="Space-separated chromosomes to plot (default: all)")
    parser.add_argument("--density", type=str, default='100kb', help="Window for SNP density, kb or Mb suffix (default: 100kb)")
    args = parser.parse_args(args)

    meta, bins = read_table(args.BINS)
    _, wins = read_table(args.WINDOWS)
    ncells = meta['NCELLS']
    thr = meta['MIN_SNP_READS'] / ncells if 'MIN_SNP_READS' in meta else None  # None for given (e.g. UGP) bins
    dens = int(args.density[:-2]) * (1000 if args.density.endswith('kb') else 1000000) if args.density[-2:] in ('kb', 'Mb') else int(args.density)
    for b in bins:
        for k in ('START', 'END', 'SIZE', 'TUMOR_READS', 'NORMAL_READS'):
            b[k] = int(b[k])
        b['SNP_READS_PER_CELL'] = float(b['SNP_READS_PER_CELL'])
        v = b.get('SNP_READS_MIN_CELL', 'nan')
        b['SNP_READS_MIN_CELL'] = float('nan') if v in ('NA', '') else float(v)
    chrs = sorted(set(b['CHR'] for b in bins), key=order)
    if args.chromosomes:
        chrs = [c for c in chrs if c in args.chromosomes.split()]
    W = defaultdict(list)
    for w in wins:
        W[w['CHR']].append(w)

    totT = float(sum(b['TUMOR_READS'] for b in bins))
    totN = float(sum(b['NORMAL_READS'] for b in bins))
    limits = sorted(set(b['LIMIT'] for b in bins))
    legend = [Patch(color=color(l), label=l) for l in limits]

    with PdfPages(args.output) as pdf:
        # Summary page
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        sizes = np.array([b['SIZE'] for b in bins], dtype=float)
        edges = np.logspace(np.log10(max(1.0, sizes.min())), np.log10(sizes.max() + 1), 40)
        ax[0].hist([[b['SIZE'] for b in bins if b['LIMIT'] == l] for l in limits], bins=edges, stacked=True,
                   color=[color(l) for l in limits], label=limits)
        ax[0].set_xscale('log')
        ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p : '{:g} kb'.format(v / 1000.0)))
        ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[0].set_xlabel('Bin size')
        ax[0].set_ylabel('Number of bins')
        ax[0].legend(title='Deciding criterion', fontsize=8)
        ax[1].hist([[b['SNP_READS_PER_CELL'] for b in bins if b['LIMIT'] == l] for l in limits], bins=40, stacked=True,
                   color=[color(l) for l in limits])
        if thr is not None:
            ax[1].axvline(thr, color='k', ls='--', lw=1)
        ax[1].set_xlabel('SNP-covering reads per cell per bin (mean over cells)')
        if thr is not None:
            title = ('{} bins, {} cells, median size {:.0f} kb; thresholds (pseudobulk): SNP reads >= {:.0f} ({:.1f}/cell), '
                     'SNPs >= {:.0f}, tumor reads >= {:.0f}, normal reads >= {:.0f}').format(
                         len(bins), int(ncells), np.median(sizes) / 1000.0, meta['MIN_SNP_READS'], thr, meta['MIN_SNPS'],
                         meta['MIN_TUMOR_READS'], meta['MIN_NORMAL_READS'])
        else:
            title = '{} given bins ({}), {} cells, median size {:.0f} kb'.format(
                         len(bins), os.path.basename(str(meta.get('SOURCE', ''))), int(ncells), np.median(sizes) / 1000.0)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        for c in chrs:
            ws = W[c]
            cb = [b for b in bins if b['CHR'] == c]
            fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios' : [1, 2, 1.5, 1.5]})
            mb = 1e6
            start = np.array([int(w['START']) for w in ws], dtype=float)
            snps = np.array([float(w['NSNPS']) for w in ws])
            sread = np.array([float(w['SNP_READS']) for w in ws]) / ncells
            binid = [w['BIN'] for w in ws]

            # 1. SNP density
            k = (start // dens).astype(int)
            agg = np.bincount(k, weights=snps)
            pos = np.nonzero(np.bincount(k) > 0)[0]
            ax[0].bar(pos * dens / mb, agg[pos], width=dens / mb, align='edge', color='#555555', lw=0)
            ax[0].set_ylabel('SNPs per\n{}'.format(args.density))

            # 2. Filling of bins
            acc = np.zeros(len(ws))
            run = 0.0
            for x in range(len(ws)):
                if x > 0 and binid[x] != binid[x - 1]:
                    run = 0.0
                run += sread[x]
                acc[x] = run
            lim = {b['BIN'] if 'BIN' in b else '{}:{}-{}'.format(c, b['START'], b['END']) : b['LIMIT'] for b in cb}
            starts_by_bin = defaultdict(list)
            for x, bid in enumerate(binid):
                starts_by_bin[bid].append(x)
            for bid, xs in starts_by_bin.items():
                xs = np.array(xs)
                col = color(lim.get(bid, 'dropped')) if bid != 'dropped' else '#d62728'
                ax[1].plot(np.append(start[xs], int(ws[xs[-1]]['END'])) / mb, np.append(acc[xs], acc[xs[-1]]), color=col, lw=0.8, drawstyle='steps-post')
            if thr is not None:
                ax[1].axhline(thr, color='k', ls='--', lw=1, label='threshold (mean per cell)')
            for b in cb:
                ax[1].axvline(b['START'] / mb, color='#cccccc', lw=0.3, zorder=0)
                if not np.isnan(b['SNP_READS_MIN_CELL']):
                    ax[1].hlines(b['SNP_READS_MIN_CELL'], b['START'] / mb, b['END'] / mb, color='#d62728', lw=1.5)
            ax[1].plot([], [], color='#d62728', lw=1.5, label='least-covered cell in bin')
            ax[1].legend(fontsize=7, loc='upper right')
            ax[1].set_ylabel('SNP reads per cell\naccumulated in bin')
            top = thr * 2.5 if thr is not None else np.percentile([b['SNP_READS_PER_CELL'] for b in cb], 95) * 1.5
            ax[1].set_ylim(0, max(top, 1))

            # 3. Bin sizes
            for b in cb:
                ax[2].hlines(b['SIZE'] / 1000.0, b['START'] / mb, b['END'] / mb, color=color(b['LIMIT']), lw=3)
                if b['CLOSE'] != 'met':
                    ax[2].hlines(b['SIZE'] / 1000.0, b['START'] / mb, b['END'] / mb, color='k', lw=1, linestyles=':' if b['CLOSE'] == 'met+tail' else '-')
            ax[2].set_yscale('log')
            lo, hi = min(b['SIZE'] for b in cb) / 1000.0, max(b['SIZE'] for b in cb) / 1000.0
            ax[2].set_ylim(lo / 1.5, hi * 6)  # room for the legend above the largest bin
            ax[2].set_ylabel('Bin size (kb)')
            ax[2].legend(handles=legend + [Patch(facecolor='white', edgecolor='k', ls=':', label='met+tail'),
                                           Patch(facecolor='white', edgecolor='k', label='unmet')], fontsize=7, ncol=6, loc='upper right')

            # 4. Pseudobulk RDR
            for b in cb:
                if b['NORMAL_READS'] > 0:
                    r = (b['TUMOR_READS'] / totT) / (b['NORMAL_READS'] / totN)
                    ax[3].hlines(r, b['START'] / mb, b['END'] / mb, color='k', lw=2)
            ax[3].axhline(1.0, color='#999999', ls=':', lw=1)
            ax[3].set_ylabel('Pseudobulk RDR')
            ax[3].set_xlabel('{} (Mb)'.format(c))
            fig.suptitle('{}: {} bins'.format(c, len(cb)))
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
    sys.stderr.write('Plots written to {}\n'.format(args.output))


if __name__ == '__main__':
    main()
