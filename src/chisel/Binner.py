#!/usr/bin/env python2.7
"""
Variable-length (adaptive) genomic binning for CHISEL.

Port of the window-merge binning of the Universal-Genotyping-Pipeline
(raphael-group/Universal-Genotyping-Pipeline, workflow/scripts/script_utils/
segmentation_utils.py: build_adaptive_bins / _merge_bins_to_bbs), which is the
successor of HATCHet2's adaptive_bins_arm (Myers et al. 2023).

The genome is tiled into small fixed windows (default 1kb) inside independent
regions (chromosome arms by default). Consecutive windows are merged greedily
into a bin until the bin holds enough
    (1) SNP-covering reads in the tumor pseudobulk,
    (2) heterozygous SNPs,
    (3) read starts in the tumor pseudobulk and in the matched normal.
As in UGP's single-cell mode, every criterion is applied to the PSEUDOBULK of
the selected cells, never to individual cells. The per-cell options are a
convenience: a per-cell threshold t is converted to a pseudobulk threshold of
t * (number of selected cells), i.e. a bound on the mean count per cell.

Bins never span two regions. A trailing run that misses the criteria is merged
into the previous bin; a region that never meets them becomes a single bin.

Output: a TSV in BED coordinates (0-based, half-open) with one row per bin,
which can be passed to `RDREstimator.py --bins` to count reads per cell.
"""

import os, sys
import argparse
import shlex
import tempfile
import subprocess as sp
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict, Counter

import numpy as np

try:
    from Utils import *
except ImportError:  # imported from outside the package directory (e.g. python3 tests)
    from chisel.Utils import *


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Length of chr1 identifies the reference version when it is not given
CHR1_LENGTHS = {249250621 : 'hg19', 248956422 : 'hg38'}


def parse_size(s):
    s = str(s)
    try:
        if s[-2:] == 'kb':
            return int(s[:-2]) * 1000
        elif s[-2:] == 'Mb':
            return int(s[:-2]) * 1000000
        else:
            return int(s)
    except ValueError:
        raise ValueError('Size must be a number, optionally ending with either "kb" or "Mb"!')


def parse_args(args):
    description = "Compute variable-length bins from the pseudobulk of single cells (UGP window-merge binning)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-t","--tumor", required=True, type=str, help="Barcoded single-cell BAM file")
    parser.add_argument("-n","--normal", required=True, type=str, help="BAM file for matched normal sample")
    parser.add_argument("-r","--reference", type=str, required=True, help="Reference genome (its .dict must exist)")
    parser.add_argument("-l","--listcells", type=str, required=False, default=None, help="File whose first column lists the selected cells, e.g. total.tsv from RDREstimator (a row named 'normal' is ignored)")
    parser.add_argument("-b","--baf", type=str, required=False, default=None, help="Per-cell SNP counts produced by BAFEstimator (baf.tsv); needs -l")
    parser.add_argument("--snps", type=str, required=False, default=None, help="Pooled mode, instead of -l/-b: TSV of SNP-covering reads of all cells per het SNP, columns CHR POS DEPTH (1-based POS); the tumor BAM is then counted without selecting cells and per-cell thresholds use --ncells")
    parser.add_argument("--ncells", type=int, required=False, default=0, help="Number of cells for per-cell thresholds in pooled mode (default: 0, per-cell thresholds disabled)")
    parser.add_argument("--counts", type=str, required=False, default=None, help="Cache of the window counts (.npz): loaded if it exists, written otherwise, so several thresholds do not recount the BAMs (default: none)")
    parser.add_argument("-c", "--chromosomes", type=str, required=False, default=' '.join(['chr{}'.format(i) for i in range(1, 23)]), help="Space-separated list of chromosomes between apices (default: chr1 ... chr22)")
    parser.add_argument("-w","--windowsize", type=str, required=False, default="1kb", help="Size of the fixed windows that are merged into bins, with or without \"kb\" or \"Mb\" (default: 1kb)")
    parser.add_argument("--minsnpreadspercell", type=float, required=False, default=25.0, help="Minimum mean number of SNP-covering reads per cell in every bin; multiplied by the number of cells to obtain a pseudobulk threshold (default: 25)")
    parser.add_argument("--minsnpreads", type=float, required=False, default=0.0, help="Minimum number of SNP-covering reads in the tumor pseudobulk of every bin; the larger of this and the per-cell threshold is used (default: 0)")
    parser.add_argument("--mintotalreadspercell", type=float, required=False, default=0.0, help="Minimum mean number of read starts per cell in every bin; multiplied by the number of cells (default: 0)")
    parser.add_argument("--mintotalreads", type=float, required=False, default=5000.0, help="Minimum number of read starts in every bin in both the tumor pseudobulk and the matched normal; for the tumor the larger of this and the per-cell threshold is used (default: 5000, as in UGP)")
    parser.add_argument("--minsnps", type=int, required=False, default=1, help="Minimum number of heterozygous SNPs in every bin (default: 1)")
    parser.add_argument("--regions", type=str, required=False, default=None, help="BED file of regions that bins cannot span, e.g. chromosome arms (default: chromosome arms from the built-in centromere table of --refversion)")
    parser.add_argument("--refversion", type=str, required=False, default='auto', choices=['auto', 'hg19', 'hg38', 'none'], help="Reference version used to split chromosomes into arms when --regions is not given; 'auto' detects it from the length of chr1, 'none' uses whole chromosomes (default: auto)")
    parser.add_argument("--blacklist", type=str, required=False, default=None, help="BED file of regions to exclude; regions are split at excluded intervals (default: none)")
    parser.add_argument("--keepsnpfree", required=False, default=False, action='store_true', help="Keep bins without any SNP, which only occur when a whole region misses the criteria (default: such bins are dropped, as in UGP)")
    parser.add_argument("--given", type=str, required=False, default=None, help="Do not compute bins but describe these precomputed bins (TSV/BED with columns CHR START END, e.g. UGP's bb.tsv.gz; a region_id column is kept); thresholds and regions are then ignored (default: compute bins)")
    parser.add_argument("--summary", type=str, required=False, default=None, help="Also write the number of bins and their mean and median size to this TSV (default: not written)")
    parser.add_argument("--windows", type=str, required=False, default=None, help="Write the per-window trace (counts and assigned bin of every window) to this file, gzipped if it ends with .gz; it is the input of BinPlotter.py (default: not written)")
    parser.add_argument("--notrim", required=False, default=False, action='store_true', help="Do not trim windows without normal reads (e.g. assembly gaps, telomeres) from the ends of every region (default: trimming enabled)")
    parser.add_argument("-s","--samtools", required=False, default=None, type=str, help="Path to the directory to \"samtools\" executable (default: samtools in $PATH)")
    parser.add_argument("-j","--jobs", required=False, type=int, default=0, help="Number of parallel jobs to use (default: equal to number of available processors)")
    parser.add_argument("--cellprefix", type=str, required=False, default='CB:Z:', help="Prefix of cell barcode field in SAM format (default: CB:Z:)")
    parser.add_argument("--cellsuffix", type=str, required=False, default='', help="Suffix of cell barcode field in SAM format (default: none)")
    parser.add_argument("--outdir", required=False, default='./', type=str, help="Directory for temporary files (default: current directory)")
    args = parser.parse_args(args)

    if args.snps is None and (args.listcells is None or args.baf is None):
        raise ValueError("Give either -l and -b (cells mode) or --snps (pooled mode)!")
    if args.snps is not None and (args.listcells is not None or args.baf is not None):
        raise ValueError("--snps (pooled mode) cannot be combined with -l/-b!")
    for f, name in [(args.tumor, 'Tumor BAM'), (args.normal, 'Normal BAM'), (args.reference, 'Reference genome'),
                    (args.listcells, 'List of cells'), (args.baf, 'BAF file'), (args.snps, 'SNP depth file')]:
        if f is not None and not os.path.isfile(f):
            raise ValueError("{} does not exist: {}".format(name, f))
    if not os.path.isfile(os.path.splitext(args.reference)[0] + '.dict'):
        raise ValueError("The dictionary .dict of the reference genome not found! Please remember to index it.")
    for f, name in [(args.regions, 'Regions BED'), (args.blacklist, 'Blacklist BED')]:
        if f is not None and not os.path.isfile(f):
            raise ValueError("{} does not exist: {}".format(name, f))
    if args.given is not None and not os.path.isfile(args.given):
        raise ValueError("Given bins file does not exist: {}".format(args.given))
    if not os.path.isdir(args.outdir):
        raise ValueError("Output directory does not exist: {}".format(args.outdir))
    windowsize = parse_size(args.windowsize)
    if windowsize < 1:
        raise ValueError("The window size must be positive!")
    for v, name in [(args.minsnpreadspercell, 'minsnpreadspercell'), (args.minsnpreads, 'minsnpreads'),
                    (args.mintotalreadspercell, 'mintotalreadspercell'), (args.mintotalreads, 'mintotalreads'), (args.minsnps, 'minsnps')]:
        if v < 0:
            raise ValueError("{} must be non-negative!".format(name))
    samtools = args.samtools
    if not samtools:
        samtools = "samtools"
    elif os.path.isdir(samtools):
        samtools = os.path.join(samtools, "samtools")
    if which(samtools) is None:
        raise ValueError("samtools has not been found or is not executable!")
    if not args.jobs:
        args.jobs = mp.cpu_count()
    if args.jobs < 1:
        raise ValueError("The number of jobs must be positive!")

    return {
        'tumor' : args.tumor,
        'normal' : args.normal,
        'ref' : args.reference,
        'listcells' : args.listcells,
        'snps' : args.snps,
        'ncells' : args.ncells,
        'counts' : args.counts,
        'baf' : args.baf,
        'chrs' : args.chromosomes.split(),
        'windowsize' : windowsize,
        'minsnpreadspercell' : args.minsnpreadspercell,
        'minsnpreads' : args.minsnpreads,
        'mintotalreadspercell' : args.mintotalreadspercell,
        'mintotalreads' : args.mintotalreads,
        'minsnps' : args.minsnps,
        'regions' : args.regions,
        'refversion' : args.refversion,
        'blacklist' : args.blacklist,
        'keepsnpfree' : args.keepsnpfree,
        'trim' : not args.notrim,
        'windows' : args.windows,
        'summary' : args.summary,
        'given' : args.given,
        'samtools' : samtools,
        'j' : args.jobs,
        'prefix' : args.cellprefix,
        'suffix' : args.cellsuffix,
        'outdir' : args.outdir
    }


def main(args=None, stdout_file=None):
    log('Parsing and checking arguments')
    args = parse_args(args)
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]), level='INFO')

    log('Reading chromosome lengths')
    chrlen = read_chr_lengths(args['ref'], args['chrs'])

    if args['snps'] is None:
        log('Reading selected cells')
        cells = read_cells(args['listcells'])
        log('Number of selected cells: {}'.format(len(cells)), level='INFO')
        if len(cells) == 0:
            raise ValueError('No cell found in {}'.format(args['listcells']))
        ncells = len(cells)
    else:
        cells = None  # pooled mode: every read of the tumor BAM, SNP depths from --snps
        ncells = args['ncells']
        log('Pooled mode, number of cells for per-cell values: {}'.format(ncells if ncells else 'not given'), level='INFO')

    if args['given'] is None:
        log('Defining regions')
        regions = get_regions(chrlen, args)
        log('Number of regions: {}'.format(sum(len(regions[c]) for c in regions)), level='INFO')
    else:
        log('Reading given bins')
        given = read_given(args['given'], chrlen)
        log('Number of given bins: {}'.format(sum(len(given[c]) for c in given)), level='INFO')

    log('Counting read starts in {}bp windows of normal and tumor pseudobulk'.format(args['windowsize']))
    counts = load_counts(args['counts'], chrlen, args['windowsize'])
    if counts is None:
        counts = count_windows(chrlen, cells, args)
        save_counts(args['counts'], counts, args['windowsize'])

    log('Counting SNP-covering reads of the tumor pseudobulk')
    if args['snps'] is None:
        snpreads, snpcount = count_snps(args['baf'], chrlen, cells, args['windowsize'])
    else:
        snpreads, snpcount = count_pooled_snps(args['snps'], chrlen, args['windowsize'])
    log('Number of SNPs: {}'.format(int(sum(snpcount[c].sum() for c in snpcount))), level='INFO')

    if args['given'] is None:
        thres = get_thresholds(args, ncells)
        log('Pseudobulk thresholds per bin: SNP reads >= {snp:.0f}, #SNPs >= {nsnps}, tumor reads >= {tumor:.0f}, normal reads >= {normal:.0f}'.format(**thres), level='INFO')
        log('Merging windows into bins')
        bins, windows = make_bins(regions, counts, snpreads, snpcount, chrlen, thres, args, max(ncells, 1))
        meta = {'NCELLS' : max(ncells, 1), 'WINDOWSIZE' : args['windowsize'], 'MIN_SNP_READS' : thres['snp'], 'MIN_SNPS' : thres['nsnps'],
                'MIN_TUMOR_READS' : thres['tumor'], 'MIN_NORMAL_READS' : thres['normal']}
    else:
        log('Describing given bins')
        bins, windows = describe_bins(given, counts, snpreads, snpcount, chrlen, args, max(ncells, 1))
        meta = {'NCELLS' : max(ncells, 1), 'WINDOWSIZE' : args['windowsize'], 'SOURCE' : os.path.abspath(args['given'])}

    if args['baf'] is not None:
        log('Counting SNP-covering reads of every cell in every bin')
        percell_snp_stats(bins, args['baf'], chrlen, cells)
    else:
        pooled_snp_stats(bins, snpreads, snpcount, max(ncells, 1), args['windowsize'])
    summarize(bins, max(ncells, 1))

    log('Writing bins')
    write_bins(bins, meta, stdout_file)
    if args['summary'] is not None:
        write_summary(bins, args['summary'])
    if args['windows'] is not None:
        log('Writing per-window trace to {}'.format(args['windows']))
        write_windows(windows, meta, args['windows'])
    log('KTHXBYE')


##################################################
# Inputs

def norm(c):
    return c[3:] if c.startswith('chr') else c


def read_chr_lengths(ref, chromosomes):
    """Lengths of the requested chromosomes, keyed by the names used in the reference .dict"""
    want = {norm(c) for c in chromosomes}
    ends = {}
    with open(os.path.splitext(ref)[0] + '.dict', 'r') as i:
        for l in i:
            if l.startswith('@SQ'):
                f = dict(p.split(':', 1) for p in l.strip().split('\t')[1:] if ':' in p)
                if norm(f['SN']) in want:
                    ends[f['SN']] = int(f['LN'])
    missing = want - {norm(c) for c in ends}
    if missing:
        raise ValueError('Chromosomes not found in the reference dictionary with or without chr-notation: {}'.format(','.join(sorted(missing))))
    return ends


def read_cells(f):
    """First column of every line, skipping headers and the normal row of total.tsv"""
    cells = []
    with open(f, 'r') as i:
        for l in i:
            p = l.strip().replace(',', '\t').split()
            if len(p) > 0 and p[0][0] != '#' and p[0] != 'normal':
                cells.append(p[0])
    return sorted(set(cells))


def read_bed(f, chrlen):
    """BED intervals on the analysed chromosomes, keyed by the reference names"""
    names = {norm(c) : c for c in chrlen}
    import gzip
    res = defaultdict(list)
    with (gzip.open(f, 'r') if f.endswith('.gz') else open(f, 'r')) as i:
        for l in i:
            if not isinstance(l, str):
                l = l.decode('utf-8')
            if l.startswith('#') or l.startswith('track') or l.startswith('browser') or len(l.strip()) == 0:
                continue
            p = l.strip().split()
            if norm(p[0]) in names:
                c = names[norm(p[0])]
                s, e = max(0, int(p[1])), min(chrlen[c], int(p[2]))
                if s < e:
                    res[c].append((s, e))
    return {c : sorted(res[c]) for c in res}


def get_regions(chrlen, args):
    """Regions that bins cannot span, snapped to the window grid and minus the blacklist"""
    W = args['windowsize']
    if args['regions'] is not None:
        regions = read_bed(args['regions'], chrlen)
        missing = set(chrlen) - set(regions)
        if missing:
            log('No region given for chromosomes {}, these are skipped'.format(','.join(sorted(missing, key=orderchrs))), level='WARN')
    else:
        version = args['refversion']
        if version == 'auto':
            c1 = [c for c in chrlen if norm(c) == '1']
            version = CHR1_LENGTHS.get(chrlen[c1[0]], 'none') if c1 else 'none'
            log('Detected reference version: {}'.format(version), level='INFO')
        if version == 'none':
            log('Chromosome arms are unknown, each chromosome is a single region', level='WARN')
            regions = {c : [(0, chrlen[c])] for c in chrlen}
        else:
            regions = arms(chrlen, os.path.join(DATA_DIR, '{}.centromeres.txt'.format(version)))
    if args['blacklist'] is not None:
        regions = subtract(regions, read_bed(args['blacklist'], chrlen))
    snap = (lambda v, c : min(chrlen[c], int(round(float(v) / W)) * W))
    snapped = {c : [(snap(s, c), snap(e, c) if e < chrlen[c] else chrlen[c]) for s, e in regions[c]] for c in regions}
    return {c : [r for r in snapped[c] if r[0] < r[1]] for c in snapped if any(r[0] < r[1] for r in snapped[c])}


def arms(chrlen, fcen):
    cen = defaultdict(list)
    with open(fcen, 'r') as i:
        for l in i:
            p = l.strip().split()
            if len(p) >= 3:
                cen[norm(p[0])].append((int(p[1]), int(p[2])))
    res = {}
    for c in chrlen:
        if norm(c) in cen:
            cs, ce = min(s for s, e in cen[norm(c)]), max(e for s, e in cen[norm(c)])
            res[c] = [r for r in [(0, min(cs, chrlen[c])), (min(ce, chrlen[c]), chrlen[c])] if r[0] < r[1]]
        else:
            log('No centromere for {}, it is a single region'.format(c), level='WARN')
            res[c] = [(0, chrlen[c])]
    return res


def subtract(regions, excluded):
    res = {}
    for c in regions:
        out = []
        for s, e in regions[c]:
            cur = s
            for xs, xe in excluded.get(c, []):
                if xe <= cur or xs >= e:
                    continue
                if xs > cur:
                    out.append((cur, xs))
                cur = max(cur, xe)
            if cur < e:
                out.append((cur, e))
        res[c] = out
    return res


##################################################
# Counting

def count_windows(chrlen, cells, args):
    """Read starts per window: {'normal' : {chr : array}, 'tumor' : {chr : array}}, tumor restricted to the selected cells"""
    fd, flist = tempfile.mkstemp(prefix='cells.', suffix='.tmp', dir=args['outdir'])
    with os.fdopen(fd, 'w') as o:
        o.write('\n'.join(cells) + '\n' if cells is not None else '')
    try:
        jobs = [(s, c) for c in sorted(chrlen, key=orderchrs) for s in ('normal', 'tumor')]
        bar = ProgressBar(total=len(jobs), length=40, verbose=False)
        initargs = (args['normal'], args['tumor'], args['samtools'], args['windowsize'], flist if cells is not None else None, args['prefix'], args['suffix'], chrlen)
        pool = Pool(processes=min(args['j'], len(jobs)), initializer=init_counting, initargs=initargs)
        counts = {'normal' : {}, 'tumor' : {}}
        for s, c, arr in pool.imap_unordered(counting, jobs):
            counts[s][c] = arr
            bar.progress(advance=True, msg="Counted {} on {}".format(s, c))
        pool.close()
        pool.join()
    finally:
        os.remove(flist)
    for c in chrlen:
        if counts['tumor'][c].sum() == 0:
            log('No read of the selected cells found on {}, please check cell barcodes and prefix/suffix'.format(c), level='WARN')
    return counts


def init_counting(_normal, _tumor, sam, W, flist, prefix, suffix, _chrlen):
    global cmd_sam_normal, cmd_sam_tumor, cmd_awk_normal, cmd_awk_tumor, chrlen, wsize
    chrlen = _chrlen
    wsize = W
    # Same filters as RDREstimator; the window of a read is the one containing its (0-based) leftmost position
    cmd_sam_normal = "{} view -F 1796 -q 13 {} {}".format(sam, _normal, "{}")
    cmd_sam_tumor = "{} view -F 1796 -q 13 {} {}".format(sam, _tumor, "{}")
    cmd_awk_normal = ["awk", "-v", "W={}".format(W), '{ X[int(($4 - 1) / W)]++ } END{ for (i in X) print i, X[i] }']
    # Barcodes are extracted exactly as in RDREstimator so that they match the names in total.tsv
    if flist is None:  # pooled mode: every read of the tumor BAM
        cmd_awk_tumor = cmd_awk_normal
        return
    cmd_awk_tumor = ["awk", "-v", "W={}".format(W), "-v", "L={}".format(flist), "-v", "P={}".format(len(prefix)),
                     'BEGIN{ while ((getline line < L) > 0) OK[line] = 1 } '
                     '{ if (match($0, /' + prefix + '[ACGT]+' + suffix + '/)) { bc = substr($0, RSTART + P, RLENGTH - P); if (bc in OK) X[int(($4 - 1) / W)]++ } } '
                     'END{ for (i in X) print i, X[i] }']


def counting(job):
    s, c = job
    cmd_sam = cmd_sam_normal if s == 'normal' else cmd_sam_tumor
    cmd_awk = cmd_awk_normal if s == 'normal' else cmd_awk_tumor
    sam = sp.Popen(shlex.split(cmd_sam.format(c)), stdout=sp.PIPE, stderr=sp.PIPE)
    awk = sp.Popen(cmd_awk, stdin=sam.stdout, stdout=sp.PIPE, stderr=sp.PIPE)
    sam.stdout.close()
    stdout, stderr = awk.communicate()
    sam.wait()
    if sam.returncode != 0 or awk.returncode != 0:
        raise RuntimeError('Counting failed for {} on {}: {}'.format(s, c, stderr))
    arr = np.zeros(nwindows(chrlen[c], wsize), dtype=np.float64)
    if not isinstance(stdout, str):
        stdout = stdout.decode('utf-8')
    for l in stdout.strip().split('\n'):
        if l:
            k, v = l.split()
            k = int(k)
            if 0 <= k < arr.size:
                arr[k] += int(v)
    return s, c, arr


def load_counts(f, chrlen, W):
    if f is None or not os.path.isfile(f):
        return None
    d = np.load(f)
    if int(d['__windowsize__']) != W:
        raise ValueError('Counts cache {} has window size {}, not {}'.format(f, int(d['__windowsize__']), W))
    counts = {'normal' : {}, 'tumor' : {}}
    for c in chrlen:
        for s in counts:
            key = '{}|{}'.format(s, c)
            if key not in d.files:
                return None  # cache made for other chromosomes
            counts[s][c] = d[key]
    log('Loaded window counts from {}'.format(f), level='INFO')
    return counts


def save_counts(f, counts, W):
    if f is not None:
        arrays = {'{}|{}'.format(s, c) : counts[s][c] for s in counts for c in counts[s]}
        arrays['__windowsize__'] = np.array(W)
        np.savez_compressed(f, **arrays)
        if not f.endswith('.npz') and os.path.isfile(f + '.npz'):
            os.rename(f + '.npz', f)


def count_pooled_snps(fsnps, chrlen, W):
    """Pooled SNP-covering reads and number of SNPs per window from a TSV CHR POS DEPTH (1-based POS)"""
    import gzip
    names = {norm(c) : c for c in chrlen}
    reads = {c : np.zeros(nwindows(chrlen[c], W), dtype=np.float64) for c in chrlen}
    count = {c : np.zeros(nwindows(chrlen[c], W), dtype=np.float64) for c in chrlen}
    with (gzip.open(fsnps, 'r') if fsnps.endswith('.gz') else open(fsnps, 'r')) as i:
        for l in i:
            if not isinstance(l, str):
                l = l.decode('utf-8')
            p = l.split()
            if not p or p[0][0] == '#' or norm(p[0]) not in names:
                continue
            c = names[norm(p[0])]
            k = (int(p[1]) - 1) // W
            if 0 <= k < reads[c].size:
                reads[c][k] += float(p[2])
                count[c][k] += 1
    return reads, count


def pooled_snp_stats(bins, snpreads, snpcount, ncells, W):
    """SNP fields of bins from window counts (pooled mode has no per-cell counts)"""
    for b in bins:
        a, z = int((b['START'] + W - 1) // W), int((b['END'] + W - 1) // W)
        b['NSNPS'] = int(snpcount[b['CHR']][a:z].sum())
        b['SNP_READS'] = int(snpreads[b['CHR']][a:z].sum())
        b['SNP_READS_PER_CELL'] = '{:.2f}'.format(b['SNP_READS'] / float(ncells))
        b['SNP_READS_MIN_CELL'] = 'NA'
        b['SNP_READS_MEDIAN_CELL'] = 'NA'
    return bins


def nwindows(length, W):
    return int((length + W - 1) // W)


def count_snps(fbaf, chrlen, cells, W):
    """Pseudobulk SNP-covering reads and number of SNPs per window from baf.tsv (#CHR POS CELL A B, 1-based POS)"""
    names = {norm(c) : c for c in chrlen}
    cells = set(cells)
    reads = {c : np.zeros(nwindows(chrlen[c], W), dtype=np.float64) for c in chrlen}
    snps = {c : set() for c in chrlen}
    with open(fbaf, 'r') as i:
        for l in i:
            if l[0] == '#':
                continue
            p = l.strip().split()
            if len(p) < 5 or norm(p[0]) not in names or p[2] not in cells:
                continue
            c, pos = names[norm(p[0])], int(p[1])
            k = (pos - 1) // W
            if 0 <= k < reads[c].size:
                reads[c][k] += int(p[3]) + int(p[4])
                snps[c].add(pos)
    count = {}
    for c in chrlen:
        count[c] = np.zeros(reads[c].size, dtype=np.float64)
        if snps[c]:
            idx = (np.array(sorted(snps[c]), dtype=np.int64) - 1) // W
            np.add.at(count[c], idx[(idx >= 0) & (idx < count[c].size)], 1)
    return reads, count


##################################################
# Binning

def get_thresholds(args, ncells):
    return {
        'snp' : max(args['minsnpreads'], args['minsnpreadspercell'] * ncells),
        'nsnps' : args['minsnps'],
        'tumor' : max(args['mintotalreads'], args['mintotalreadspercell'] * ncells),
        'normal' : args['mintotalreads']
    }


def merge_bins_to_bbs(bin_reads, bin_nsnps, min_snp_reads_vec, min_snp_per_bin,
                      bin_total_reads, min_total_reads_vec, require_snp=True):
    """Greedy adaptive merge of consecutive windows into bins.

    Same rule and output as UGP's _merge_bins_to_bbs: a bin covering windows
    [s, i) closes at the first boundary i (1 <= i <= B-1) where
        snp_ok   = sum of bin_reads[s:i, j] >= min_snp_reads_vec[j] for every j
                   and sum of bin_nsnps[s:i] >= min_snp_per_bin
        total_ok = sum of bin_total_reads[s:i, j] >= min_total_reads_vec[j] for every j
        (snp_ok or not require_snp) and total_ok.
    The trailing run is its own bin if it meets the same test (and is not the
    whole input), otherwise it is merged into the previous bin.
    Since counts are non-negative, the closing boundary is found by binary
    search on cumulative sums, so the cost is O(#bins * log #windows) instead
    of a Python loop over all windows.

    Unlike UGP, min_total_reads may differ per column (UGP uses one value).

    Returns
        bb_ids : (B,) int64 bin index of every window
        n_bbs  : number of bins closed before the last-run adjustment (as UGP)
    """
    B = len(bin_nsnps)
    bb_ids = np.zeros(B, dtype=np.int64)
    if B == 0:
        return bb_ids, 0
    bin_reads = np.asarray(bin_reads, dtype=np.float64).reshape(B, -1)
    bin_total_reads = np.asarray(bin_total_reads, dtype=np.float64).reshape(B, -1)

    def cum(x):
        out = np.zeros(B + 1, dtype=np.float64)
        np.cumsum(x, out=out[1:])
        return out

    # Counts are integers, so acc >= t is equivalent to acc >= ceil(t)
    crit = []
    if require_snp:
        crit.extend((cum(bin_reads[:, j]), np.ceil(min_snp_reads_vec[j])) for j in range(bin_reads.shape[1]))
        crit.append((cum(bin_nsnps), np.ceil(min_snp_per_bin)))
    min_total_reads_vec = np.broadcast_to(np.asarray(min_total_reads_vec, dtype=np.float64), (bin_total_reads.shape[1],))
    crit.extend((cum(bin_total_reads[:, j]), np.ceil(min_total_reads_vec[j])) for j in range(bin_total_reads.shape[1]))

    def close(s):
        # smallest i > s with C[i] - C[s] >= t for every criterion; B + 1 if none
        i = s + 1
        for C, t in crit:
            i = max(i, int(np.searchsorted(C, C[s] + t, side='left')))
        return i

    bb_id, s = 0, 0
    i = close(s)
    while i <= B - 1:
        bb_ids[s:i] = bb_id
        bb_id += 1
        s = i
        i = close(s)
    if i == B and s > 0:
        bb_ids[s:] = bb_id
        bb_id += 1
    else:
        bb_ids[s:] = bb_id - 1 if bb_id > 0 else 0
    return bb_ids, bb_id


def read_given(f, chrlen):
    """Precomputed bins: {chr : [(start, end, region)]} with chromosome names of the reference.
    Columns CHR START END are the first three; a column named region_id or REGION is kept."""
    import gzip
    names = {norm(c) : c for c in chrlen}
    res = defaultdict(list)
    ridx = None
    op = gzip.open if f.endswith('.gz') else open
    with op(f, 'r') as i:
        for l in i:
            if not isinstance(l, str):
                l = l.decode('utf-8')
            p = l.rstrip('\n').split('\t') if '\t' in l else l.split()
            if not p:
                continue
            if l.startswith('##'):
                continue
            if l.startswith('#'):
                head = [h.lstrip('#') for h in p]
                ridx = next((x for x, h in enumerate(head) if h in ('region_id', 'REGION')), None)
                continue
            if norm(p[0]) not in names:
                continue
            c = names[norm(p[0])]
            s, e = int(p[1]), min(int(p[2]), chrlen[c])
            if not s < e:
                continue
            region = '{}_{}'.format(c, p[ridx]) if ridx is not None and ridx < len(p) else c
            res[c].append((s, e, region))
    res = {c : sorted(res[c]) for c in res}
    for c in res:
        if any(a[1] > b[0] for a, b in zip(res[c][:-1], res[c][1:])):
            raise ValueError('Found overlapping bins on {} in {}'.format(c, f))
    if len(res) == 0:
        raise ValueError('No bin found in {} on the given chromosomes'.format(f))
    return res


def describe_bins(given, counts, snpreads, snpcount, chrlen, args, ncells):
    """Window counts of precomputed bins; a window belongs to the bin containing its start.
    SNP fields are filled exactly from baf.tsv by percell_snp_stats."""
    W = args['windowsize']
    bins, windows = [], []
    for c in sorted(given, key=orderchrs):
        for s, e, region in given[c]:
            a, z = int((s + W - 1) // W), int((e + W - 1) // W)
            ks = np.arange(a, z)
            wstart = np.maximum(s, ks * W)
            wend = np.minimum(np.minimum(e, (ks + 1) * W), chrlen[c])
            tum, nor = counts['tumor'][c][a:z], counts['normal'][c][a:z]
            rec = {'CHR' : c, 'START' : s, 'END' : e, 'SIZE' : e - s, 'REGION' : region, 'NWINDOWS' : int(z - a),
                   'TUMOR_READS' : int(tum.sum()), 'NORMAL_READS' : int(nor.sum()), 'CLOSE' : 'given', 'LIMIT' : 'given',
                   'BIN' : '{}:{}-{}'.format(c, s, e)}
            rec['READS_PER_CELL'] = '{:.1f}'.format(rec['TUMOR_READS'] / float(ncells))
            bins.append(rec)
            if z > a:
                windows.append((c, wstart, wend, region, rec['BIN'], tum, nor, snpreads[c][a:z], snpcount[c][a:z]))
    return bins, windows


def percell_snp_stats(bins, fbaf, chrlen, cells):
    """SNP-covering reads of every selected cell in every bin, from baf.tsv. Sets NSNPS, SNP_READS,
    SNP_READS_PER_CELL (mean), SNP_READS_MIN_CELL and SNP_READS_MEDIAN_CELL (over all selected
    cells, zeros included)."""
    import bisect
    names = {norm(c) : c for c in chrlen}
    cidx = {e : x for x, e in enumerate(cells)}
    bychr = defaultdict(list)
    for x, b in enumerate(bins):
        bychr[b['CHR']].append((b['START'], b['END'], x))
    starts = {c : [s for s, e, x in sorted(bychr[c])] for c in bychr}
    ordered = {c : sorted(bychr[c]) for c in bychr}
    mat = np.zeros((len(bins), len(cells)), dtype=np.float64)
    snps = [set() for b in bins]
    with open(fbaf, 'r') as i:
        for l in i:
            if l[0] == '#':
                continue
            p = l.split()
            if len(p) < 5 or norm(p[0]) not in names or p[2] not in cidx:
                continue
            c = names[norm(p[0])]
            if c not in starts:
                continue
            pos0 = int(p[1]) - 1
            y = bisect.bisect_right(starts[c], pos0) - 1
            if y >= 0:
                s, e, x = ordered[c][y]
                if s <= pos0 < e:
                    mat[x, cidx[p[2]]] += int(p[3]) + int(p[4])
                    snps[x].add(pos0)
    for x, b in enumerate(bins):
        b['NSNPS'] = len(snps[x])
        b['SNP_READS'] = int(mat[x].sum())
        b['SNP_READS_PER_CELL'] = '{:.2f}'.format(mat[x].mean())
        b['SNP_READS_MIN_CELL'] = int(mat[x].min())
        b['SNP_READS_MEDIAN_CELL'] = '{:.1f}'.format(np.median(mat[x]))
    return bins


CRITERIA = ['snp_reads', 'snps', 'tumor_reads', 'normal_reads']


def explain(a, z, crit):
    """How the bin of windows [a, z) was decided.

    For every criterion, the number of windows from a that are needed to meet it.
    Returns (CLOSE, LIMIT, need) where
      CLOSE = 'met'       the bin closed as soon as all criteria were met
              'met+tail'  all criteria were met before z, the windows after that are a
                          trailing run of the region that missed them and was absorbed
              'unmet'     some criterion is not met even with all windows (the whole
                          region is too poor in reads and it is a single bin)
      LIMIT = criteria met last (they decided the boundary), or the unmet ones
    """
    need = {}
    for name, C, t in crit:
        need[name] = max(1, int(np.searchsorted(C, C[a] + t, side='left')) - a)
    unmet = [n for n in CRITERIA if n in need and a + need[n] > z]
    if unmet:
        return 'unmet', '+'.join(unmet), need
    m = max(need.values())
    limit = '+'.join(n for n in CRITERIA if n in need and need[n] == m)
    return ('met' if a + m == z else 'met+tail'), limit, need


def make_bins(regions, counts, snpreads, snpcount, chrlen, thres, args, ncells):
    W = args['windowsize']
    bins = []
    windows = []
    cum = (lambda x : np.concatenate([[0.0], np.cumsum(x)]))
    for c in sorted(regions, key=orderchrs):
        for rid, (rs, re_) in enumerate(regions[c]):
            region = '{}_{}'.format(c, rid)
            ks, ke = rs // W, nwindows(re_, W)
            norm_r = counts['normal'][c][ks:ke]
            if args['trim']:
                nz = np.nonzero(norm_r > 0)[0]
                if nz.size == 0:
                    log('Region {}:{}-{} has no normal reads and is skipped'.format(c, rs, re_), level='WARN')
                    continue
                ks, ke = ks + nz[0], ks + nz[-1] + 1
            tum = counts['tumor'][c][ks:ke]
            nor = counts['normal'][c][ks:ke]
            srd = snpreads[c][ks:ke]
            sct = snpcount[c][ks:ke]
            ids, n = merge_bins_to_bbs(srd.reshape(-1, 1), sct, [thres['snp']], thres['nsnps'],
                                       np.column_stack([tum, nor]), [thres['tumor'], thres['normal']])
            crit = [('snp_reads', cum(srd), np.ceil(thres['snp'])), ('snps', cum(sct), np.ceil(thres['nsnps'])),
                    ('tumor_reads', cum(tum), np.ceil(thres['tumor'])), ('normal_reads', cum(nor), np.ceil(thres['normal']))]
            wstart = np.maximum(rs, (ks + np.arange(ids.size)) * W)
            wend = np.minimum(np.minimum(re_, (ks + np.arange(ids.size) + 1) * W), chrlen[c])
            # ids is non-decreasing, so every bin is a run of consecutive windows
            cuts = np.concatenate([[0], np.flatnonzero(np.diff(ids)) + 1, [ids.size]])
            for a, z in zip(cuts[:-1], cuts[1:]):
                close, limit, need = explain(a, z, crit)
                rec = {'CHR' : c, 'START' : int(wstart[a]), 'END' : int(wend[z - 1]), 'REGION' : region,
                       'NWINDOWS' : int(z - a), 'NSNPS' : int(sct[a:z].sum()), 'SNP_READS' : int(srd[a:z].sum()),
                       'TUMOR_READS' : int(tum[a:z].sum()), 'NORMAL_READS' : int(nor[a:z].sum()),
                       'CLOSE' : close, 'LIMIT' : limit}
                rec['SNP_READS_PER_CELL'] = '{:.2f}'.format(rec['SNP_READS'] / float(ncells))
                rec['READS_PER_CELL'] = '{:.1f}'.format(rec['TUMOR_READS'] / float(ncells))
                rec['SIZE'] = rec['END'] - rec['START']
                if rec['NSNPS'] == 0 and not args['keepsnpfree']:
                    log('Dropping bin {}:{}-{} without SNPs'.format(c, rec['START'], rec['END']), level='WARN')
                    rec['BIN'] = 'dropped'
                else:
                    rec['BIN'] = '{}:{}-{}'.format(c, rec['START'], rec['END'])
                    bins.append(rec)
                windows.append((c, wstart[a:z], wend[a:z], region, rec['BIN'], tum[a:z], nor[a:z], srd[a:z], sct[a:z]))
    if len(bins) == 0:
        raise ValueError('No bin has been created!')
    return bins, windows


FIELDS = ['CHR', 'START', 'END', 'SIZE', 'REGION', 'NWINDOWS', 'NSNPS', 'SNP_READS', 'TUMOR_READS', 'NORMAL_READS',
          'SNP_READS_PER_CELL', 'SNP_READS_MEDIAN_CELL', 'SNP_READS_MIN_CELL', 'READS_PER_CELL', 'CLOSE', 'LIMIT']


def header_lines(meta):
    return ['##{}={}'.format(k, meta[k]) for k in sorted(meta)]


def write_bins(bins, meta, stdout_file=None):
    lines = header_lines(meta) + ['#' + '\t'.join(FIELDS)] + ['\t'.join(str(b[f]) for f in FIELDS) for b in bins]
    if stdout_file is not None:
        with open(stdout_file, 'w') as o:
            o.write('\n'.join(lines) + '\n')
    else:
        sys.stdout.write('\n'.join(lines) + '\n')


def write_summary(bins, f):
    sizes = np.array([b['END'] - b['START'] for b in bins], dtype=np.float64)
    with open(f, 'w') as o:
        o.write('BINS\tMEAN_SIZE\tMEDIAN_SIZE\n{}\t{:.0f}\t{:.0f}\n'.format(len(bins), sizes.mean(), np.median(sizes)))


def write_windows(windows, meta, f):
    import gzip
    o = gzip.open(f, 'wt' if sys.version_info[0] >= 3 else 'w') if f.endswith('.gz') else open(f, 'w')
    try:
        o.write('\n'.join(header_lines(meta)) + '\n')
        o.write('#CHR\tSTART\tEND\tREGION\tBIN\tTUMOR_READS\tNORMAL_READS\tSNP_READS\tNSNPS\n')
        for c, ws, we, region, b, tum, nor, srd, sct in windows:
            for x in range(len(ws)):
                o.write('{}\t{}\t{}\t{}\t{}\t{:.0f}\t{:.0f}\t{:.0f}\t{:.0f}\n'.format(c, ws[x], we[x], region, b, tum[x], nor[x], srd[x], sct[x]))
    finally:
        o.close()


def summarize(bins, ncells):
    span = np.array([b['SIZE'] for b in bins], dtype=np.float64)
    snpc = np.array([b['SNP_READS'] for b in bins], dtype=np.float64) / ncells
    q = (lambda x : ' '.join('{}={:.4g}'.format(n, v) for n, v in zip(['min', 'q25', 'median', 'q75', 'max'], np.percentile(x, [0, 25, 50, 75, 100]))))
    log('Number of bins: {}'.format(len(bins)), level='INFO')
    log('Bin size (bp): {}'.format(q(span)), level='INFO')
    log('Mean SNP-covering reads per cell per bin: {}'.format(q(snpc)), level='INFO')
    if all(b['SNP_READS_MIN_CELL'] != 'NA' for b in bins):
        log('Minimum over cells of SNP-covering reads per bin: {}'.format(q(np.array([b['SNP_READS_MIN_CELL'] for b in bins], dtype=np.float64))), level='INFO')
    count = (lambda key : ', '.join('{}: {}'.format(k, v) for k, v in sorted(Counter(b[key] for b in bins).items())))
    log('How bins closed: {}'.format(count('CLOSE')), level='INFO')
    log('Criteria deciding the boundaries: {}'.format(count('LIMIT')), level='INFO')
    rows = [['REGION', 'START', 'END', 'BINS', 'MEDIAN_SIZE', 'MIN_SIZE', 'MAX_SIZE', 'LIMITS']]
    for r in sorted(set(b['REGION'] for b in bins), key=(lambda r : (orderchrs(r.rsplit('_', 1)[0]), int(r.rsplit('_', 1)[1])))):
        rb = [b for b in bins if b['REGION'] == r]
        sz = [b['SIZE'] for b in rb]
        rows.append([r, str(min(b['START'] for b in rb)), str(max(b['END'] for b in rb)), str(len(rb)), str(int(np.median(sz))),
                     str(min(sz)), str(max(sz)), ', '.join('{}:{}'.format(k, v) for k, v in sorted(Counter(b['LIMIT'] for b in rb).items()))])
    width = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    log('Bins per region:\n' + '\n'.join('  '.join(v.ljust(w) for v, w in zip(row, width)) for row in rows), level='INFO')


if __name__ == '__main__':
    main()
