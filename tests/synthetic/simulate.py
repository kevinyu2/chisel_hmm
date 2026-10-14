#!/usr/bin/env python3
"""
Simulate a small barcoded single-cell DNA-seq dataset with known haplotype-specific
copy numbers, to test CHISEL end-to-end with fixed-size and variable-length bins.

Design (all coordinates 0-based, half-open):
  * two chromosomes (chr1 24 Mb, chr2 16 Mb) with an N-gap "centromere" without reads;
  * heterozygous SNP density alternates between dense (1 / 600 bp) and sparse
    (1 / 10 kb) stretches, as variable-length bins are designed for;
  * three populations of cells: normal diploid, clone A, clone B, with arm-level,
    focal (~1 Mb, placed in SNP-dense stretches), LOH and mirrored-subclonal events.

Outputs in OUTDIR: ref.fa(+fai,dict), normal.bam, cells.bam (+bai), phased.tsv
(#CHR POS PHASE), regions.bed (chromosome arms), truth.tsv (per-population
segments) and cells.tsv (cell -> population).

Requires numpy and samtools in PATH (or SAMTOOLS env variable).
"""
import os, sys, argparse, subprocess
import numpy as np

SAMTOOLS = os.environ.get('SAMTOOLS', 'samtools')
READLEN = 100
BASES = np.array(list('ACGT'))

CHROMS = {'chr1': 24000000, 'chr2': 16000000}
CENTROMERES = {'chr1': (11000000, 12000000), 'chr2': (7000000, 7800000)}

# SNP density stretches: (chrom, start, end, mean distance between het SNPs)
DENSITY = [
    ('chr1', 0, 4000000, 600), ('chr1', 4000000, 8000000, 10000), ('chr1', 8000000, 11000000, 600),
    ('chr1', 12000000, 14000000, 10000), ('chr1', 14000000, 19000000, 600), ('chr1', 19000000, 24000000, 10000),
    ('chr2', 0, 3000000, 10000), ('chr2', 3000000, 7000000, 600), ('chr2', 7800000, 10000000, 10000),
    ('chr2', 10000000, 16000000, 600),
]

# Haplotype-specific copy numbers (a, b) that differ from (1, 1); later events override earlier ones
EVENTS = {
    'normal': [],
    'A': [('chr1', 0, 11000000, (2, 1)),          # arm gain (clonal)
          ('chr1', 15500000, 16500000, (3, 1)),   # focal amplification, 1 Mb in a dense stretch
          ('chr1', 19000000, 24000000, (1, 0)),   # LOH (clonal)
          ('chr2', 3000000, 7000000, (2, 2)),     # balanced gain
          ('chr2', 13000000, 16000000, (2, 1))],  # mirrored-subclonal with B
    'B': [('chr1', 0, 11000000, (2, 1)),
          ('chr1', 8500000, 9500000, (2, 0)),     # focal copy-neutral LOH in the gained arm, 1 Mb
          ('chr1', 19000000, 24000000, (1, 0)),
          ('chr2', 11000000, 12000000, (1, 0)),   # focal deletion, 1 Mb in a dense stretch
          ('chr2', 13000000, 16000000, (1, 2))],
}


def segments(pop):
    """Piecewise-constant (chrom, start, end, a, b) covering every chromosome except centromeres"""
    out = []
    for c, L in CHROMS.items():
        cs, ce = CENTROMERES[c]
        cuts = {0, L, cs, ce}
        for ec, s, e, _ in EVENTS[pop]:
            if ec == c:
                cuts |= {s, e}
        cuts = sorted(cuts)
        for s, e in zip(cuts[:-1], cuts[1:]):
            if cs <= s and e <= ce:
                continue
            cn = (1, 1)
            for ec, es, ee, ecn in EVENTS[pop]:
                if ec == c and es <= s and e <= ee:
                    cn = ecn
            out.append((c, s, e) + cn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--ncells', type=int, nargs=3, default=[12, 40, 28], help='normal, clone A, clone B cells')
    ap.add_argument('--cellcov', type=float, default=0.06, help='per-cell coverage of a diploid genome')
    ap.add_argument('--normalcov', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)

    # Reference with N centromeres
    ref = {}
    with open(out('ref.fa'), 'w') as o:
        for c, L in CHROMS.items():
            seq = BASES[rng.randint(0, 4, size=L)]
            cs, ce = CENTROMERES[c]
            seq[cs:ce] = 'N'
            ref[c] = seq
            o.write('>{}\n'.format(c))
            s = ''.join(seq)
            o.write('\n'.join(s[i:i + 60] for i in range(0, L, 60)) + '\n')
    subprocess.check_call([SAMTOOLS, 'faidx', out('ref.fa')])
    subprocess.check_call([SAMTOOLS, 'dict', out('ref.fa'), '-o', out('ref.dict')])

    # Heterozygous SNPs: alleles of haplotype A / B, true phase
    snps = {}
    with open(out('phased.tsv'), 'w') as o:
        for c, s, e, d in DENSITY:
            pos = np.sort(np.unique(rng.randint(s + READLEN, e - READLEN, size=(e - s) // d)))
            snps.setdefault(c, []).append(pos)
        for c in CHROMS:
            pos = np.concatenate(snps[c])
            refb = ref[c][pos]
            alt = np.array([BASES[(np.where(BASES == r)[0][0] + rng.randint(1, 4)) % 4] for r in refb])
            phase = rng.randint(0, 2, size=pos.size)  # 0: hapA=REF (0|1), 1: hapA=ALT (1|0)
            hapA = np.where(phase == 0, refb, alt)
            hapB = np.where(phase == 0, alt, refb)
            snps[c] = (pos, hapA, hapB)
            for p, ph in zip(pos, phase):
                o.write('{}\t{}\t{}\n'.format(c, p + 1, '0|1' if ph == 0 else '1|0'))

    header = '@HD\tVN:1.6\tSO:unsorted\n' + ''.join('@SQ\tSN:{}\tLN:{}\n'.format(c, L) for c, L in CHROMS.items())
    quality = 'I' * READLEN

    def reads_sam(segs, nreads, tag):
        """Reads drawn proportionally to length * total copy number; haplotype by allele-specific copy number"""
        w = np.array([(e - s - READLEN) * (a + b) for c, s, e, a, b in segs], dtype=float)
        n = rng.multinomial(nreads, w / w.sum())
        lines = []
        for (c, s, e, a, b), k in zip(segs, n):
            if k == 0:
                continue
            starts = rng.randint(s, e - READLEN, size=k)
            onA = rng.uniform(size=k) < (a / float(a + b))
            pos, hA, hB = snps[c]
            lo = np.searchsorted(pos, starts)
            hi = np.searchsorted(pos, starts + READLEN)
            for st, A, l, h in zip(starts, onA, lo, hi):
                seq = ref[c][st:st + READLEN].copy()
                if h > l:
                    seq[pos[l:h] - st] = (hA if A else hB)[l:h]
                lines.append('r{}_{}\t{}\t{}\t{}\t60\t{}M\t*\t0\t0\t{}\t{}{}\n'.format(
                    tag, len(lines), 16 if rng.randint(2) else 0, c, st + 1, READLEN, ''.join(seq), quality,
                    '\tCB:Z:' + tag if tag != 'N' else ''))
        return lines

    def write_bam(lines, f):
        p = subprocess.Popen('{} view -b -o - - | {} sort -o {} -'.format(SAMTOOLS, SAMTOOLS, f), shell=True, stdin=subprocess.PIPE)
        p.stdin.write(header.encode())
        for l in lines:
            p.stdin.write(l.encode())
        p.stdin.close()
        assert p.wait() == 0
        subprocess.check_call([SAMTOOLS, 'index', f])

    genome = sum(L - (CENTROMERES[c][1] - CENTROMERES[c][0]) for c, L in CHROMS.items())
    print('normal BAM', file=sys.stderr)
    write_bam(reads_sam(segments('normal'), int(args.normalcov * genome / READLEN), 'N'), out('normal.bam'))

    pops = ['normal'] * args.ncells[0] + ['A'] * args.ncells[1] + ['B'] * args.ncells[2]
    barcodes = set()
    while len(barcodes) < len(pops):
        barcodes.add(''.join(BASES[rng.randint(0, 4, size=12)]))
    barcodes = sorted(barcodes)
    rng.shuffle(barcodes)
    lines = []
    with open(out('cells.tsv'), 'w') as o:
        o.write('#CELL\tPOPULATION\n')
        for bc, pop in zip(barcodes, pops):
            segs = segments(pop)
            mean_cn = sum((e - s) * (a + b) for c, s, e, a, b in segs) / float(sum(e - s for c, s, e, a, b in segs))
            nreads = rng.poisson(args.cellcov * genome / READLEN * mean_cn / 2.0)
            lines.extend(reads_sam(segs, nreads, bc))
            o.write('{}\t{}\n'.format(bc, pop))
    print('cells BAM ({} reads)'.format(len(lines)), file=sys.stderr)
    write_bam(lines, out('cells.bam'))

    with open(out('regions.bed'), 'w') as o:
        for c, L in CHROMS.items():
            cs, ce = CENTROMERES[c]
            o.write('{}\t0\t{}\t{}_p\n{}\t{}\t{}\t{}_q\n'.format(c, cs, c, c, ce, L, c))
    with open(out('truth.tsv'), 'w') as o:
        o.write('#POPULATION\tCHR\tSTART\tEND\tA\tB\n')
        for pop in EVENTS:
            for seg in segments(pop):
                o.write('\t'.join(map(str, (pop,) + seg)) + '\n')
    print('done', file=sys.stderr)


if __name__ == '__main__':
    main()
