#!/usr/bin/env python2.7

import os, sys
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import argparse
import chisel

src = os.path.dirname(chisel.__file__)
from ..Utils import *


def parse_args():
    description = "CHISEL command to run the complete pipeline starting from the 4 required data: (1) Barcoded single-cell BAM; (2) Matched-normal BAM; (3) Reference genome; (4) Phased VCF."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-x","--rundir", required=False, default='./', type=str, help="Running directory (default: current directory)")
    parser.add_argument("-t","--tumor", required=True, type=str, help="Barcoded single-cell BAM file")
    parser.add_argument("-n","--normal", required=True, type=str, help="Matched-normal BAM file")
    parser.add_argument("-r","--reference", type=str, required=True, help="Reference genome")
    parser.add_argument("-l","--listphased", type=str, required=True, help="Phased SNPs file (lines of heterozygous germline SNPs must contain either 0|1 or 1|0)")
    parser.add_argument("-b","--size", type=str, required=False, default="5Mb", help="Bin size, with or without \"kb\" or \"Mb\"")
    parser.add_argument("-k", "--blocksize", required=False, type=str, default="50kb", help="Size of the haplotype blocks (default: 50kb, use 0 to disable)")
    parser.add_argument("-c", "--chromosomes", type=str, required=False, default=' '.join(['chr{}'.format(i) for i in range(1, 23)]), help="Space-separeted list of chromosomes between apices (default: \"chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22\")")
    parser.add_argument("-m","--minreads", type=int, required=False, default=300000, help="Minimum number total reads to select cells (default: 300000)")
    parser.add_argument("-p","--maxploidy", required=False, type=int, default=4, help="Maximum total copy number to consider for balanced cluster (default: 4, corresponding to a WGD)")
    parser.add_argument("-K","--upperk", required=False, type=int, default=100, help="Maximum number of bin clusters (default: 100, use 0 to consider maximum number of clusters)")
    parser.add_argument("--addgccorr", required=False, default=False, action='store_true', help="Add additional custome correction for GC bias (default: disabled)")
    parser.add_argument("--nophasecorr", required=False, default=False, action='store_true', help="Disable correction for given phasing bias (default: enabled)")
    parser.add_argument("--bcftools", required=False, default=None, type=str, help="Path to the directory to \"bcftools\" executable, required in default mode (default: bcftools is directly called as it is in user $PATH)")
    parser.add_argument("--samtools", required=False, default=None, type=str, help="Path to the directory to \"samtools\" executable, required in default mode (default: samtools is directly called as it is in user $PATH)")
    parser.add_argument("--cellprefix", type=str, required=False, default='CB:Z:', help="Prefix of cell barcode field in SAM format (default: CB:Z:)")
    parser.add_argument("--cellsuffix", type=str, required=False, default=None, help="Suffix of cell barcode field in SAM format (default: none)")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: None)")
    parser.add_argument("-j","--jobs", required=False, type=int, default=0, help="Number of parallele jobs to use (default: equal to number of available processors)")
    vb = parser.add_argument_group('variable-length bins', 'Replace the fixed-size bins of -b/--size by bins that are merged from small windows until the pseudobulk of all cells has enough SNP-covering and total reads (window-merge binning of the Universal-Genotyping-Pipeline / HATCHet2). Fixed-size bins are still computed first to select cells.')
    vb.add_argument("--bins", type=str, required=False, default=None, help="Use these precomputed variable-length bins, e.g. UGP's bb/MSR<n>/bulk/bb.tsv.gz or any TSV/BED with columns CHR START END (default: none)")
    vb.add_argument("--varbins", required=False, default=False, action='store_true', help="Compute variable-length bins within CHISEL with the options below (default: disabled)")
    vb.add_argument("--windowsize", type=str, required=False, default="1kb", help="Size of the windows merged into bins (default: 1kb)")
    vb.add_argument("--minsnpreadspercell", type=float, required=False, default=25.0, help="Minimum mean number of SNP-covering reads per cell in every bin (default: 25)")
    vb.add_argument("--minsnpreads", type=float, required=False, default=0.0, help="Minimum number of SNP-covering reads in the tumor pseudobulk of every bin (default: 0)")
    vb.add_argument("--mintotalreadspercell", type=float, required=False, default=0.0, help="Minimum mean number of read starts per cell in every bin (default: 0)")
    vb.add_argument("--mintotalreads", type=float, required=False, default=5000.0, help="Minimum number of read starts in every bin in both the tumor pseudobulk and the matched normal (default: 5000)")
    vb.add_argument("--minsnps", type=int, required=False, default=1, help="Minimum number of heterozygous SNPs in every bin (default: 1)")
    vb.add_argument("--regions", type=str, required=False, default=None, help="BED file of regions that bins cannot span (default: chromosome arms of --refversion)")
    vb.add_argument("--refversion", type=str, required=False, default='auto', choices=['auto', 'hg19', 'hg38', 'none'], help="Reference version for chromosome arms (default: auto-detected from chr1 length)")
    vb.add_argument("--blacklist", type=str, required=False, default=None, help="BED file of regions to exclude from bins (default: none)")
    args = parser.parse_args()

    if not os.path.isdir(args.rundir):
        raise ValueError("Running directory does not exists: {}".format(args.rundir))
    if not os.path.isfile(args.tumor):
        raise ValueError("Barcoded single-cell BAM file does not exist: {}".format(args.tumor))
    if not os.path.isfile(args.normal):
        raise ValueError("Matched-normal BAM file does not exist: {}".format(args.normal))
    if not os.path.isfile(args.reference):
        raise ValueError("Reference genome file does not exist: {}".format(args.reference))
    if not os.path.isfile(args.listphased):
        raise ValueError("Phased SNPs file does not exist: {}".format(args.listphased))
    if args.seed and args.seed < 1:
        raise ValueError("The random seed  must be positive!")
    if args.minreads < 1:
        raise ValueError("The minimum number of reads must be positive!")
    if args.maxploidy < 3:
        raise ValueError("The maximum total copy number to consider for balanced cluster must be at least 2!")
    if args.upperk < 1:
        raise ValueError("The maximum number of clusters must be positive!")
    if args.bins is not None and args.varbins:
        raise ValueError("Use either --bins (precomputed bins) or --varbins (bins computed by CHISEL), not both!")
    if args.bins is not None:
        if not os.path.isfile(args.bins):
            raise ValueError("Bins file does not exist: {}".format(args.bins))
        args.bins = os.path.abspath(args.bins)
    for f in [args.regions, args.blacklist]:
        if f is not None and not os.path.isfile(f):
            raise ValueError("File does not exist: {}".format(f))
    for f in ['regions', 'blacklist']:
        if getattr(args, f) is not None:
            setattr(args, f, os.path.abspath(getattr(args, f)))

    size = 0
    try:
        if args.size[-2:] == "kb":
            size = int(args.size[:-2]) * 1000
        elif args.size[-2:] == "Mb":
            size = int(args.size[:-2]) * 1000000
        else:
            size = int(args.size)
    except:
        raise ValueError("Size must be a number, optionally ending with either \"kb\" or \"Mb\"!")

    blocksize = 0
    try:
        if args.blocksize[-2:] == "kb":
            blocksize = int(args.blocksize[:-2]) * 1000
        elif args.blocksize[-2:] == "Mb":
            blocksize = int(args.blocksize[:-2]) * 1000000
        else:
            blocksize = int(args.blocksize)
    except:
        raise ValueError("Size must be a number, optionally ending with either \"kb\" or \"Mb\"!")

    if not args.jobs:
        args.jobs = mp.cpu_count()
    if args.jobs < 1:
        raise ValueError("The number of jobs must be positive!")

    bcftools = args.bcftools
    if not bcftools:
        bcftools = "bcftools"
    if which(bcftools) is None:
        raise ValueError("bcftools has not been found or is not executable!")

    samtools = args.samtools
    if not samtools:
        samtools = "samtools"
    if which(samtools) is None:
        raise ValueError("samtools has not been found or is not executable!")

    return {
        "rundir" : args.rundir,
        "tumor" : args.tumor,
        "normal" : args.normal,
        "reference" : args.reference,
        "listphased" : args.listphased,
        "binsize" : size,
        "blocksize" : blocksize,
        "chromosomes" : args.chromosomes,
        "minreads" : args.minreads,
        "addgccorr" : args.addgccorr,
        "phasecorr" : not args.nophasecorr,
        "bcftools" : bcftools,
        "samtools" : samtools,
        "maxploidy" : args.maxploidy,
        "upperk" : args.upperk,
        "cellprefix" : args.cellprefix,
        "cellsuffix" : args.cellsuffix,
        "seed" : args.seed,
        "jobs" : args.jobs,
        "varbins" : args.varbins,
        "givenbins" : args.bins,
        "windowsize" : args.windowsize,
        "minsnpreadspercell" : args.minsnpreadspercell,
        "minsnpreads" : args.minsnpreads,
        "mintotalreadspercell" : args.mintotalreadspercell,
        "mintotalreads" : args.mintotalreads,
        "minsnps" : args.minsnps,
        "regions" : args.regions,
        "refversion" : args.refversion,
        "blacklist" : args.blacklist
    }


def main():
    log('Parsing and checking arguments', level='PROGRESS')
    args = parse_args()
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]) + '\n', level='INFO')

    log('Setting directories', level='PROGRESS')
    dbaf, drdr, dcom, dcal, dclo, dplo = setup(args)
    def get_comp(name):
        comp = os.path.join(src, name)
        if not os.path.isfile(comp):
            raise ValueError("{} not found in src directory of bin i.e. {}, is anything been moved?".format(name, src))
        return comp

    log('Computing RDRs', level='PROGRESS')
    cmd = 'python2.7 {} -n {} -t {} -r {} -b {} -m {} -j {} -c \"{}\" --outdir {}'
    cmd = cmd.format(get_comp('RDREstimator.py'), args['normal'], args['tumor'], args['reference'], args['binsize'], args['minreads'], args['jobs'], args['chromosomes'], drdr)
    if args['samtools'] is not None:
        cmd += " -s {}".format(args['samtools'])
    cmd += " --cellprefix {}".format(args['cellprefix'])
    if args['cellsuffix'] is not None:
        cmd += " --cellsuffix {}".format(args['cellsuffix'])
    runcmd(cmd, drdr, out='rdr.tsv')
    lcel = os.path.join(drdr, 'total.tsv')
    rdr = os.path.join(drdr, 'rdr.tsv')

    log('Computing BAFs', level='PROGRESS')
    cmd = 'python2.7 {} -n {} -t {} -r {} -j {} -c {} -l {}'
    cmd = cmd.format(get_comp('BAFEstimator.py'), args['normal'], args['tumor'], args['reference'], args['jobs'], lcel, args['listphased'])
    if args['samtools'] is not None:
        cmd += " -s {}".format(args['samtools'])
    if args['bcftools'] is not None:
        cmd += " -b {}".format(args['bcftools'])
    cmd += " --cellprefix {}".format(args['cellprefix'])
    if args['cellsuffix'] is not None:
        cmd += " --cellsuffix {}".format(args['cellsuffix'])
    runcmd(cmd, dbaf, out='baf.tsv')
    baf = os.path.join(dbaf, 'baf.tsv')

    if args['varbins'] or args['givenbins']:
        dbin = os.path.join(args['rundir'], 'bins')
        if not os.path.isdir(dbin):
            os.mkdir(dbin)

        if args['givenbins']:
            log('Describing given variable-length bins', level='PROGRESS')
        else:
            log('Computing variable-length bins', level='PROGRESS')
        cmd = 'python2.7 {} -t {} -n {} -r {} -l {} -b {} -c \"{}\" -w {} -j {} --outdir {}'
        cmd = cmd.format(get_comp('Binner.py'), args['tumor'], args['normal'], args['reference'], lcel, baf, args['chromosomes'], args['windowsize'], args['jobs'], dbin)
        if args['givenbins']:
            cmd += ' --given {}'.format(args['givenbins'])
        else:
            cmd += ' --minsnpreadspercell {} --minsnpreads {} --mintotalreadspercell {} --mintotalreads {} --minsnps {} --refversion {}'.format(
                    args['minsnpreadspercell'], args['minsnpreads'], args['mintotalreadspercell'], args['mintotalreads'], args['minsnps'], args['refversion'])
            if args['regions'] is not None:
                cmd += " --regions {}".format(args['regions'])
            if args['blacklist'] is not None:
                cmd += " --blacklist {}".format(args['blacklist'])
        if args['samtools'] is not None:
            cmd += " -s {}".format(args['samtools'])
        cmd += " --cellprefix {}".format(args['cellprefix'])
        if args['cellsuffix'] is not None:
            cmd += " --cellsuffix {}".format(args['cellsuffix'])
        cmd += " --windows {}".format(os.path.join(dbin, 'windows.tsv.gz'))
        runcmd(cmd, dbin, out='bins.tsv', log='log_binning')
        fbins = os.path.join(dbin, 'bins.tsv')
        if not os.path.isfile(fbins) or os.path.getsize(fbins) == 0:
            raise RuntimeError('Variable-length binning failed, see {}'.format(os.path.join(dbin, 'log_binning')))

        log('Plotting variable-length bins', level='PROGRESS')
        cmd = 'python2.7 {} {} {} -o {}'.format(get_comp('BinPlotter.py'), fbins, os.path.join(dbin, 'windows.tsv.gz'), os.path.join(dbin, 'bins.pdf'))
        try:
            runcmd(cmd, dbin, log='log_binplot')
        except Exception as e:
            log('Plotting bins failed: {}'.format(e), level='WARN')

        if args['givenbins']:
            fbins = args['givenbins']  # the exact coordinates of the given file
        log('Computing RDRs on variable-length bins', level='PROGRESS')
        cmd = 'python2.7 {} -n {} -t {} -r {} -l {} -j {} -c \"{}\" --bins {} --outdir {}'
        cmd = cmd.format(get_comp('RDREstimator.py'), args['normal'], args['tumor'], args['reference'], lcel, args['jobs'], args['chromosomes'], fbins, dbin)
        if args['samtools'] is not None:
            cmd += " -s {}".format(args['samtools'])
        cmd += " --cellprefix {}".format(args['cellprefix'])
        if args['cellsuffix'] is not None:
            cmd += " --cellsuffix {}".format(args['cellsuffix'])
        runcmd(cmd, dbin, out='rdr.tsv', log='log_rdr')
        rdr = os.path.join(dbin, 'rdr.tsv')
        lcel = os.path.join(dbin, 'total.tsv')

    log('Combining RDRs and BAFs', level='PROGRESS')
    cmd = 'python2.7 {} -r {} -b {} -j {} -k {} -l {}'
    cmd = cmd.format(get_comp('Combiner.py'), rdr, baf, args['jobs'], args['blocksize'], lcel)
    if args['seed'] is not None:
        cmd += " --seed {}".format(args['seed'])
    if args['addgccorr']:
        cmd += " --gccorr {}".format(args['reference'])
    if not args['phasecorr']:
        cmd += " --nophasecorr"
    runcmd(cmd, dcom, out='combo.tsv')
    com = os.path.join(dcom, 'combo.tsv')

    log('Calling', level='PROGRESS')
    cmd = 'python2.7 {} {} -P {} -K {} -j {}'
    cmd = cmd.format(get_comp('Caller.py'), com, args['maxploidy'], args['upperk'], args['jobs'])
    if args['seed'] is not None:
        cmd += " --seed {}".format(args['seed'])
    runcmd(cmd, dcal, out='calls.tsv')
    calls = os.path.join(dcal, 'calls.tsv')

    log('Cloning', level='PROGRESS')
    cmd = 'python2.7 {} {}'
    cmd = cmd.format(get_comp('Cloner.py'), calls)
    if args['seed'] is not None:
        cmd += " --seed {}".format(args['seed'])
    if args['varbins'] or args['givenbins']:
        cmd += " --weighted"
    runcmd(cmd, dclo, out='mapping.tsv')
    mapping = os.path.join(dclo, 'mapping.tsv')

    log('Plotting', level='PROGRESS')
    os.chdir(dplo)
    up = (lambda f : os.path.join(os.pardir, f))
    cmd = 'python2.7 {} {} -m {}'
    cmd = cmd.format(os.path.join(src, 'Plotter.py'), up(calls), up(mapping))
    runcmd(cmd, './')
    os.chdir(os.pardir)


def setup(args):
    if any(os.path.isdir(os.path.join(args['rundir'], x)) for x in ['baf', 'rdr', 'combo', 'calls', 'clones', 'plots']):
        log('Some of the working folders already exist in the running directory and content will be overwritten, please interrupt the process if this was not intended.', level='WARN')

    dbaf = os.path.join(args['rundir'], 'baf')
    if not os.path.isdir(dbaf):
        os.mkdir(dbaf)

    drdr = os.path.join(args['rundir'], 'rdr')
    if not os.path.isdir(drdr):
        os.mkdir(drdr)

    dcom = os.path.join(args['rundir'], 'combo')
    if not os.path.isdir(dcom):
        os.mkdir(dcom)

    dcal = os.path.join(args['rundir'], 'calls')
    if not os.path.isdir(dcal):
        os.mkdir(dcal)

    dclo = os.path.join(args['rundir'], 'clones')
    if not os.path.isdir(dclo):
        os.mkdir(dclo)

    dplo = os.path.join(args['rundir'], 'plots')
    if not os.path.isdir(dplo):
        os.mkdir(dplo)

    return dbaf, drdr, dcom, dcal, dclo, dplo


if __name__ == '__main__':
    main()
