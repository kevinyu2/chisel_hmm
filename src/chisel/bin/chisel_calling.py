#!/usr/bin/env python2.7

import os
import argparse
import subprocess as sp
import multiprocessing as mp
import shlex
import datetime
import re

import chisel

src = os.path.dirname(chisel.__file__)
from ..Utils import *
from ..HMMClusterizer import add_driver_arguments, driver_options, caller_flags


def parse_args():
    description = "CHISEL command to re-run the inference of allele- and haplotype-specific copy numbers, cell clustering, and plotting. This steps starts from estimated RDRs and BAFs."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("INPUT", nargs='?', default='combo/combo.tsv', type=str, help="Input file with combined RDR and BAF per bin and per cell (default: combo/combo.tsv)")
    parser.add_argument("-x","--rundir", required=False, default='./', type=str, help="Running directory (default: current directory)")
    parser.add_argument("-A","--sensitivity", required=False, type=float, default=1.0, help="Sensitivity of model selection for ploidy (default: 1, increase this parameter to lower sensitivity to noisy data, adjust this value (e.g. 2, 4, ..., 10, ...) to better deal with high-variance data (e.g. low coverage, small number of cells, low number of phased SNPs, etc...)")
    parser.add_argument("-P","--maxploidy", required=False, type=int, default=4, help="Maximum total copy number to consider for balanced cluster (default: 4, corresponding to a WGD)")
    parser.add_argument("-K","--upperk", required=False, type=int, default=100, help="Maximum number of bin clusters (default: 100, use 0 to consider maximum number of clusters)")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: None)")
    parser.add_argument("-j","--jobs", required=False, type=int, default=0, help="Number of parallele jobs to use (default: equal to number of available processors)")
    parser.add_argument("--clusters", required=False, type=str, default=None, help="Take the bin clusters from this file (CHR START END CLUSTER) instead of computing them, e.g. oracle clusters for diagnosis (default: none)")
    parser.add_argument("--weighted", required=False, default=False, action='store_true', help="Length-weighted distance between cells when cloning, as chisel uses with variable-length bins (--varbins/--bins) (default: disabled)")
    add_driver_arguments(parser)
    parser.add_argument("--cloning", required=False, type=str, default='divisive', choices=['divisive', 'hamming'], help="Grouping of cells into clones: divisive = top-down splitting on RDR and BAF with permutation-tested splits, computed before calling (calls/cellgroups.tsv); hamming = CHISEL's original clustering of the called states, after calling (implies --nopseudobulk) (default: divisive)")
    parser.add_argument("--nopseudobulk", required=False, default=False, action='store_true', help="With divisive cloning: cluster bins (--hmm) on every cell instead of the cell groups' pooled reads, and correct copy numbers by majority vote of each clone's cells instead of calling its pooled reads (default: pseudobulk)")
    args = parser.parse_args()
    hmmopts = driver_options(args)
    if args.clusters is not None:
        if not os.path.isfile(args.clusters):
            raise ValueError("Clusters file does not exist: {}".format(args.clusters))
        if hmmopts:
            raise ValueError("Use either --clusters or --hmm")

    if not os.path.isfile(args.INPUT):
        raise ValueError("Input file does not exist: {}".format(args.INPUT))
    if not os.path.isdir(args.rundir):
        raise ValueError("Running directory does not exists: {}".format(args.rundir))
    if args.seed and args.seed < 1:
        raise ValueError("The random seed  must be positive!")
    if args.maxploidy < 3:
        raise ValueError("The maximum total copy number to consider for balanced cluster must be at least 2!")
    if args.upperk < 1:
        raise ValueError("The maximum number of clusters must be positive!")
    if not args.jobs:
        args.jobs = mp.cpu_count()
    if args.jobs < 1:
        raise ValueError("The number of jobs must be positive!")

    return {
        "INPUT" : args.INPUT,
        "rundir" : args.rundir,
        "sensitivity" : args.sensitivity,
        "maxploidy" : args.maxploidy,
        "upperk" : args.upperk,
        "seed" : args.seed,
        "jobs" : args.jobs,
        "weighted" : args.weighted,
        "hmmopts" : hmmopts,
        "cloning" : args.cloning,
        "pseudobulk" : args.cloning == 'divisive' and not args.nopseudobulk,
        "minsize" : 14,
        "clusters" : os.path.abspath(args.clusters) if args.clusters else None
    }


def main():
    log('Parsing and checking arguments', level='PROGRESS')
    args = parse_args()
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]) + '\n', level='INFO')

    log('Setting directories', level='PROGRESS')
    dcal, dclo, dplo = setup(args, force=False)
    def get_comp(name):
        comp = os.path.join(src, name)
        if not os.path.isfile(comp):
            raise ValueError("{} not found in src directory of bin i.e. {}, is anything been moved?".format(name, src))
        return comp


    groups = None
    if args['cloning'] == 'divisive':
        log('Grouping cells (divisive clustering on RDR and BAF)', level='PROGRESS')
        cmd = 'python2.7 {} {} -s {}'.format(get_comp('CellClusterer.py'), args['INPUT'], args['minsize'])
        if args['seed'] is not None:
            cmd += " --seed {}".format(args['seed'])
        runcmd(cmd, dcal, out='cellgroups.tsv', log='cellgroups.log')
        groups = os.path.join(dcal, 'cellgroups.tsv')
        if args['hmmopts'] and args['pseudobulk'] and not args['hmmopts'].get('hmmgroups'):
            args['hmmopts']['hmmgroups'] = os.path.abspath(groups)
            log('HMM bin clustering on the pooled reads of the cell groups', level='INFO')

    log('Calling', level='PROGRESS')
    cmd = 'python2.7 {} {} -A {} -P {} -K {} -j {}'
    cmd = cmd.format(get_comp('Caller.py'), args['INPUT'], args['sensitivity'], args['maxploidy'], args['upperk'], args['jobs'])
    if args['seed'] is not None:
        cmd += " --seed {}".format(args['seed'])
    cmd += caller_flags(args['hmmopts'], os.path.join(dcal, 'hmm_selection.tsv'))
    if args['clusters']:
        cmd += " --clusters {}".format(args['clusters'])
    runcmd(cmd, dcal, out='calls.tsv')
    calls = os.path.join(dcal, 'calls.tsv')

    log('Cloning', level='PROGRESS')
    cmd = 'python2.7 {} {}'
    cmd = cmd.format(get_comp('Cloner.py'), calls)
    if args['seed'] is not None:
        cmd += " --seed {}".format(args['seed'])
    if args['weighted']:
        cmd += " --weighted"
    cmd += " --clustering {} -P {}".format(args['cloning'], args['maxploidy'])
    if groups is not None:
        cmd += " --groups {}".format(os.path.abspath(groups))
    if not args['pseudobulk']:
        cmd += " --nopseudobulk"
    runcmd(cmd, dclo, out='mapping.tsv')
    mapping = os.path.join(dclo, 'mapping.tsv')

    log('Plotting', level='PROGRESS')
    os.chdir(dplo)
    up = (lambda f : os.path.join(os.pardir, f))
    cmd = 'python2.7 {} {} -m {}'
    cmd = cmd.format(os.path.join(src, 'Plotter.py'), up(calls), up(mapping))
    runcmd(cmd, './')
    os.chdir(os.pardir)


def setup(args, force=True):
    dcal = os.path.join(args['rundir'], 'calls')
    if os.path.isdir(dcal):
        log("The calls sub-directory in the running directory already exists, results will be overwritten!", level='WARN')
    else:
        os.mkdir(dcal)

    dclo = os.path.join(args['rundir'], 'clones')
    if os.path.isdir(dclo):
        log("The clones sub-directory in the running directory already exists, results will be overwritten!", level='WARN')
    else:
        os.mkdir(dclo)

    dplo = os.path.join(args['rundir'], 'plots')
    if os.path.isdir(dplo):
        log("The plots sub-directory in the running directory already exists, results will be overwritten!\n", level='WARN')
    else:
        os.mkdir(dplo)

    return dcal, dclo, dplo


if __name__ == '__main__':
    main()
