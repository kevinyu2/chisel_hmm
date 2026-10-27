#!/usr/bin/env python2.7

import sys, os
import argparse
import shutil
import warnings

from itertools import cycle
from collections import defaultdict
from collections import Counter

import numpy as np
import scipy
import scipy.cluster
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance

from Utils import *


def parse_args(args):
    description = "Infer clones as subpopulations of cells with the same complement of CNAs and outputs a file with the mapping of every cell to the corresponding clone."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("INPUT", type=str, help="Input file with RDR, BAF, and inferred copy numbers.")
    parser.add_argument("-f", "--maxdiff", required=False, type=float, default=0.07, help="Maximum fraction of the genome with different copy-number states allowed in a clone (default: 0.07, when -1 is chosen the maximum cluster method of SciPy is used)")
    parser.add_argument("-r", "--refinement", required=False, type=float, default=0.15, help="Maximum difference to assign noisy cells to a clone (default: 0.15)")
    parser.add_argument("-s", "--minsize", required=False, type=int, default=14, help="Minimum size of subpopultation to define a clone (default: 14)")
    parser.add_argument("-l", "--linkage", required=False, type=str, default='weighted', help="Linkage method to use for the hierarchical clustering (default: weighted, it must be a valid linkage method available in SciPy when using a non-euclidean distance, i.e. 'single', 'complete', 'average', 'weighted')")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: none)")
    parser.add_argument("--weighted", required=False, default=False, action='store_true', help="Weight every bin by its length when measuring differences between cells, so that --maxdiff and --refinement are fractions of the genome also with variable-length bins (default: disabled, every bin has the same weight)")
    parser.add_argument("--clustering", required=False, type=str, default='hamming', choices=['hamming', 'divisive'], help="How cells are grouped: hamming = CHISEL's hierarchical clustering of the called copy-number states (-f, -l, -r); divisive = top-down splitting on the cells' RDR and BAF with permutation-tested splits (see CellClusterer.py; -s is the minimum clone size) (default: hamming)")
    dv = parser.add_argument_group('divisive clustering (--clustering divisive)')
    dv.add_argument("--rdrweight", type=float, default=0.5, help="Weight of RDR vs BAF (1 - w) in the cell distances (default: 0.5)")
    dv.add_argument("--noisescale", default=False, action='store_true', help="Express RDR and BAF in units of their expected counting noise (default: disabled)")
    dv.add_argument("--perm", type=int, default=20, help="Permutations per split test (default: 20)")
    dv.add_argument("--permq", type=float, default=95.0, help="Noise floor = this percentile of the permuted top PC variance (default: 95)")
    dv.add_argument("--floormargin", type=float, default=1.5, help="A signal PC must exceed the noise floor by this factor (default: 1.5)")
    dv.add_argument("--ashman", type=float, default=2.0, help="Minimum Ashman's D between the two halves of a split (default: 2)")
    dv.add_argument("--peel", type=int, default=5, help="Times to set aside small outlier groups and retry a split (default: 5)")
    dv.add_argument("--maxclones", type=int, default=30, help="Stop splitting at this many clusters (default: 30)")
    dv.add_argument("--noreassign", default=False, action='store_true', help="Keep the split tree's assignment, no final reassignment (default: disabled)")
    dv.add_argument("--outlierz", type=float, default=5.0, help="Set apart cells whose noise level is more than this many robust SDs above the typical cell (0 = off) (default: 5)")
    dv.add_argument("--assignoutliers", default=False, action='store_true', help="Give the noisy cells set apart the nearest clone (default: disabled, they keep their own calls, clone None)")
    args = parser.parse_args(args)

    if not os.path.isfile(args.INPUT):
        raise ValueError('ERROR: input file does not exist!')
    if (not 0.0 <= args.maxdiff <= 1.0) and args.maxdiff != -1:
        raise ValueError('ERROR: the maximum different fraction of the genome must be either within [0, 1] or equal to -1!')
    if args.refinement is None:
        args.refinement = args.maxdiff
    if not 0.0 <= args.refinement <= 1.0:
        raise ValueError('ERROR: the refinement must be either within [0, 1]!')
    if args.minsize <= 0:
        raise ValueError('ERROR: the minimum size of subpopulations must be positive!')
    if not args.linkage in {'single', 'complete', 'average', 'weighted'}:
        raise ValueError('ERROR: the linkage method is invalid or not available for non-euclidean distances!')
    if args.clustering == 'divisive':
        if not 0.0 <= args.rdrweight <= 1.0:
            raise ValueError('ERROR: --rdrweight must be within [0, 1]!')
        if args.perm < 1 or not 0 < args.permq <= 100 or args.floormargin <= 0 or args.maxclones < 1:
            raise ValueError('ERROR: invalid divisive clustering parameters!')
    if args.seed and args.seed < 0:
        raise ValueError("Random seed must be positive or zero!")
    else:
        np.random.seed(args.seed)

    return {
        'input' : args.INPUT,
        'maxdiff' : args.maxdiff,
        'refinement' : args.refinement,
        'minsize' : args.minsize,
        'linkage' : args.linkage,
        'seed' : args.seed,
        'weighted' : args.weighted,
        'clustering' : args.clustering,
        'divisive' : {'rdr_weight' : args.rdrweight, 'noise_scale' : args.noisescale, 'n_perm' : args.perm, 'q' : args.permq,
                      'margin' : args.floormargin, 'ashman' : args.ashman, 'peel' : args.peel,
                      'max_clusters' : args.maxclones, 'reassign' : not args.noreassign,
                      'outlier_z' : args.outlierz, 'assign_outliers' : args.assignoutliers}
    }


def main(args=None, stdout_file=None):
    log('Parsing and checking arguments')
    args = parse_args(args)
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]), level='INFO')

    log('Reading input')
    cns, pos, cells = reading(args['input'])

    log('Clustering cells in clones')
    weights = binweights(pos) if args['weighted'] else None
    if args['clustering'] == 'divisive':
        clus = divisive_clustering(args['input'], cells, args['minsize'], args['seed'], args['divisive'])
    else:
        clus = clustering(cns, pos, cells, args['maxdiff'], args['linkage'], weights)

    log('Selecting clones')
    clones = selecting(clus, args['minsize'])
    log('Number of identified clones: {}'.format(len(set(clones.values()))), level='INFO')

    if len(clones) > 0 and args['refinement'] >= 0.0 and args['clustering'] != 'divisive':
        log('Refining clustering')
        clones, clus = refining(cns, clus, clones, args['refinement'], weights)
    log('Number of discarded cells: {} over {} in total'.format(len(set(cells) - set(clones.keys())), len(set(cells))), level='INFO')

    log('Profiling clones')
    profiles = profiling(cns, clus)
    
    log('Writing clone map')
    header = '\t'.join(['#CELL', 'CLUSTER', 'CLONE'])
    if stdout_file is not None:
        with open(stdout_file, 'w') as f:
            f.write(header + '\n')
            for c in cells:
                f.write('\t'.join(map(str, [c, clus[c], 'Clone{}'.format(clones[c]) if c in clones else 'None'])) + '\n')
    else:
        print header
        for c in cells:
            print '\t'.join(map(str, [c, clus[c], 'Clone{}'.format(clones[c]) if c in clones else 'None']))

    log('Writing clone-corrected copy numbers in provided input')
    ftmp = args['input'] + '_TMP'
    assert not os.path.isfile(ftmp), "Temporary file {} does already exist!".format(ftmp)
    form = (lambda p : ((p[0], int(p[1]), int(p[2])), p[3], p[0:12]))
    with open(args['input'], 'r') as i:
        with open(ftmp, 'w') as o:
            for l in i:
                if '#' != l[0]:
                    b, e, val = form(l.strip().split())
                    o.write('\t'.join(val + ['{}|{}'.format(*profiles[b][clus[e]])]) + '\n')
                else:
                    o.write('\t'.join(['#CHR', 'START', 'END', 'CELL', 'NORM_COUNT', 'COUNT', 'RDR', 'A_COUNT', 'B_COUNT', 'BAF', 'CLUSTER', 'HAP_CN', 'CORRECTED_HAP_CN']) + '\n')
    shutil.move(ftmp, args['input'])


def reading(f):
    cns = defaultdict(lambda : dict())
    form = (lambda p : ((p[0], int(p[1]), int(p[2])), p[3], tuple(map(int, p[11].split('|')))))
    with open(f, 'r') as i:
        for l in i:
            if l[0] != '#' and len(l) > 1:
                b, c, cn = form(l.strip().split())
                assert c not in cns[b] # and c not in stuff[b]
                cns[b][c] = cn
    cns = dict(cns)
    orderchrs = (lambda x : int(''.join([l for l in x if l.isdigit()])))
    order = (lambda b : (orderchrs(b[0]), int(b[1]), int(b[2])))
    pos = sorted(cns.keys(), key=order)
    cells = sorted(set(c for b in cns for c in cns[b]))
    return cns, pos, cells


def binweights(pos):
    w = {b : float(b[2] - b[1]) for b in pos}
    if len(set(w.values())) == 1 or min(w.values()) <= 0:
        return None # equal lengths are the unweighted case, which is used to keep results identical
    return w


def clustering(cns, pos, cells, maxdiff, linkage, weights=None):
    states = {s : x for x, s in enumerate(set(cns[b][c] for b in pos for c in cells))}
    data = [[states[cns[b][c]] for b in pos] for c in cells]
    if weights is None:
        linkage = hier.linkage(data, method=linkage, metric='hamming', optimal_ordering=True)
    else:
        dist = scipy.spatial.distance.pdist(np.array(data), metric='hamming', w=np.array([weights[b] for b in pos]))
        linkage = hier.linkage(dist, method=linkage, optimal_ordering=True)
    if maxdiff != -1:
        clus = hier.fcluster(linkage, t=maxdiff, criterion='distance')
    else:
        clus = hier.fcluster(linkage, t=len(cells), criterion='maxclust')
    return {e : clus[i] for i, e in enumerate(cells)}


def divisive_clustering(path, cells, minsize, seed, opts):
    """Cells grouped top-down on their RDR and BAF (CellClusterer.divisive) -> {cell : cluster id from 1}."""
    from CellClusterer import read_features, feature_matrix, divisive
    fcells, bins, arr = read_features(path)
    assert list(fcells) == list(cells), 'cells of the features and of the calls differ'
    X = feature_matrix(arr, rdr_weight=opts['rdr_weight'], noise_scale=opts['noise_scale'])
    log('Divisive clustering of {} cells on {} features ({} bins x RDR and BAF)'.format(X.shape[0], X.shape[1], len(bins)), level='INFO')
    labels, steps = divisive(X, np.random.RandomState(seed if seed is not None else 0), min_size=minsize,
                             n_perm=opts['n_perm'], q=opts['q'], margin=opts['margin'], ashman=opts['ashman'],
                             peel=opts['peel'], max_clusters=opts['max_clusters'], reassign=opts['reassign'],
                             outlier_z=opts['outlier_z'])
    log('Divisive splits:\n' + '\n'.join(steps), level='INFO')
    noisy = np.flatnonzero(labels < 0)
    if len(noisy) and opts['assign_outliers'] and (labels >= 0).any():
        cen = np.array([X[labels == k].mean(0) for k in range(labels.max() + 1)])
        labels[noisy] = ((X[noisy][:, None, :] - cen[None, :, :]) ** 2).sum(2).argmin(1)
        log('{} noisy cells given the nearest clone'.format(len(noisy)), level='INFO')
    clus, nxt = {}, labels.max() + 2
    for i, e in enumerate(fcells):
        if labels[i] >= 0:
            clus[e] = int(labels[i]) + 1
        else:                                            # noisy cell: a cluster of its own (clone None, own calls)
            clus[e] = nxt
            nxt += 1
    return clus


def selecting(clus, minsize):
    size = {i : sum(clus[c] == i for c in clus) for i in set(clus.values())}
    return {c : clus[c] for c in clus if size[clus[c]] >= minsize}


def refining(cns, clus, chosen, maxdiff, weights=None):
    clones = set(chosen.values())
    safeargmax = (lambda C : argmax(C) if len(C) > 0 else (1, 1))
    getcn = (lambda g, i : safeargmax(Counter([cns[g][c] for c in chosen if chosen[c] == i])))
    profile = {g : {i : getcn(g, i) for i in clones} for g in cns}
    diff = (lambda i, c, g : 1 if profile[g][i] != cns[g][c] else 0)
    if weights is None:
        weight = (lambda i, c : float(sum(diff(i, c, g) for g in profile)) / float(len(profile)))
    else:
        totw = float(sum(weights[g] for g in profile))
        weight = (lambda i, c : sum(weights[g] * diff(i, c, g) for g in profile) / totw)
    closest = (lambda c : min([(i, weight(i, c)) for i in clones], key=(lambda x : x[1])))
    ref = {c : closest(c) for c in clus if c not in chosen.keys()}
    newclones = {c : chosen[c] if c in chosen else ref[c][0] for c in clus if c in chosen or ref[c][1] <= maxdiff}
    newclus = {c : newclones[c] if c in newclones else clus[c] for c in clus}
    assert False not in set(len({clus[c], chosen[c], newclus[c], newclones[c]}) == 1 for c in chosen)
    return newclones, newclus


def profiling(cns, clus):
    clones = set(clus.values())
    # safeargmax = (lambda C : argmax(C) if len(C) > 0 else (1, 1))
    # getcn = (lambda g, i : safeargmax(Counter([cns[g][c] for c in clus if clus[c] == i])))
    mapclo = {i : filter(lambda e : clus[e] == i, clus.keys()) for i in clones}
    assert all(len(mapclo[i]) > 0 for i in mapclo), 'Found cluster assignment with no corresponding cell'
    getcn = (lambda g, i : argmax(Counter([cns[g][e] for e in mapclo[i]])))
    return {g : {i : getcn(g, i) for i in clones} for g in cns}    


if __name__ == '__main__':
    main()
