"""
Equivalence of CHISEL's Binner.merge_bins_to_bbs with the reference kernel of the
Universal-Genotyping-Pipeline (segmentation_utils._merge_bins_to_bbs).

Python 3 only (UGP needs numba). Run with
    UGP_DIR=/path/to/Universal-Genotyping-Pipeline python3 -m pytest tests/ugp_equivalence
"""
import os, sys, importlib.util
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
UGP_DIR = os.environ.get('UGP_DIR', os.path.join(HERE, '..', '..', '..', 'ugp'))
UGP_FILE = os.path.join(UGP_DIR, 'workflow', 'scripts', 'script_utils', 'segmentation_utils.py')
if not os.path.isfile(UGP_FILE):
    pytest.skip('UGP not found, set UGP_DIR', allow_module_level=True)

spec = importlib.util.spec_from_file_location('ugp_seg', UGP_FILE)
ugp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ugp)
import builtins
builtins.unichr = chr  # CHISEL's Utils is Python 2 code
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
from chisel.Binner import merge_bins_to_bbs


def run_both(reads, nsnps, msr, mspb, totals, mtr, require_snp):
    B = len(nsnps)
    ref_ids, ref_n = ugp._merge_bins_to_bbs(
        np.ascontiguousarray(reads, dtype=np.float64), np.ascontiguousarray(nsnps, dtype=np.int64),
        np.ascontiguousarray(msr, dtype=np.float64), int(mspb),
        np.ascontiguousarray(totals, dtype=np.float64), float(mtr), bool(require_snp),
        np.arange(B, dtype=np.int64))
    ids, n = merge_bins_to_bbs(reads, nsnps, msr, mspb, totals, [mtr] * totals.shape[1], require_snp)
    return (ref_ids, ref_n), (ids, n)


@pytest.mark.parametrize('seed', range(3000))
def test_random(seed):
    rng = np.random.RandomState(seed)
    B = rng.randint(0, 300)
    M, Mall = rng.randint(1, 4), rng.randint(1, 5)
    sparsity = rng.uniform(0, 0.95)  # many empty windows, like SNP-free 1kb windows
    reads = rng.poisson(rng.uniform(0, 30), size=(B, M)) * (rng.uniform(size=(B, 1)) > sparsity)
    nsnps = (reads.sum(axis=1) > 0) * rng.randint(1, 4, size=B)
    totals = rng.poisson(rng.uniform(0, 100), size=(B, Mall))
    msr = rng.choice([0, 1, 50, 200, 1000], size=M).astype(float)
    mspb = rng.choice([0, 1, 3, 10])
    mtr = float(rng.choice([0, 10, 500, 5000, 10**7]))
    require_snp = bool(rng.randint(2))
    (rid, rn), (cid, cn) = run_both(reads, nsnps, msr, mspb, totals, mtr, require_snp)
    assert rn == cn
    np.testing.assert_array_equal(rid, cid)


@pytest.mark.parametrize('B', [0, 1, 2, 3])
def test_tiny(B):
    for thr in [0, 1, 5, 1000]:
        reads = np.ones((B, 1)) * 2
        (rid, rn), (cid, cn) = run_both(reads, np.ones(B), [thr], 1, np.ones((B, 2)) * 3, thr, True)
        assert rn == cn
        np.testing.assert_array_equal(rid, cid)
