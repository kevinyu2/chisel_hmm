import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src', 'chisel', 'CellBinPlotter.py')


@pytest.fixture
def calls(tmpdir):
    rng = np.random.RandomState(1)
    p = tmpdir.join('calls.tsv')
    with open(str(p), 'w') as f:
        f.write('#CHR\tSTART\tEND\tCELL\tNORM_COUNT\tCOUNT\tRDR\tA_COUNT\tB_COUNT\tBAF\tCLUSTER\tCN_STATE\n')
        for s, e in ((0, 400000), (400000, 800000)):
            for c in range(50):
                n = rng.poisson(10)
                b = rng.binomial(n, 0.2)
                f.write('\t'.join(map(str, ['chr1', s, e, 'c%d' % c, 60, 50, 0.8, n - b, b, (float(b) / n if n else 0.5), 0, '2|0'])) + '\n')
    return str(p)


def run(*args):
    p = subprocess.Popen([sys.executable, SCRIPT] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    return type('R', (), {'returncode': p.returncode, 'stdout': out, 'stderr': err})


def test_table_for_one_bin(calls, tmpdir):
    out = str(tmpdir.join('o'))
    r = run(calls, '--at', 'chr1:500000', '-o', out, '--nofig')
    assert r.returncode == 0, r.stderr
    lines = open(out + '.tsv').read().strip().split('\n')
    assert len(lines) == 51 and lines[1].split('\t')[1:3] == ['400000', '800000']


def test_overlapping_two_bins_is_an_error(calls, tmpdir):
    r = run(calls, '--bin', 'chr1:100000-600000', '-o', str(tmpdir.join('o')), '--nofig')
    assert r.returncode != 0 and 'overlaps 2 bins' in r.stderr
