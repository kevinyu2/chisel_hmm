import os
import tempfile
import numpy as np
import pytest

from chisel.Binner import merge_bins_to_bbs, explain, subtract, get_thresholds, count_snps, nwindows
from chisel.RDREstimator import read_bins
from chisel.Cloner import binweights


def ids_of(reads, nsnps, msr, mspb, totals, mtr):
    return merge_bins_to_bbs(np.array(reads, dtype=float).reshape(-1, 1), np.array(nsnps), [msr], mspb,
                             np.array(totals, dtype=float).reshape(len(nsnps), -1), mtr)


def test_closes_at_first_window_meeting_all_criteria():
    # SNP reads reach 10 after 3 windows, total reads 150 after 2: the bin closes after 3
    ids, n = ids_of([5, 0, 5, 5, 5, 5], [1, 0, 1, 1, 1, 1], 10, 1, [100] * 6, [150])
    assert list(ids) == [0, 0, 0, 1, 1, 1] and n == 2


def test_short_trailing_run_is_merged_into_previous_bin():
    ids, n = ids_of([5, 0, 5, 5, 1], [1, 0, 1, 1, 1], 10, 1, [100] * 5, [150])
    assert list(ids) == [0] * 5 and n == 1


def test_region_that_never_meets_thresholds_is_one_bin():
    ids, n = ids_of([1, 1, 1], [1, 1, 1], 100, 1, [10] * 3, [0])
    assert list(ids) == [0, 0, 0] and n == 0


def test_every_criterion_and_column_is_required():
    # SNP reads are plenty, but the second column (normal) needs 3 windows
    ids, n = ids_of([50] * 6, [1] * 6, 10, 1, [[100, 10]] * 6, [100, 30])
    assert list(ids) == [0, 0, 0, 1, 1, 1]
    # min SNPs: windows without SNPs do not count
    ids, n = ids_of([50, 50, 50, 50], [1, 0, 1, 1], 0, 2, [1] * 4, [0])
    assert list(ids) == [0, 0, 0, 0]


def test_zero_thresholds_give_one_bin_per_window():
    ids, n = ids_of([0, 3, 0], [0, 1, 0], 0, 0, [0, 0, 0], [0])
    assert list(ids) == [0, 1, 2]


def test_explain_reports_deciding_criterion_and_tail():
    cum = (lambda x : np.concatenate([[0.0], np.cumsum(x)]))
    srd, tot = np.array([5., 0, 5, 5, 1]), np.array([100.] * 5)
    crit = [('snp_reads', cum(srd), 10), ('tumor_reads', cum(tot), 150)]
    assert explain(0, 3, crit)[:2] == ('met', 'snp_reads')
    assert explain(0, 5, crit)[:2] == ('met+tail', 'snp_reads')
    assert explain(3, 5, crit)[:2] == ('unmet', 'snp_reads')
    crit = [('snp_reads', cum(srd), 5), ('tumor_reads', cum(tot), 150)]
    assert explain(0, 2, crit)[:2] == ('met', 'tumor_reads')


def test_subtract_blacklist():
    assert subtract({'c': [(0, 100)]}, {'c': [(10, 20), (50, 60)]}) == {'c': [(0, 10), (20, 50), (60, 100)]}
    assert subtract({'c': [(0, 100)]}, {'c': [(0, 100)]}) == {'c': []}


def test_per_cell_thresholds_scale_with_cells():
    args = {'minsnpreads': 100, 'minsnpreadspercell': 25, 'mintotalreads': 5000, 'mintotalreadspercell': 0, 'minsnps': 1}
    t = get_thresholds(args, 40)
    assert t['snp'] == 1000 and t['tumor'] == 5000 and t['normal'] == 5000
    assert get_thresholds(args, 2)['snp'] == 100


def test_count_snps_windows_are_zero_based():
    with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False) as f:
        f.write('chr1\t1000\tAA\t1\t2\nchr1\t1001\tAA\t3\t0\nchr1\t1001\tCC\t1\t1\nchr1\t5\tGG\t9\t9\n')
    try:
        reads, count = count_snps(f.name, {'chr1': 3000}, ['AA', 'CC'], 1000)
    finally:
        os.remove(f.name)
    # POS 1000 (1-based) is base 999, in window 0; POS 1001 is in window 1; cell GG is not selected
    assert list(reads['chr1']) == [3, 5, 0] and list(count['chr1']) == [1, 1, 0]
    assert nwindows(3000, 1000) == 3 and nwindows(3001, 1000) == 4


def test_read_bins_skips_headers_and_checks_overlaps():
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write('##NCELLS=3\n#CHR\tSTART\tEND\tSIZE\nchr1\t100\t200\t100\nchr1\t0\t100\t100\nchr2\t0\t50\t50\nchr3\t0\t5\t5\n')
    try:
        bins = read_bins(f.name, ['chr1', 'chr2'])
        assert bins == {'chr1': [(0, 100), (100, 200)], 'chr2': [(0, 50)]}
        with open(f.name, 'a') as o:
            o.write('chr1\t150\t250\n')
        with pytest.raises(ValueError):
            read_bins(f.name, ['chr1'])
    finally:
        os.remove(f.name)


def test_cloner_weights_only_for_unequal_bins():
    assert binweights([('chr1', 0, 10), ('chr1', 10, 20)]) is None
    assert binweights([('chr1', 0, 10), ('chr1', 10, 40)]) == {('chr1', 0, 10): 10.0, ('chr1', 10, 40): 30.0}


UGP_BB = '#CHR\tSTART\tEND\t#SNPS\tregion_id\tswitchprobs\tfeature_id\n1\t0\t1000\t2\t0\t0.05\tintergenic\n1\t1000\t3000\t1\t0\t0.5\tintergenic\n'


def write_gz(text):
    import gzip
    f = tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False)
    f.close()
    with gzip.open(f.name, 'wb') as o:
        o.write(text.encode('utf-8'))
    return f.name


def test_read_bins_reads_ugp_gz_and_renames_chromosomes():
    f = write_gz(UGP_BB)
    try:
        assert read_bins(f, ['chr1']) == {'chr1': [(0, 1000), (1000, 3000)]}
    finally:
        os.remove(f)


def test_given_bins_keep_region_and_per_cell_snp_stats():
    from chisel.Binner import read_given, percell_snp_stats
    f = write_gz(UGP_BB)
    with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False) as b:
        # 1-based positions: 1000 is base 999 (bin 0), 1001 is base 1000 (bin 1); cell CC has nothing in bin 1
        b.write('chr1\t1000\tAA\t2\t1\nchr1\t1000\tCC\t1\t0\nchr1\t1001\tAA\t4\t4\nchr1\t2999\tBB\t1\t1\n')
    try:
        given = read_given(f, {'chr1': 2500})
        assert given == {'chr1': [(0, 1000, 'chr1_0'), (1000, 2500, 'chr1_0')]}  # END clipped to the chromosome
        bins = [{'CHR': 'chr1', 'START': s, 'END': e} for s, e, r in given['chr1']]
        percell_snp_stats(bins, b.name, {'chr1': 2500}, ['AA', 'BB', 'CC'])
        assert (bins[0]['NSNPS'], bins[0]['SNP_READS'], bins[0]['SNP_READS_MIN_CELL']) == (1, 4, 0)
        assert (bins[1]['NSNPS'], bins[1]['SNP_READS'], bins[1]['SNP_READS_MIN_CELL']) == (1, 8, 0)  # POS 2999 is beyond the clipped END
        assert bins[0]['SNP_READS_PER_CELL'] == '1.33' and bins[0]['SNP_READS_MEDIAN_CELL'] == '1.0'
    finally:
        os.remove(f)
        os.remove(b.name)


def test_summary_accepts_text_region_ids():
    # UGP's real bb.tsv.gz has region_id chr1p / chr1q (regression: the summary assumed numbers)
    from chisel.Binner import summarize
    bins = [{'CHR': 'chr1', 'START': s, 'END': e, 'SIZE': e - s, 'REGION': r, 'SNP_READS': 10, 'SNP_READS_MIN_CELL': 1,
             'CLOSE': 'given', 'LIMIT': 'given'} for s, e, r in [(200, 300, 'chr1_chr1q'), (0, 100, 'chr1_chr1p')]]
    summarize(bins, 1)
