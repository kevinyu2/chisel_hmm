# Variable-length bins

`chisel --varbins` replaces CHISEL's fixed-size bins with bins that are as long as needed to collect a fixed amount of allelic signal. It follows the window-merge binning of the [Universal-Genotyping-Pipeline](https://github.com/raphael-group/Universal-Genotyping-Pipeline) (`build_adaptive_bins`), which succeeded HATCHet2's adaptive binning (Myers et al. 2023). In SNP-dense regions bins are short, and in SNP-poor regions they are long.

## Two ways to get the bins

- `chisel --bins FILE` uses precomputed bins, e.g. UGP's `bb/MSR<n>/bulk/bb.tsv.gz` (UGP per-cell mode) or any TSV/BED with columns CHR START END (0-based, half-open; gzip is fine; `chr` prefixes are matched to the BAM). CHISEL describes them in `bins/` (bins.tsv with `CLOSE`/`LIMIT` = `given`, bins.pdf, windows.tsv.gz) and counts reads per cell on their exact coordinates.
- `chisel --varbins` computes the bins within CHISEL, as described below.

Both write `bins.tsv` with `SNP_READS_PER_CELL` (mean over cells), `SNP_READS_MEDIAN_CELL` and `SNP_READS_MIN_CELL` (least-covered selected cell), so bins made by UGP's every-cell rule and by CHISEL's mean rule can be compared directly.

## How bins are made (`--varbins`)

The binning runs after the BAF step (`Binner.py`).

1. **Regions.** These are chromosome arms, from the centromeres of hg19 or hg38. The version is detected from the length of chr1. You can override them with `--regions BED`. Intervals in `--blacklist BED` are removed. Bins never span two regions.
2. **Windows.** Every region is tiled into windows of `--windowsize` (default 1 kb). For each window, the binner counts:
   - read starts of the selected cells, summed into a pseudobulk (`samtools view | awk`, one pass per chromosome);
   - read starts of the matched normal;
   - SNP-covering reads and heterozygous SNPs of the pseudobulk, from `baf.tsv`.

   Windows without normal reads are trimmed from the ends of every region, which removes assembly gaps and telomeres.
3. **Greedy merge.** Consecutive windows are merged until the bin meets every criterion, all measured on the pseudobulk:

   | criterion | threshold |
   |---|---|
   | SNP-covering reads | max(`--minsnpreads`, `--minsnpreadspercell` x #cells), default 25 per cell |
   | heterozygous SNPs | `--minsnps`, default 1 |
   | tumor read starts | max(`--mintotalreads`, `--mintotalreadspercell` x #cells), default 5000 |
   | normal read starts | `--mintotalreads`, default 5000 |

   A trailing run at the end of a region that misses the criteria is merged into the previous bin. A region that never meets them is a single bin. Bins without SNPs are dropped. The merge gives exactly the same bins as UGP's `_merge_bins_to_bbs` (tested in `tests/ugp_equivalence`).
4. **Per-cell counting.** `RDREstimator.py --bins` counts reads per cell in the new bins. Combiner, Caller and the rest of the pipeline run unchanged. Cloner runs with `--weighted`, so differences between cells are measured as fractions of the genome rather than of the bins.

Because the thresholds are on the pseudobulk of the cells, the per-cell thresholds bound the **mean** count per cell. They do not bound the count of every cell.

## Outputs (`bins/`)

| file | content |
|---|---|
| `bins.tsv` | one row per bin; `##` lines hold the thresholds used. `CLOSE` = `met` (closed when all criteria were met), `met+tail` (also absorbed a short trailing run), `unmet` (the region never met them). `LIMIT` = the criterion met last, which decided the boundary |
| `bins.pdf` | summary page, then per chromosome: SNP density; SNP reads per cell accumulating in each bin against the threshold; bin sizes coloured by `LIMIT`; pseudobulk RDR |
| `windows.tsv.gz` | every window with its counts and bin |
| `rdr.tsv`, `total.tsv` | per-cell RDRs on the bins (input of the combining step) |
| `log_binning` | summary and a per-region table of bins |

## Standalone

```
chisel_bins -t cells.bam -n normal.bam -r ref.fa -l rdr/total.tsv -b baf/baf.tsv -c "chr1 chr2" --windows windows.tsv.gz > bins.tsv
chisel_binsplot bins.tsv windows.tsv.gz -o bins.pdf
chisel_rdr ... or: python2.7 RDREstimator.py -t cells.bam -n normal.bam -r ref.fa -l rdr/total.tsv --bins bins.tsv > rdr.tsv
```

## Notes

- The SNP criterion counts reads of the tumor cells, so regions with fewer copies need more genome to reach it, and losses get longer bins.
- `--addgccorr` regresses raw bin counts on GC content, which mixes in bin length when bins vary in size.
- Plots of `chisel_plotting` use the bin index as x-axis, so SNP-dense regions look wider.
