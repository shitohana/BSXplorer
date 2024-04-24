# BSXplorer

Analytical framework for BS-seq data comparison and visualization.

For Python API reference manual and tutorials visit: https://shitohana.github.io/BSXplorer.

# How to cite

Yuditskiy, K., Bezdvornykh, I., Kazantseva, A. et al. BSXplorer: analytical framework for exploratory analysis of BS-seq data. BMC Bioinformatics 25, 96 (2024). https://doi.org/10.1186/s12859-024-05722-9

# Installation

```commandline
pip install bsxplorer
```

# Console usage

- [bsxplorer-metagene](#bsxplorer-metagene) - Create Metagene report
- [bsxplorer-category](#bsxplorer-category) - Categorize regions by P_cg metric and render HTML-report
- [bsxplorer-chr](#bsxplorer-chr) - Visualize chromosome methylation levels and render HTML-report

## Config file

All console scripts **require configuration file** – tab separated file with columns:

Column **NAMES** should **NOT** be **INCLUDED** in real configuration file.

| <ins>Group name</ins> | <ins>Path to report</ins> | <ins>Path to annotation</ins> | Flank length | Min region length | Region type | Genome type | Report type |
|-----------------------|---------------------------|-------------------------------|--------------|-------------------|-------------|-------------|-------------|
| Mock                  | mock-1.CX_report.txt      | annotation.gff                | 2000         | 0                 | gene        | gff         | bismark     |
| Mock                  | mock-2.CX_report.txt      | annotation.gff                | 2000         | 0                 | gene        | gff         | bismark     |
| Infected              | infected-1.CX_report.txt  | annotation.gff                | 2000         | 0                 | gene        | gff         | bismark     |
| Infected              | infected-2.CX_report.txt  | annotation.gff                | 2000         | 0                 | gene        | gff         | bismark     |

Columns with <ins>underlined</ins> are required.

Currently only `gff` genome_type and only `bismark` report_type are supported in colsole version of BSXplorer, although you still can read various formats from Python API.

## bsxplorer-metagene

```commandline
bsxplorer-metagene --help
                                                                                   
usage: BSXplorer [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t] [-s {wmean,mean,median,min,max,1pgeom}] [-u UBIN] [-d DBIN] [-b BBIN] [-q QUANTILE] [-C CONFIDENCE] [-S SMOOTH] [-H VRESOLUTION] [-V HRESOLUTION]
                 [--separate_strands] [--export {pdf,svg,none}] [--ticks TICKS TICKS TICKS TICKS TICKS]
                 config

Metagene report creation tool

positional arguments:
  config                Path to config file

optional arguments:
  -h, --help            show this help message and exit
  -o NAME, --out NAME   Output filename (default: Metagene_Report_12-02-24_10-48-20)
  --dir DIR             Output and working dir (default: $CWD)
  -m BLOCK_MB, --block_mb BLOCK_MB
                        Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.) (default: 50)
  -t, --threads         Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()` (default: False)
  -s {wmean,mean,median,min,max,1pgeom}, --sumfunc {wmean,mean,median,min,max,1pgeom}
                        Summary function to calculate density for bin with. (default: wmean)
  -u UBIN, --ubin UBIN  Number of windows for upstream region (default: 50)
  -d DBIN, --dbin DBIN  Number of windows for downstream downstream (default: 50)
  -b BBIN, --bbin BBIN  Number of windows for body region (default: 100)
  -q QUANTILE, --quantile QUANTILE
                        Quantile of most varying genes to draw on clustermap (default: 0.75)
  -C CONFIDENCE, --confidence CONFIDENCE
                        Probability for confidence bands for line-plot. 0 if disabled (default: 0.95)
  -S SMOOTH, --smooth SMOOTH
                        Windows for SavGol function. (default: 10)
  -H VRESOLUTION        Vertical resolution for heat-map (default: 100)
  -V HRESOLUTION        Vertical resolution for heat-map (default: 100)
  --separate_strands    Do strands need to be processed separately (default: False)
  --export {pdf,svg,none}
                        Export format for plots (set none to disable) (default: pdf)
  --ticks TICKS TICKS TICKS TICKS TICKS
                        Names of ticks (5 labels with ; separator in " brackets) (default: None)
```

## bsxplorer-category

```commandline
bsxplorer-category --help
usage: BSXplorer-Categorise [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t] [-s {wmean,mean,median,min,max,1pgeom}] [-u UBIN] [-d DBIN] [-b BBIN] [-q QUANTILE] [-C CONFIDENCE] [-S SMOOTH] [-H VRESOLUTION] [-V HRESOLUTION]
                            [--separate_strands] [--export {pdf,svg,none}] [--ticks TICKS TICKS TICKS TICKS TICKS] [--cytosine_p CYTOSINE_P] [--min_cov MIN_COV] [--region_p REGION_P] [--save_cat | --no-save_cat]
                            config

BM, UM categorisation tool

positional arguments:
  config                Path to config file

optional arguments:
  -h, --help            show this help message and exit
  -o NAME, --out NAME   Output filename (default: Metagene_Report_12-02-24_10-51-50)
  --dir DIR             Output and working dir (default: $CWD)
  -m BLOCK_MB, --block_mb BLOCK_MB
                        Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.) (default: 50)
  -t, --threads         Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()` (default: False)
  -s {wmean,mean,median,min,max,1pgeom}, --sumfunc {wmean,mean,median,min,max,1pgeom}
                        Summary function to calculate density for bin with. (default: wmean)
  -u UBIN, --ubin UBIN  Number of windows for upstream region (default: 50)
  -d DBIN, --dbin DBIN  Number of windows for downstream downstream (default: 50)
  -b BBIN, --bbin BBIN  Number of windows for body region (default: 100)
  -q QUANTILE, --quantile QUANTILE
                        Quantile of most varying genes to draw on clustermap (default: 0.75)
  -C CONFIDENCE, --confidence CONFIDENCE
                        Probability for confidence bands for line-plot. 0 if disabled (default: 0.95)
  -S SMOOTH, --smooth SMOOTH
                        Windows for SavGol function. (default: 10)
  -H VRESOLUTION        Vertical resolution for heat-map (default: 100)
  -V HRESOLUTION        Vertical resolution for heat-map (default: 100)
  --separate_strands    Do strands need to be processed separately (default: False)
  --export {pdf,svg,none}
                        Export format for plots (set none to disable) (default: pdf)
  --ticks TICKS TICKS TICKS TICKS TICKS
                        Names of ticks (5 labels with ; separator in " brackets) (default: None)
  --cytosine_p CYTOSINE_P
                        P-value for binomial test to consider cytosine methylated (default: .05)
  --min_cov MIN_COV     Minimal coverage for cytosine to keep (default: 2)
  --region_p REGION_P   P-value for binomial test to consider region methylated (default: .05)
  --save_cat, --no-save_cat
                        Does categories need to be saved (default: True) (default: True)
```

## bsxplorer-chr

```commandline
bsxplorer-chr --help     
usage: BSXplorer-ChrLevels [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t THREADS] [-w WINDOW] [-l MIN_LENGTH] [-C CONFIDENCE] [-S SMOOTH] [--export {pdf,svg,none}] [--separate_strands] config

Chromosome methylation levels visualisation tool

positional arguments:
  config                Path to config file

optional arguments:
  -h, --help            show this help message and exit
  -o NAME, --out NAME   Output filename (default: Metagene_Report_12-02-24_10-52-40)
  --dir DIR             Output and working dir (default: $CWD)
  -m BLOCK_MB, --block_mb BLOCK_MB
                        Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.) (default: 50)
  -t THREADS, --threads THREADS
                        Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()` (default: True)
  -w WINDOW, --window WINDOW
                        Length of windows in bp (default: 1000000)
  -l MIN_LENGTH, --min_length MIN_LENGTH
                        Minimum length of chromosome to be analyzed (default: 1000000)
  -C CONFIDENCE, --confidence CONFIDENCE
                        Probability for confidence bands for line-plot. 0 if disabled (default: 0.95)
  -S SMOOTH, --smooth SMOOTH
                        Windows for SavGol function. (default: 10)
  --export {pdf,svg,none}
                        Export format for plots (set none to disable) (default: pdf)
  --separate_strands    Do strands need to be processed separately (default: False)
```
