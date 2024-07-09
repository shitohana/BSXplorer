=============
Console usage
=============

*   :ref:`bsxplorer-metagene`

*   :ref:`bsxplorer-category`

*   :ref:`bsxplorer-chr`

*   :ref:`bsxplorer-bam`

^^^^^^^^^^^^^^^^^^
bsxplorer-metagene
^^^^^^^^^^^^^^^^^^

.. code-block:: console

    usage: bsxplorer-metagene [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t] [-s {wmean,mean,median,min,max,1pgeom}] [-u UBIN] [-d DBIN] [-b BBIN] [-q QUANTILE] [-C CONFIDENCE] [-S SMOOTH] [-H VRESOLUTION] [-V HRESOLUTION]
                              [--separate_strands] [--export {pdf,svg,none}] [--ticks TICKS TICKS TICKS TICKS TICKS]
                              config

    Metagene report creation tool

    positional arguments:
      config                Path to config file

    options:
      -h, --help            show this help message and exit
      -o NAME, --out NAME   Output filename (default: Metagene_Report_08-07-24_18-47-26)
      --dir DIR             Output and working dir (default: /Users/shitohana/Desktop/PycharmProjects/BSXplorer/tests)
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
                            Names of ticks (- character should be escaped with double reverse slash) (default: None)



^^^^^^^^^^^^^^^^^^
bsxplorer-category
^^^^^^^^^^^^^^^^^^

.. code-block:: console


    usage: bsxplorer-categorise [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t] [-s {wmean,mean,median,min,max,1pgeom}] [-u UBIN] [-d DBIN] [-b BBIN] [-q QUANTILE] [-C CONFIDENCE] [-S SMOOTH] [-H VRESOLUTION] [-V HRESOLUTION]
                                [--separate_strands] [--export {pdf,svg,none}] [--ticks TICKS TICKS TICKS TICKS TICKS] [--cytosine_p CYTOSINE_P] [--min_cov MIN_COV] [--region_p REGION_P] [--save_cat | --no-save_cat]
                                config

    BM, UM categorisation tool

    positional arguments:
      config                Path to config file

    options:
      -h, --help            show this help message and exit
      -o NAME, --out NAME   Output filename (default: Metagene_Report_08-07-24_18-49-15)
      --dir DIR             Output and working dir (default: /Users/shitohana/Desktop/PycharmProjects/BSXplorer/tests)
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
                            Names of ticks (- character should be escaped with double reverse slash) (default: None)
      --cytosine_p CYTOSINE_P
                            P-value for binomial test to consider cytosine methylated (default: .05)
      --min_cov MIN_COV     Minimal coverage for cytosine to keep (default: 2)
      --region_p REGION_P   P-value for binomial test to consider region methylated (default: .05)
      --save_cat, --no-save_cat
                            Does categories need to be saved (default: True)


^^^^^^^^^^^^^
bsxplorer-chr
^^^^^^^^^^^^^

.. code-block:: console


    usage: bsxplorer-chr [-h] [-o NAME] [--dir DIR] [-m BLOCK_MB] [-t THREADS] [-w WINDOW] [-l MIN_LENGTH] [-C CONFIDENCE] [-S SMOOTH] [--export {pdf,svg,none}] [--separate_strands] config

    Chromosome methylation levels visualisation tool

    positional arguments:
      config                Path to config file

    options:
      -h, --help            show this help message and exit
      -o NAME, --out NAME   Output filename (default: Metagene_Report_08-07-24_18-47-14)
      --dir DIR             Output and working dir (default: /Users/shitohana/Desktop/PycharmProjects/BSXplorer/tests)
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
        # TODO CHANGE DEFAULT WINDOWS
      -S SMOOTH, --smooth SMOOTH
                            Windows for SavGol function. (default: 10)
      --export {pdf,svg,none}
                            Export format for plots (set none to disable) (default: pdf)
      --separate_strands    Do strands need to be processed separately (default: False)

^^^^^^^^^^^^^
bsxplorer-bam
^^^^^^^^^^^^^

.. code-block:: console

    usage: bsxplorer-bam [-h] --bam BAM --bai BAI [-f FASTA] [--bamtype {bismark}] [-m {report,stats}] [--to_type {bismark,cgmap,bedgraph,coverage,binom}] [--stat {ME,EPM,PDR}] [--stat_param STAT_PARAM] [--stat_md STAT_MD]
                         [-g GFF] [-c {CG,CHG,CHH,all}] [-q {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42}] [-s] [--no_qc] [-t THREADS] [-n BATCH_N]
                         [-a READAHEAD]
                         output

    BAM to report reader converter tool.

    positional arguments:
      output                Path to output file.

    options:
      -h, --help            show this help message and exit
      --bam BAM             Path to SORTED .bam file with alignments (default: None)
      --bai BAI             Path to .bai index file (default: None)
      -f FASTA, --fasta FASTA
                            Path to .fasta file with reference sequence for full cytosine report. (default: None)
      --bamtype {bismark}   Type of aligner which was used for generating BAM. (default: bismark)
      -m {report,stats}, --mode {report,stats}
      --to_type {bismark,cgmap,bedgraph,coverage,binom}
                            Specifies the output file type if mode is set to 'report'. (default: bismark)
      --stat {ME,EPM,PDR}   Specifies the BAM stat type if mode is set to 'stats' (default: ME)
      --stat_param STAT_PARAM
                            See docs for specifical stat parameters. (default: 4)
      --stat_md STAT_MD     Minimum number of reads for cytosine to be analysed (if mode is 'stats') (default: 4)
      -g GFF, --gff GFF     Path to regions genome coordinates .gff file, if cytosines need to be filtered. (default: None)
      -c {CG,CHG,CHH,all}, --context {CG,CHG,CHH,all}
                            Filter cytosines by specific methylation context (default: all)
      -q {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42}, --min_qual {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42}
                            Filter cytosines by read Phred score quality (default: None)
      -s, --skip_converted  Skip reads aligned to converted sequence (default: False)
      --no_qc               Do not calculate QC stats (default: False)
      -t THREADS, --threads THREADS
                            How many threads will be used for reading the BAM file. (default: 1)
      -n BATCH_N, --batch_n BATCH_N
                            Number of reads per batch. (default: 10000.0)
      -a READAHEAD, --readahead READAHEAD
                            Number of batches to be read before processing. (default: 5)
