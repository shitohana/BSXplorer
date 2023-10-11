# BismarkPlot
Comprehensive tool for visualizing genome-wide cytosine data.

See the docs: https://shitohana.github.io/BismarkPlot

Right now only ``coverage2cytosine`` input is supported, but support for other input types will be added soon.

## Installation

```commandline
pip install bismarkplot
```

## Usage
You can use ```bismarkplot``` either as python library or directly from console after installing it. 

Console options:
- *bismarkplot-metagene* - methylation density visualizing tool. 
- *bismarkplot-chrs* - chromosome methylation levels visualizing tool.

### bismarkplot-metagene

```commandline
usage: BismarkPlot. [-h] [-o OUT] [-g GENOME] [-r {gene,exon,tss,tes}] [-b BATCH] [-c CORES] [-f FLENGTH] [-u UWINDOWS] [-d DWINDOWS] [-m MLENGTH] [-w GWINDOWS] [--line] [--heatmap] [--box] [--violin]
                    [-S SMOOTH] [-L LABELS [LABELS ...]] [-H H] [-V V] [--dpi DPI] [-F {png,pdf,svg}]
                    filename [filename ...]

Metagene visualizing tool.

positional arguments:
  filename              path to bismark methylation_extractor files

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     output base name (default: current/path)
  -g GENOME, --genome GENOME
                        path to GFF genome file (default: None)
  -r {gene,exon,tss,tes}, --region {gene,exon,tss,tes}
                        path to GFF genome file (default: gene)
  -b BATCH, --batch BATCH
                        number of rows to be read from bismark file by batch (default: 1000000)
  -c CORES, --cores CORES
                        number of cores to use (default: None)
  -f FLENGTH, --flength FLENGTH
                        length in bp of flank regions (default: 2000)
  -u UWINDOWS, --uwindows UWINDOWS
                        number of windows for upstream (default: 50)
  -d DWINDOWS, --dwindows DWINDOWS
                        number of windows for downstream (default: 50)
  -m MLENGTH, --mlength MLENGTH
                        minimal length in bp of gene (default: 4000)
  -w GWINDOWS, --gwindows GWINDOWS
                        number of windows for genes (default: 100)
  --line                line-plot enabled (default: False)
  --heatmap             heat-map enabled (default: False)
  --box                 box-plot enabled (default: False)
  --violin              violin-plot enabled (default: False)
  -S SMOOTH, --smooth SMOOTH
                        windows for smoothing (default: 10)
  -L LABELS [LABELS ...], --labels LABELS [LABELS ...]
                        labels for plots (default: None)
  -H H                  vertical resolution for heat-map (default: 100)
  -V V                  vertical resolution for heat-map (default: 100)
  --dpi DPI             dpi of output plot (default: 200)
  -F {png,pdf,svg}, --format {png,pdf,svg}
                        format of output plots (default: pdf)
```

### bismarkplot-chrs

```commandline
usage: BismarkPlot [-h] [-o DIR] [-b N] [-c CORES] [-w N] [-m N] [-S FLOAT] [-F {png,pdf,svg}] path/to/txt [path/to/txt ...]

Chromosome methylation levels visualization.

positional arguments:
  path/to/txt           path to bismark methylation_extractor file

options:
  -h, --help            show this help message and exit
  -o DIR, --out DIR     output base name (default: current/path)
  -b N, --batch N       number of rows to be read from bismark file by batch (default: 1000000)
  -c CORES, --cores CORES
                        number of cores to use (default: None)
  -w N, --wlength N     number of windows for genes (default: 100000)
  -m N, --mlength N     minimum chromosome length (default: 1000000)
  -S FLOAT, --smooth FLOAT
                        windows for smoothing (0 - no smoothing, 1 - straight line (default: 50)
  -F {png,pdf,svg}, --format {png,pdf,svg}
                        format of output plots (default: pdf)

```