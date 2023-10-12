# BismarkPlot
Comprehensive tool for visualizing genome-wide cytosine data.

See the docs: https://shitohana.github.io/BismarkPlot

Right now only ``coverage2cytosine`` input is supported, but support for other input types will be added soon.

# Installation

```commandline
pip install bismarkplot
```

# Console usage
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

# Python

BismarkPlot provides a large variety of function for manipulating with cytosine methylation data.

## Basic workflow

Below we will show the basic BismarkPlot workflow.

### Single sample

```python
import bismarkplot
# Firstly, we need to read the regions annotation (e.g. reference genome .gff)
genome = bismarkplot.Genome.from_gff("path/to/genome.gff")  
# Next we need to filter regions of interest from the genome
genes = genome.gene_body(min_length=4000, flank_length=2000)

# Now we need to calculate metagene data
metagene = bismarkplot.Metagene.from_file(
    file = "path/to/CX_report.txt",
    genome=genes,                         # filtered regions
    upstream_windows = 500,               
    gene_windows = 1000,
    downstream_windows = 500,
    batch_size= 10**7                     # number of lines to be read simultaneously
)

# Our metagene contains all methylation contexts and both strands, so we need to filter it (as in dplyr)
filtered = metagene.filter(context = "CG", strand = "+")
# We are ready to plot
lp = filtered.line_plot()                 # line plot data
lp.draw().savefig("path/to/lp.pdf")       # matplotlib.Figure

hm = filtered.heat_map(ncol=200, nrow=200)
lp.draw().savefig("path/to/hm.pdf")       # matplotlib.Figure
```
Output:

<img src="https://user-images.githubusercontent.com/43905117/274546389-8b97edcb-6ab4-4f17-a970-389819fbeaec.png" width="300">
<img src="https://user-images.githubusercontent.com/43905117/274546419-079e004b-8f6e-4ce9-a3dc-fc4ad9592adc.png" width="300">

### Multiple samples, same specie

```python
# We can initialize genome like in previous example

filenames = ["report1.txt", "report2.txt", "report3.txt"]
metagenes = bismarkplot.MetageneFiles.from_list(filenames, labels = ["rep1", "rep2", "rep3"], ...)  # rest of params from previous example

# Our metagenes contains all methylation contexts and both strands, so we need to filter it (as in dplyr)
filtered = metagenes.filter(context = "CG", strand = "+")

# Now we can draw line-plot or heatmap like in previous example, or plot distribution statistics as shown below
trimmed = filtered.trim_flank()           # we want to analyze only gene bodies
trimmed.box_plot(showfliers=True).savefig(...)
trimmed.violin_plot().savefig(...)

# If data is technical replicates we can merge them into single DataFrame and analyze as one
merged = filtered.merge()
```