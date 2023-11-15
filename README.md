# BismarkPlot

Analytical framework for BS-seq data comparison and visualization.

See the docs: https://shitohana.github.io/BismarkPlot

Right now only ``coverage2cytosine`` input is supported, but support for other input types will be added soon.

# Installation

```commandline
pip install bismarkplot
```

# Console usage
You can use ```bismarkplot``` either as python library or directly from console after installing it. 

Console options:
- `bismarkplot-metagene` - methylation density visualizing tool. 
- `bismarkplot-chrs` - chromosome methylation levels visualizing tool.

### bismarkplot-metagene

```commandline
usage: BismarkPlot. [-h] [-o NAME] [--dir DIR] [-g GENOME] [-r {gene,exon,tss,tes}] [-b BATCH] [-c CORES] [-f FLENGTH] [-u UWINDOWS] [-d DWINDOWS] [-m MLENGTH] [-w GWINDOWS] [--line] [--heatmap]
                    [--box] [--violin] [-S SMOOTH] [-L LABELS [LABELS ...]] [-C CONFIDENCE] [-H VRESOLUTION] [-V HRESOLUTION] [--dpi DPI] [-F {png,pdf,svg}]
                    filename [filename ...]

Metagene visualizing tool.

positional arguments:
  filename              path to bismark methylation_extractor files

optional arguments:
  -h, --help            show this help message and exit
  -o NAME, --out NAME   output base name (default: plot)
  --dir DIR             output dir (default: /Users/shitohana/PycharmProjects/BismarkPlot_test)
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
  -C CONFIDENCE, --confidence CONFIDENCE
                        probability for confidence bands for line-plot. 0 if disabled (default: 0)
  -H VRESOLUTION        vertical resolution for heat-map (default: 100)
  -V HRESOLUTION        vertical resolution for heat-map (default: 100)
  --dpi DPI             dpi of output plot (default: 200)
  -F {png,pdf,svg}, --format {png,pdf,svg}
                        format of output plots (default: pdf)
```

Example:

```commandline
bismarkplot-metagene -g path/to/genome.gff -r gene -f 2000 -m 4000  -u 500 -d 500 -w 1000 -b 1000000 --line --heatmap --box --violin --dpi 200 -f pdf -S 50 report1.txt report2.txt report3.txt report4.txt 
```

[Result](#multiple-samples-same-specie)

### bismarkplot-chrs

```commandline
usage: BismarkPlot [-h] [-o NAME] [-d DIR] [-b N] [-c CORES] [-w N] [-m N] [-S FLOAT] [-F {png,pdf,svg}] [--dpi DPI] path/to/txt

Chromosome methylation levels visualization.

positional arguments:
  path/to/txt           path to bismark methylation_extractor file

optional arguments:
  -h, --help            show this help message and exit
  -o NAME, --out NAME   output base name (default: plot)
  -d DIR, --dir DIR     output dir (default: /Users/shitohana/PycharmProjects/BismarkPlot_test)
  -b N, --batch N       number of rows to be read from bismark file by batch (default: 1000000)
  -c CORES, --cores CORES
                        number of cores to use (default: None)
  -w N, --wlength N     number of windows for chromosome (default: 100000)
  -m N, --mlength N     minimum chromosome length (default: 1000000)
  -S FLOAT, --smooth FLOAT
                        windows for smoothing (0 - no smoothing, 1 - straight line (default: 50)
  -F {png,pdf,svg}, --fmt {png,pdf,svg}
                        format of output plots (default: pdf)
  --dpi DPI             dpi of output plot (default: 200)
```

Example:

```commandline
bismarkplot-chrs -b 10000000 -w 10000 -m 1000000 -s 10 -f pdf path/to/CX_report.txt
```

[Result](#chromosome-levels)

# Python

BismarkPlot provides a large variety of function for manipulating with cytosine methylation data.

## Metagene

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
hm.draw().savefig("path/to/hm.pdf")       # matplotlib.Figure
```
Output for _Brachypodium distachyon_:

<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/280025496-b2336c72-5109-42d4-a770-f0a480ebf40d.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/280025490-c72da09a-7841-471a-bc39-086aa77f65e4.png" width="300">
</p>

If metagene is not filtered by context, **all available contexts will be plotted**:

```python
filtered_by_strand = metagene.filter(strand == "+")
lp = filtered_by_strand.line_plot()
lp.draw()
```

Output for _Brachypodium distachyon_:

<p align="middle">
    <img width="300" src="https://user-images.githubusercontent.com/43905117/280023042-849599c1-4b36-47e2-8b8f-6c9b9389b48e.png">
</p>

**Confidence bands** can be visualized via setting the `confidence` parameter in `LinePlot.draw()`

```python
lp.draw(confidence=.95)
```

Output for _Brachypodium distachyon_: 

<p align="middle">
    <img width="300" src="https://user-images.githubusercontent.com/43905117/280023017-e1167a90-83d7-46d5-aa45-545d6bdbc033.png">
</p>

### Heat-map clusterisation

Genes can be clustered to minimize distances between them before plotting heat-map. This can be useful for capturing 
overall methylation patterns in sample. _This operation is very time consuming. It is advised to set small number of
windows (< 50)_.

```python
metagene = bismarkplot.Metagene.from_file(
    file = "path/to/CX_report.txt",
    genome=genes,                         # filtered regions
    upstream_windows = 5, gene_windows = 10, downstream_windows = 5,
)
clustered = metagene.clustering(
    count_threshold=5,                    # Minimum counts per window
    dist_method="euclidean",              # See scipy.spatial.distance.pdist
    clust_method="average"                # See scipy.cluster.hierarchy.linkage
)

# Heatmap with optimized distances between genes will be drawn
clustered.draw().savefig("path/to/clustered_hm.pdf")
```
Output for _Brachypodium distachyon_ - CHG

<p align="middle">
    <img width="300" src="https://user-images.githubusercontent.com/43905117/282321746-baf97da1-6a35-4a17-9772-5a2f4d67e6a4.png">
</p>

### Genes dynamicTreeCut

To shrink clustered heat-map and capture main patterns genes can be split into modules using 
[dynamicTreeCut algorithm](https://github.com/kylessmith/dynamicTreeCut/tree/master) by Peter Langfelder and Bin Zhang.
Then genes can be plotted as heat-map as previous example:

```python
# Parameters are the same as for cutreeHybrid (see dynamicTreeCut)
modules = clustered.modules(deepSplit = 1)

modules.draw().savefig("path/to/modules_hm.pdf")
```

Output for _Brachypodium distachyon_ - CHG

<p align="middle">
    <img width="300" src="https://user-images.githubusercontent.com/43905117/282321739-df4486f6-9b1f-467e-87ea-cfd441f80e0a.png">
</p>

### Smoothing the line plot

Smoothing is very useful, when input signal is very weak (e.g. mammalian non-CpG contexts)

```python
# mouse CHG methylation example
filtered = metagene.filter(context = "CHG", strand = "+")
lp.draw(smooth = 0).savefig("path/to/lp.pdf")       # no smooth
lp.draw(smooth = 50).savefig("path/to/lp.pdf")      # smoothed with window length = 50
```

Output for _Mus musculus_:

<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274557328-5a087a43-5731-4cef-aa90-cf2ce046c747.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274557346-97e10689-609c-4032-a14d-5893b6797d59.png" width="300">
</p>

### Multiple samples, same specie

```python
# We can initialize genome like in previous example

filenames = ["report1.txt", "report2.txt", "report3.txt", "report4.txt"]
metagenes = bismarkplot.MetageneFiles.from_list(filenames, labels = ["1", "2", "3", "4"], ...)  # rest of params from previous example

# Our metagenes contains all methylation contexts and both strands, so we need to filter it (as in dplyr)
filtered = metagenes.filter(context = "CG", strand = "+")

# Now we can draw line-plot or heatmap like in previous example, or plot distribution statistics as shown below
trimmed = filtered.trim_flank()           # we want to analyze only gene bodies
trimmed.box_plot(showfliers=False).savefig(...)
trimmed.violin_plot().savefig(...)

# If data is technical replicates we can merge them into single DataFrame and analyze as one
merged = filtered.merge()
```

Output for _Brachypodium distachyon_:

<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274546531-8516858a-8203-4e98-98a9-7351efb79d29.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274546553-f2617948-4d74-4f1e-9543-e4fff49deae7.png" width="300">
</p>
<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274546624-9a2da41b-5c3b-4f65-baee-29086a40e020.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274546690-83757110-83cc-4f5f-ad97-b233faa54b97.png" width="300">
</p>

### Multiple samples, multiple species

```python
# For analyzing samples with different reference genomes, we need to initialize several genomes instances
genome_filenames = ["arabidopsis.gff", "brachypodium.gff", "cucumis.gff", "mus.gff"]
reports_filenames = ["arabidopsis.txt", "brachypodium.txt", "cucumis.txt", "mus.txt"]

genomes = [
    bismarkplot.Genome.from_gff(file).gene_body(...) for file in genome_filenames
]

# Now we read reports
metagenes = []
for report, genome in zip(reports_filenames, genomes):
    metagene = bismarkplot.Metagene(report, genome = genome, ...)
    metagenes.append(metagene)

# Initialize MetageneFiles
labels = ["A. thaliana", "B. distachyon", "C. sativus", "M. musculus"]
metagenes = Bismarkplot.MetageneFiles(metagenes, labels)
# Now we can plot them like in previous example
```

Output:

<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274552095-bdb87510-9093-4092-8b30-db71ec8ef12d.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274552066-a26350e8-8f66-4ffd-8a24-a0882051149a.png" width="300">
</p>
<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274552038-641ac683-b43f-4a6a-8636-dd32f7226f28.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274552121-d28949f3-cb6c-48b2-8f6d-81043aed7c13.png" width="300">
</p>

### Different regions

Other genomic regions from .gff can be analyzed too with ```.exon``` or ```.near_tss/.near_tes``` option for ```bismarkplot.Genome```

```python
exons = [
    bismarkplot.Genome.from_gff(file).exon(min_length=100) for file in genome_filenames
]
metagenes = []
for report, exon in zip(reports_filenames, exons):
    metagene = bismarkplot.Metagene(report, genome = exon, 
                                    upstream_windows = 0,   # !!!
                                    downstream_windows = 0, # !!!
                                    ...)
    metagenes.append(metagene)
# OR
tss = [
    bismarkplot.Genome.from_gff(file).near_tss(min_length = 2000, flank_length = 2000) for file in genome_filenames
]
metagenes = []
for report, t in zip(reports_filenames, tss):
    metagene = bismarkplot.Metagene(report, genome = t, 
                                    upstream_windows = 1000,# same number of windows
                                    gene_windows = 1000,    # same number of windows
                                    downstream_windows = 0, # !!!
                                    ...)
    metagenes.append(metagene)
```

Exon output:

<p float="left" align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274564386-767d8bea-87c8-41c5-b43f-bbb93c987844.png" width="300">
    <img src="https://user-images.githubusercontent.com/43905117/274564376-1c662e9b-4194-443a-9f83-5e92bf2387cc.png" width="300">
</p>

TSS output:
<p align="middle">
    <img src="https://user-images.githubusercontent.com/43905117/274552171-40be1461-9907-4d16-a6d3-a44ad53178ea.png" width="300">
</p>

## Chromosome levels

BismarkPlot allows user to visualize chromosome methylation levels across full genome

```python
import bismarkplot
chr = bismarkplot.ChrLevels.from_file(
    "path/to/CX_report.txt",
    window_length=10**5,                  # window length in bp
    batch_size=10**7,                     
    chr_min_length = 10**6,               # minimum chr length in bp
)
fig, axes = plt.subplots()

for context in ["CG", "CHG", "CHH"]:
     chr.filter(strand="+", context=context).draw(
         (fig, axes),                     # to plot contexts on same axes
         smooth=10,                       # window number for smoothing
         label=context                    # labels for lines
     )

fig.savefig(f"chrom.pdf", dpi = 200)
```

Output for _Arabidopsis thaliana_:

<img src="https://user-images.githubusercontent.com/43905117/274563188-6efc5b71-9c83-4fe0-8b5a-767db6e1acb4.png">

Output for _Brachypodium distachyon_:

<img src="https://user-images.githubusercontent.com/43905117/274563210-4f5dc20a-4ab3-4e52-8263-6ebe7b0623d5.png">
