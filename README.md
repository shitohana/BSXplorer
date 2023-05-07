# BismarkPlot
A small library to plot Bismark ``methylation_exctractor`` reports.

See the docs: https://shitohana.github.io/BismarkPlot

Right now only ``coverage2cytosine`` input is supported, but support for ``bismark2bedGraph`` will be added soon.

## Example

First we need to initialize ``genome`` and ``BismarkFiles``. ``genome`` is .gff or .bed file with gene coordinates. ``BismarkFiles`` is a class, which calculates data for all plots, so their types need to be specified when it is initialized.
```python
import bismarkplot

file = 'path/to/genome.gff'

genome = bismarkplot.read_genome(
    file,
    flank_length=2000,
    min_length=4000
)

files = ['path/to/genomeWide1.txt', 'path/to/genomeWide2.txt']
bismark = bismarkplot.BismarkFiles(
    files, genome,
    flank_windows=500,
    gene_windows=2000,
    line_plot=True,
    heat_map=True,
    box_plot=True,
    bar_plot=True
)
```

Let's now draw plots themselves.

For line plots use (or ``draw_line_plots_all`` for all contexts)
```python
bismark.draw_line_plots_filtered(
    context='CG',
    strand='+',
    smooth=.05,
    labels = ['exp1', 'exp2'],
    title = 'Plot for CG+'
) 
```

For heat maps use (or ``draw_heat_maps_all`` for all contexts)
```python
bismark.draw_heat_maps_filtered(
    context='CG',
    strand='+',
    resolution=100,
    labels = ['exp1', 'exp2'],
    title = 'Heatmap for CG+'
)   
```

For box plot or bar plot use
```python
bismark.draw_box_plot(strand_specific=True, labels=['exp1', 'exp2'])
bismark.draw_bar_plot(labels=['exp1', 'exp2'])
```