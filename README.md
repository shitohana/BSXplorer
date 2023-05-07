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
Result:
![Plot for CG+](https://user-images.githubusercontent.com/43905117/236703691-023818e9-fb0d-47e6-a328-a712c9285928.png)

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

Result:
![Heatmap for CG+](https://user-images.githubusercontent.com/43905117/236703690-b46c7579-3068-4e98-82f0-9a6435c7808b.png)
For box plot or bar plot use
```python
bismark.draw_box_plot(strand_specific=True, labels=['exp1', 'exp2'])
bismark.draw_bar_plot(labels=['exp1', 'exp2'])
```

Result
![box_05_07_23:54.png](https://user-images.githubusercontent.com/43905117/236703689-9eaaa28a-1a98-4300-a0d0-83039ed9a541.png)
![bar_05_07_23:54.png](https://user-images.githubusercontent.com/43905117/236703687-f3fd1225-1ad1-45b0-9318-b2282a694e68.png