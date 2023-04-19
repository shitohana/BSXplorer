import datetime

from Bismark import Bismark
import polars as pl
from multiprocessing import cpu_count
import matplotlib.pyplot as plt


class BismarkFiles:
    def __init__(
            self,
            files: list[str],
            genome: pl.DataFrame,
            flank_windows: int = 500,
            gene_windows: int = 2000,
            batch_size: int = 10 ** 6,
            cpu: int = cpu_count(),
            line_plot: bool = True,
            heat_map: bool = False,
            bar_plot: bool = False,
            box_plot: bool = False
    ):
        self.line_plots = []
        self.heat_maps = []
        self.bar_plots = []
        self.box_plots = []

        for file in files:
            bismark = Bismark(file, genome, flank_windows, gene_windows, batch_size, cpu)
            if line_plot:
                self.line_plots.append(bismark.line_plot())
            if heat_map:
                self.line_plots.append(bismark.heat_map())
            if bar_plot:
                self.line_plots.append(bismark.bar_plot())
            if box_plot:
                self.line_plots.append(bismark.box_plot())

    def draw_line_plots_filtered(
            self,
            context: str = 'CG',
            strand: str = '+',
            smooth: float = .05,
            labels: list[str] = None,
            color: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
            title: str = None,
            out_dir: str = '',
            dpi: int = 300
    ) -> plt.Axes:

        plt.clf()
        fig, axes = plt.subplots()

        for i in range(len(self.line_plots)):
            self.line_plots[i].draw(axes, context, strand, smooth, labels[i], color, linewidth, linestyle)

        if title is None:
            title = f'lp_{context}_{strand}'
        plt.gcf().set_size_inches(7, 5)
        plt.legend(loc='best')
        time = datetime.datetime.now().strftime('%m/%d/%Y_%H:%M:%S')
        plt.savefig(f'{out_dir}/{title}_{time}.png', dpi=dpi)
        return axes
