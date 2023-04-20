import datetime

import numpy as np

from Bismark import Bismark
import polars as pl
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import colormaps
from heat_map import HeatMap


class BismarkFiles:
    """
    Class to process and plot multiple files
    """
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
            box_plot: bool = False,
            store_res: bool = False
    ):
        """

        :param files: List with paths to bismark genomeWide reports
        :param genome: polars.Dataframe with gene ranges
        :param flank_windows: Number of windows flank regions to split
        :param gene_windows: Number of windows gene regions to split
        :param batch_size: Number of rows to read by one CPU core
        :param cpu: How many cores to use. Uses every physical core by default
        :param line_plot: Whether to plot Line plot or not
        :param heat_map: Whether to plot Heat map or not
        :param bar_plot: Whether to plot Bar plot or not
        :param box_plot: Whether to plot Box plot or not
        """
        self.line_plots = []
        """List with :class:`LinePlot` data"""
        self.heat_maps = []
        """List with :class:`HeatMap` data"""
        self.bar_plots = []
        """List with :class:`BarPlot` data"""
        self.box_plots = []
        """List with :class:`BoxPlot` data"""
        self.bismarks = []
        """List with :class:`Bismark` data"""

        self.__flank_windows = flank_windows
        self.__gene_windows = gene_windows

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
            if store_res:
                self.bismarks.append(bismark)

    def draw_line_plots_filtered(
            self,
            context: str = 'CG',
            strand: str = '+',
            smooth: float = .05,
            labels: list[str] = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
            title: str = None,
            out_dir: str = '',
            dpi: int = 300
    ) -> plt.Axes:
        """
        Method to plot selected context and strand

        :param context: Methylation context to filter
        :param strand: Strand to filter
        :param smooth: Smooth * len(density) = window_length for SavGol filter
        :param labels: Labels for files data
        :param linewidth: Line width
        :param linestyle: Line width see Linestyles_
        :param title: Title of the plot
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        :return: Axes with plot

        .. _Linestyles: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html/
        """

        fig, axes = self.__plot_clear()

        labels = self.__check_labels(labels)
        title = self.__format_title(title, context, strand, 'lp')

        for plot, label in zip(self.line_plots, labels):
            plot.draw(axes, context, strand, smooth, label, linewidth, linestyle)

        axes.legend(loc='best')
        axes.set_title(title, fontstyle='italic')
        self.__add_flank_lines(axes)

        fig.set_size_inches(7, 5)
        plt.savefig(f'{out_dir}/{title}_{self.__current_time()}.png', dpi=dpi)

        return axes

    def draw_heat_maps_filtered(
            self,
            context: str = 'CG',
            strand: str = '+',
            resolution: int = 100,
            labels: list[str] = None,
            title: str = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        plt.clf()
        if len(self.heat_maps) > 3:
            subplots_y = 2
        else:
            subplots_y = 1

        subplots_x = len(self.heat_maps) // subplots_y
        fig, axes = plt.subplots(subplots_y, subplots_x)

        labels = self.__check_labels(labels)
        title = self.__format_title(title, context, strand, 'hm')

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        data = [heat_map.filter(context, strand, resolution) for heat_map in self.heat_maps]
        vmin = 0
        vmax = np.max(np.array(data))

        for i in range(subplots_y):
            for j in range(subplots_x):
                number = i * subplots_x + j
                if subplots_y > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                im_data = data[number]
                image = ax.imshow(im_data, interpolation="nearest", aspect='auto', cmap=colormaps['cividis'], vmin=vmin, vmax=vmax)

                ax.set(title=labels[number])

                self.__add_flank_lines(ax)
                plt.colorbar(image, ax=ax)

        plt.title(title, fontstyle='italic')
        fig.set_size_inches(6 * subplots_x, 5 * subplots_y)
        plt.savefig(f'{out_dir}/{title}_{self.__current_time()}.png', dpi=dpi)

    def draw_line_plots_all(
            self,
            smooth: float = .05,
            labels: list[str] = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
            titles: list[str] = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        """
        Method to plot all contexts and strands

        :param smooth: ``smooth * len(density) = window_length`` for SavGol filter
        :param labels: Labels for files data
        :param linewidth: Line width
        :param linestyle: Line width see Linestyles_
        :param titles: Titles of the plot
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]
        titles = self.__check_titles(titles)

        for title, (context, strand) in zip(titles, filters):
            self.draw_line_plots_filtered(context, strand, smooth, labels, linewidth, linestyle, title, out_dir, dpi)

    def draw_heat_maps_all(
            self,
            resolution: int = 100,
            labels: list[str] = None,
            titles: list[str] = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]
        titles = self.__check_titles(titles)

        for title, (context, strand) in zip(titles, filters):
            self.draw_heat_maps_filtered(context, strand, resolution, labels, title, out_dir, dpi)

    def __add_flank_lines(self, axes: plt.Axes):
        if self.__flank_windows:
            x_ticks = [self.__flank_windows - 1, self.__gene_windows]
            x_labels = ['TSS', 'TES']
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(x_labels)
            for tick in x_ticks:
                axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)

    def __check_labels(self, labels):
        if len(labels) != len(self.line_plots):
            return [None] * len(self.line_plots)
        else:
            return labels

    def __check_titles(self, titles):
        if len(titles) != len(self.line_plots):
            return [None] * len(self.line_plots)
        else:
            return titles

    @staticmethod
    def __format_title(title, context, strand, type):
        if title is None:
            return f'{type}_{context}_{strand}'
        else:
            return title

    @staticmethod
    def __current_time():
        return datetime.datetime.now().strftime('%m/%d/%Y_%H:%M:%S')

    @staticmethod
    def __plot_clear() -> (Axes, Figure):
        plt.clf()
        return plt.subplots()