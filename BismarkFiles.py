import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, colors as mpl_colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from Bismark import *


class BismarkFiles:
    """
    Class to process and plot multiple files
    """

    __uninitialized_str = '{1} is None. Ensure that you specified them in constructor'

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
        self.line_plots: list[LinePlot] = []
        self.heat_maps:  list[HeatMap]  = []
        self.bar_plots:  list[BarPlot]  = []
        self.box_plots:  list[BoxPlot]  = []
        self.bismarks:   list[Bismark]  = []

        self.single_fig_dim: tuple = (7, 5)

        self.__flank_windows = flank_windows
        self.__gene_windows = gene_windows

        for file in files:
            bismark = Bismark(file, genome, flank_windows, gene_windows, batch_size, cpu)
            if line_plot:
                self.line_plots.append(bismark.line_plot())
            if heat_map:
                self.heat_maps.append(bismark.heat_map())
            if bar_plot:
                self.bar_plots.append(bismark.bar_plot())
            if box_plot:
                self.box_plots.append(bismark.box_plot())
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

        if not self.__check_data(self.line_plots, 'LinePlots'):
            return

        _, axes = self.__plot_clear()

        labels = self.__check_labels(labels)
        title = self.__format_title(title, context, strand, 'lp')

        for plot, label in zip(self.line_plots, labels):
            plot.draw(axes, context, strand, smooth, label, linewidth, linestyle)

        axes.legend(loc='best')
        axes.set_title(title, fontstyle='italic')
        self.__add_flank_lines(axes)

        self.__set_single_fig_dim()
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
        if not self.__check_data(self.heat_maps, 'HeatMaps'):
            return

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
                image = ax.imshow(
                    im_data, interpolation="nearest", aspect='auto', cmap=colormaps['cividis'], vmin=vmin, vmax=vmax
                )

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
        if not self.__check_data(self.line_plots, 'LinePlots'):
            return

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
        """
        Method to plot all heatmaps and strands
        :param resolution: Number of vertical rows in the resulting image
        :param labels: Labels for files data
        :param titles: Titles of the plot
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        if not self.__check_data(self.heat_maps, 'HeatMaps'):
            return

        filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]
        titles = self.__check_titles(titles)

        for title, (context, strand) in zip(titles, filters):
            self.draw_heat_maps_filtered(context, strand, resolution, labels, title, out_dir, dpi)

    def draw_bar_plot(
            self,
            labels: list[str] = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        """
        Method to plot data for all contexts as bar plot. The plot isn't strand specific.
        :param labels: Labels for files data
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        if not self.__check_data(self.bar_plots, 'BarPlots'):
            return

        plt.clf()
        labels = self.__check_labels(labels)
        df = pd.DataFrame({'context': ['CG', 'CHG', 'CHH']})

        for bar_plot, label in zip(self.bar_plots, labels):
            # convert pl.Dataframe to pd.Dataframe - cast Categorical `context` as Utf8. Sort and convert to list
            df[label] = bar_plot.bismark.with_columns(pl.col('context').cast(pl.Utf8)).sort('context')['density'].to_list()

        self.__set_single_fig_dim()

        df.plot(x='context', kind='bar', stacked=False, edgecolor='k', linewidth=1)
        plt.ylabel('Methylation density')
        plt.savefig(f'{out_dir}/bar_{self.__current_time()}.png', dpi=dpi, bbox_inches='tight')

    def draw_box_plot(
            self,
            strand_specific: bool = False,
            widths: float = .6,
            labels: list[str] = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        """
        Method to plot data for all contexts as box plot
        :param strand_specific: Distinguish strand or not
        :param widths: Widths of bars
        :param labels: Labels for files data
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        if not self.__check_data(self.box_plots, 'BoxPlots'):
            return

        plt.clf()
        labels = self.__check_labels(labels)

        # filters list
        if strand_specific:
            filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]
        else:
            filters = [context for context in ['CG', 'CHG', 'CHH']]

        x_ticks, x_labels = [], []
        bplots = []
        start = 1
        for fil in filters:
            if isinstance(fil, tuple):
                context, strand = fil
            else:
                context, strand = fil, None

            data = [box_plot.filter_density(context, strand) for box_plot in self.box_plots]  # data from all samples filtered
            x_pos = np.arange(start, start + len(data))  # positions of boxes

            bplot = plt.boxplot(data, positions=x_pos, widths=widths, showfliers=False, patch_artist=True)
            bplots.append(bplot)

            for patch, color in zip(bplot['boxes'], mpl_colors.TABLEAU_COLORS):
                patch.set_facecolor(color)  # color boxes
            for median in bplot['medians']:
                median.set_color('black')  # make medians black instead of orange

            x_ticks.append(np.mean(x_pos))
            x_labels.append(f'{context}{strand}')
            start += 1

        plt.xticks(x_ticks, x_labels)

        # This block is needed to make a legend with colors and labels
        lines = []
        for (_, color) in zip(self.box_plots, mpl_colors.TABLEAU_COLORS):
            line, = plt.plot([1, 1], color)
            lines.append(line)
        plt.legend(lines, labels)
        [line.set_visible(False) for line in lines]

        self.__set_single_fig_dim()

        plt.ylabel('Methylation density')
        plt.savefig(f'{out_dir}/box_{self.__current_time()}.png', dpi=dpi)

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

    def __check_data(self, data, data_type: str):
        if data is None:
            print(self.__uninitialized_str.format(data_type))
            return False
        return True

    def __set_single_fig_dim(self):
        plt.gcf().set_size_inches(self.single_fig_dim)

    @staticmethod
    def __format_title(title, context, strand, plot_type):
        if title is None:
            return f'{plot_type}_{context}_{strand}'
        else:
            return title

    @staticmethod
    def __current_time():
        return datetime.datetime.now().strftime('%m/%d/%Y_%H:%M:%S')

    @staticmethod
    def __plot_clear() -> (Axes, Figure):
        plt.clf()
        return plt.subplots()

