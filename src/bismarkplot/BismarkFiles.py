import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, colors as mpl_colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .Bismark import *


class BismarkFiles:
    """
    Class to process and plot multiple files
    """

    __uninitialized_str = '{} is None. Ensure that you specified them in constructor'

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
        :param flank_windows: Number of windows for flank regions
        :param gene_windows: Number of windows for gene regions
        :param batch_size: Number of rows to read by one CPU core
        :param cpu: How many cores to use. Uses every physical core by default
        :param line_plot: Whether to plot Line plot or not
        :param heat_map: Whether to plot Heat map or not
        :param bar_plot: Whether to plot Bar plot or not
        :param box_plot: Whether to plot Box plot or not
        """
        self.__logger = logging.Logger('BismarkFiles')
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%H:%M:%S'))
        self.__logger.addHandler(sh)
        self.__logger.setLevel(logging.INFO)

        self.line_plots: list[LinePlot] = []
        self.heat_maps:  list[HeatMap]  = []
        self.bar_plots:  list[BarPlot]  = []
        self.box_plots:  list[BoxPlot]  = []
        self.bismarks:   list[Bismark]  = []

        self.__files_num = len(files)

        self.single_fig_dim: tuple = (7, 5)

        self.__flank_windows = flank_windows
        self.__gene_windows = gene_windows

        logging_message = 'Starting BismarkFiles initialization\n' \
            '_________________________________________________________\n' \
            f'Numbers of CPU cores: {cpu}\n' \
            f'Windows: Genes ({gene_windows}) | Flank ({flank_windows})\n' \
            f'Calculating:\n'

        for plot, enabled in zip(['Line Plot', 'Heat Map', 'Box Plot', 'Bar Plot'], [line_plot, heat_map, box_plot, bar_plot]):
            if enabled:
                logging_message += plot + '\n'

        self.__logger.info(logging_message)

        for file in files:
            self.__logger.info(f'Processing {file}')

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
    ):
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

        .. _Linestyles: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html/
        """

        out_dir = self.__format_dir(out_dir)
        if not self.__check_data(self.line_plots, 'LinePlots'):
            return

        _, axes = self.__plot_clear()

        labels = self.__check_labels(labels)
        title = self.__format_title(title, context, strand, 'lp')

        for plot, label in zip(self.line_plots, labels):
            plot.draw(axes, context, strand, smooth, label, linewidth, linestyle)

        if len(set(labels)) and list(set(labels))[0] is not None:
            axes.legend(loc='best')

        axes.set_title(title.replace('_', ' '), fontstyle='italic')
        self.__add_flank_lines(axes)

        self.__set_single_fig_dim()
        file_name = f'{title}_{self.__current_time()}.png'
        self.__logger.info(f'Line plot on {context}{strand} saved as {file_name}')
        plt.savefig(f'{out_dir}/{file_name}', dpi=dpi)

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
        """
        Method to plot heatmap for selected context and strand

        :param context: Methylation context to filter
        :param strand: Strand to filter
        :param resolution:
        :param labels: Labels for files data
        :param title: Title of the plot
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        out_dir = self.__format_dir(out_dir)
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
                ax.set_xlabel('Position')
                ax.set_ylabel('')
                self.__add_flank_lines(ax)
                plt.colorbar(image, ax=ax, label='Methylation density')

        plt.suptitle(title.replace('_', ' '), fontstyle='italic')
        fig.set_size_inches(6 * subplots_x, 5 * subplots_y)
        file_name = f'{title}_{self.__current_time()}.png'
        self.__logger.info(f'Heat Map on {context}{strand} saved as {file_name}')
        plt.savefig(f'{out_dir}/{file_name}', dpi=dpi)

    def draw_line_plots_all(
            self,
            smooth: float = .05,
            labels: list[str] = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
            out_dir: str = '',
            dpi: int = 300
    ):
        """
        Method to plot all contexts and strands

        :param smooth: ``smooth * len(density) = window_length`` for SavGol filter
        :param labels: Labels for files data
        :param linewidth: Line width
        :param linestyle: Line style
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        out_dir = self.__format_dir(out_dir)
        if not self.__check_data(self.line_plots, 'LinePlots'):
            return

        filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]

        for context, strand in filters:
            self.draw_line_plots_filtered(context, strand, smooth, labels, linewidth, linestyle, None, out_dir, dpi)

    def draw_heat_maps_all(
            self,
            resolution: int = 100,
            labels: list[str] = None,
            out_dir: str = '',
            dpi: int = 300
    ):
        """
        Method to plot all heatmaps and strands

        :param resolution: Number of vertical rows in the resulting image
        :param labels: Labels for files data
        :param out_dir: directory to save plots to
        :param dpi: DPI of output pic
        """
        out_dir = self.__format_dir(out_dir)
        if not self.__check_data(self.heat_maps, 'HeatMaps'):
            return

        filters = [(context, strand) for context in ['CG', 'CHG', 'CHH'] for strand in ['+', '-']]

        for context, strand in filters:
            self.draw_heat_maps_filtered(context, strand, resolution, labels, None, out_dir, dpi)

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
        out_dir = self.__format_dir(out_dir)
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
        plt.xlabel('Context')
        plt.ylabel('Methylation density')
        file_name = f'bar_{self.__current_time()}.png'
        self.__logger.info(f'Bar Plot saved as {file_name}')
        plt.savefig(f'{out_dir}/{file_name}', dpi=dpi, bbox_inches='tight')

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
        out_dir = self.__format_dir(out_dir)
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
        start = 0
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
            if strand is None:
                x_labels.append(f'{context}')
            else:
                x_labels.append(f'{context}{strand}')
            start += len(data)

        plt.xticks(x_ticks, x_labels)

        # This block is needed to make a legend with colors and labels
        lines = []
        for (_, color) in zip(self.box_plots, mpl_colors.TABLEAU_COLORS):
            line, = plt.plot([0, 0], color)
            lines.append(line)
        plt.legend(lines, labels)
        [line.set_visible(False) for line in lines]

        self.__set_single_fig_dim()

        plt.xlabel('Context')
        plt.ylabel('Methylation density')
        file_name = f'box_{self.__current_time()}.png'
        self.__logger.info(f'Box Plot saved as {file_name}')
        plt.savefig(f'{out_dir}/{file_name}', dpi=dpi, bbox_inches='tight')

    def __add_flank_lines(self, axes: plt.Axes):
        """
        Add flank lines to the given axis (for line plot)
        """
        if self.__flank_windows:
            x_ticks = [self.__flank_windows - 1, self.__gene_windows + self.__flank_windows]
            x_labels = ['TSS', 'TES']
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(x_labels)
            for tick in x_ticks:
                axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)

    def __check_labels(self, labels):
        """
        Check if labels length is the same as data length and fix it if it's False
        """
        if labels is not None:
            if len(labels) < self.__files_num:
                return labels + [None] * (self.__files_num - len(labels))
            elif len(labels) > self.__files_num:
                return labels[:self.__files_num]
            else:
                return labels
        return [None] * self.__files_num

    def __check_titles(self, titles):
        """
        Check if titles length is the same as data length and fix it if it's False
        """
        if titles is not None:
            if len(titles) < self.__files_num:
                self.__logger.error('Not enough titles')
                return titles + [None] * (self.__files_num - len(titles))
            elif len(titles) > self.__files_num:
                self.__logger.error('Too many titles')
                return titles[:self.__files_num]
            else:
                return titles
        return [None] * self.__files_num

    def __check_data(self, data, data_type: str):
        """
        Method to skip plot if data for it was not calculated during the initialization
        """
        if data is None:
            self.__logger.error(self.__uninitialized_str.format(data_type))
            return False
        return True

    def __set_single_fig_dim(self):
        """
        Set size for a Figure with single plot
        """
        plt.gcf().set_size_inches(self.single_fig_dim)

    @staticmethod
    def __format_title(title, context, strand, plot_type):
        """
        Format title if it's not given
        """
        if title is None:
            return f'{plot_type}{context}_{strand}'
        else:
            return title

    @staticmethod
    def __current_time():
        """
        Current time for file names
        """
        return datetime.datetime.now().strftime('%m_%d_%H:%M')

    @staticmethod
    def __plot_clear() -> (Axes, Figure):
        """
        Cleat previous plot and get new Axes, Figure
        """
        plt.clf()
        return plt.subplots()

    @staticmethod
    def __format_dir(directory):
        """
        Format directory name
        """
        if not directory:
            return os.getcwd()
        if list(directory)[-1] == '/':
            return str(list(directory[:-1]))
        return directory
