import polars as pl
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class LinePlot:
    def __init__(self, bismark: pl.DataFrame):
        density = (
            bismark.lazy()
            .groupby(
                ['fragment', 'context', 'strand']
            )
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
        ).collect()

        fragments = density.max()['fragment'].to_list()[0] + 1

        density = (
            density.lazy()
            .groupby(['context', 'strand'])
            .agg(
                pl.arange(0, fragments, dtype=density.schema['fragment'])
                .alias('fragment')
            )
            .explode('fragment')
            .join(
                density.lazy(),
                on=['context', 'strand', 'fragment'],
                how='left'
            )
            .sort('fragment')
            .interpolate()
        ).collect()
        self.bismark = density

    def filter(self, context: str = 'CG', strand: str = '+'):
        density = self.bismark.filter(
            (pl.col('context') == context) & (pl.col('strand') == strand)
        )

        density = density['density'].to_list()

        if strand == '-':
            density = np.flip(density)

        return density

    def draw(
            self,
            axes: plt.Axes = None,
            context: str = 'CG',
            strand: str = '+',
            smooth: float = .05,
            label: str = None,
            color: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
    ):
        if axes is None:
            _, axes = plt.subplots()
        data = self.filter(context, strand)
        if smooth:
            data = signal.savgol_filter(data, int(len(data) * smooth), 3, mode='nearest')
        x = np.arange(len(data))

        axes.plot(x, data, label=label, color=color, linestyle=linestyle, linewidth=linewidth)


def line_plot_data(bismark: pl.DataFrame):
    """
    This method extracts data for line plot from bismark DataFrame
    :param bismark: Bismark polars.Dataframe
    :return: polars.Dataframe with density grouped by fragment
    """
    density = (
        bismark.lazy()
        .groupby(
            ['fragment', 'context', 'strand']
        )
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
    ).collect()

    fragments = density.max()['fragment'].to_list()[0] + 1

    density = (
        density.lazy()
        .groupby(['context', 'strand'])
        .agg(
            pl.arange(0, fragments, dtype=density.schema['fragment'])
            .alias('fragment')
        )
        .explode('fragment')
        .join(
            density.lazy(),
            on=['context', 'strand', 'fragment'],
            how='left'
        )
        .sort('fragment')
        .interpolate()
    ).collect()

    return density


def line_plot_filter(density: pl.DataFrame, context: str = 'CG', strand: str = '+', smooth: float = None) -> np.ndarray:
    """
    This method filters line plot Dataframe by context and strand.
    :param density: ``line_plot_data`` output
    :param context: Methylation context to filter
    :param strand: Strand to filter
    :param smooth: smooth * len(density) = window_length for SavGol filter
    :return: numpy array with densities by fragment
    """
    if smooth > 1.0:
        print('Smooth should be less than 1. Using no smooth')
        smooth = None

    density = density.filter(
        (pl.col('context') == context) & (pl.col('strand') == strand)
    )

    density = density['density'].to_list()

    if smooth is not None:
        density = signal.savgol_filter(density, int(len(density) * smooth), 3, mode='nearest')

    if strand == '-':
        density = np.flip(density)

    return density


def draw_line_plot(
        data: list,
        flank_windows: int = 0,
        labels: list       = None,
        title: str         = '',
        out_dir: str       = ''
):
    """
    Method to plot :func:`line_plot_data`. Use :func:`line_plot_filter`.
    :param data: list with line plot data arrays
    :param flank_windows: Number of flank windows (used to set labels)
    :param labels: Line labels
    :param title: Plot title
    :param out_dir: Path to output directory
    """
    plt.clf()
    x = list(range(len(data[0])))
    for i in range(len(data)):
        label = None
        if labels:
            label = labels[i]
        plt.plot(x, data[i], lw=1, label=label)
    if flank_windows:
        x_ticks = [flank_windows - 1, len(data[0]) - flank_windows]
        x_labels = ['TSS', 'TES']
        plt.xticks(x_ticks, x_labels)
        for tick in x_ticks:
            plt.axvline(x=tick, linestyle='--', color='k', alpha=.3)
    if title:
        plt.title(title, fontstyle='italic')
    plt.ylabel('Methylation density')
    plt.xlabel('Position')
    plt.gcf().set_size_inches(7, 5)
    plt.legend(loc='best')
    plt.savefig(f'{out_dir}/{title}.png', dpi=300)


def draw_line_plots(
        data: list[pl.DataFrame],
        flank_windows: int  = 0,
        labels:        list = None,
        out_dir:       str  = '',
        smooth: float       = .05
):
    """
    Line plots for multiple DataFrames for all contexts and strands
    :param data: DataFrames with ``line_plot_
    :param flank_windows: Number of flank windows (used to set labels)
    :param labels: Line labels
    :param out_dir: Path to output directory
    :param smooth: smooth * len(density) = window_length for SavGol filter
    """

    if smooth > 1.0:
        print('Smooth should be less than 1. Using no smooth')
        smooth = None

    if labels is not None:
        if len(labels) != len(data):
            print('Labels count != data count. Using no labels')
            labels = None

    for context in ['CG', 'CHH', 'CHG']:
        for strand in ['+', '-']:
            line_data = [line_plot_filter(density, context, strand, smooth) for density in data]

            title = f'{context}{strand}'
            draw_line_plot(line_data, flank_windows, labels, title, out_dir)
