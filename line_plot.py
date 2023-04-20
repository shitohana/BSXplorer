import polars as pl
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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
            axes: Axes = None,
            context: str = 'CG',
            strand: str = '+',
            smooth: float = .05,
            label: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
    ):
        if axes is None:
            _, axes = plt.subplots()
        data = self.filter(context, strand)
        if smooth:
            data = signal.savgol_filter(data, int(len(data) * smooth), 3, mode='nearest')
        x = np.arange(len(data))
        data = data * 100  # convert to percents
        axes.plot(x, data, label=label, linestyle=linestyle, linewidth=linewidth)
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')
