import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import colormaps
from matplotlib.axes import Axes


class HeatMap:
    def __init__(self, bismark: pl.DataFrame, rescale: float = None, sort_type: int = 'count'):
        if sort_type == 'count':
            sort_expr = pl.col('sum').ceil().sum() / pl.col('count').sum()
        else:
            sort_expr = pl.col('sum').sum() / pl.col('count').sum()

        if rescale is not None:
            bismark = bismark.with_columns(
                (pl.col('fragment') * rescale).floor().cast(pl.Int32).alias('fragment')
            )

        density = (
            # sort
            bismark.lazy()
            .groupby(['chr', 'start', 'context', 'strand'])
            .agg(
                pl.col('fragment'), pl.col('sum'), pl.col('count'),
                sort_expr.alias('sort')
            )
        )

        self.bismark = density

    def filter(self, context: str = 'CG', strand: str = '+', resolution: int = 100) -> np.ndarray:
        density = self.bismark.filter(
            (pl.col('context') == context) & (pl.col('strand') == strand)
        )

        density = (
            density.lazy()
            .sort('sort', descending=True)
            # add row count
            .with_row_count(name='row')
            # round row count
            .with_columns(
                (pl.col('row') / (pl.col('row').max() + 1) * resolution).floor().alias('row').cast(pl.Int16)
            )
            .explode(['fragment', 'sum', 'count'])
            # calc sum count for row|fragment
            .groupby(['row', 'fragment', 'context', 'strand'])
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
        )

        # join with template
        fragments = density.max().collect()['fragment'].to_list()[0] + 1

        density = (
            # template
            density
            .groupby(['context', 'strand'])
            .agg(
                pl.arange(0, resolution, dtype=density.schema['row'])
                .alias('row')
            )
            .explode('row')
            .groupby(['context', 'strand', 'row'], maintain_order=True)
            .agg(
                pl.arange(0, fragments, dtype=density.schema['fragment'])
                .alias('fragment')
            )
            .explode('fragment')
            # join with orig
            .join(
                density,
                on=['context', 'strand', 'row', 'fragment'],
                how='left'
            )
            .fill_null(0)
            .sort(['row', 'fragment'])
        ).collect()

        density = np.array(
            density.groupby('row', maintain_order=True).agg(pl.col('density'))['density'].to_list(), dtype=np.float32
        )

        if strand == '-':
            density = np.fliplr(density)

        return density

    def draw(
            self,
            axes: Axes = None,
            context: str = 'CG',
            strand: str = '+',
            resolution: int = 100,
            vmin: float = None,
            vmax: float = None,
            flank_windows=0,
            label: list = None,
            data: np.ndarray = None
    ):
        if axes is None:
            _, axes = plt.subplots()

        if data is None:
            data = self.filter(context, strand, resolution)

        image = axes.imshow(
            data,
            interpolation="nearest",
            aspect='auto',
            cmap=colormaps['cividis'],
            vmin=vmin, vmax=vmax,
            label=label
        )
        x_ticks = [flank_windows - 1, len(data[0][0]) - flank_windows]
        x_labels = ['TSS', 'TES']
        axes.set(
            xticks=x_ticks,
            xticklabels=x_labels,
            yticks=[],
            ylabel='Methylation density',
            xlabel='Position'
        )
        for tick in x_ticks:
            axes.axvline(tick, linestyle='--', color='w', alpha=.3)
        plt.colorbar(image, ax=axes)
