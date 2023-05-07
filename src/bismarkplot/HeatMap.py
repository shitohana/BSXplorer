import numpy as np
import polars as pl


class HeatMap:
    """
    Base class for Heat Map data
    """
    def __init__(self, bismark: pl.DataFrame):
        """
        :param bismark: bismark polars.Dataframe from :class:`Bismark`
        """

        sort_expr = pl.col('sum').ceil().sum() / pl.col('count').sum()

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

