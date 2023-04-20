import polars as pl


class BoxPlot:
    def __init__(self, bismark: pl.DataFrame):
        self.bismark = (
            bismark.lazy()
            .groupby(['chr', 'start', 'context'])
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
            .groupby(['context'])
            .agg(pl.col('density'))
        ).collect()

    def filter(self, context: str, strand: str = None) -> pl.DataFrame:
        filter_expr = pl.col('context') == context
        if strand is not None:
            filter_expr = (pl.col('context') == context) & (pl.col('strand') == strand)
        return self.bismark.filter(filter_expr)

    def filter_density(self, context: str, strand: str = None) -> list:
        filter_expr = pl.col('context') == context
        if strand is not None:
            filter_expr = (pl.col('context') == context) & (pl.col('strand') == strand)
        return self.bismark.filter(filter_expr['density'].to_list()[0])


def boxplot_data(bismark: pl.DataFrame):
    bismark = (
        bismark.lazy()
        .groupby(['chr', 'start', 'context'])
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
        .groupby(['context'])
        .agg(pl.col('density'))
    ).collect()

    return bismark
