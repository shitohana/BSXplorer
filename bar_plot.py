import polars as pl


class BarPlot:
    def __init__(self, bismark: pl.DataFrame):
        self.bismark = (
            bismark.lazy()
            .groupby(['chr', 'start', 'context'])
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
            .groupby(['context'])
            .agg(
                pl.mean('density').alias('density')
            )
        ).collect()