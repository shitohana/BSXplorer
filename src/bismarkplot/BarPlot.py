import polars as pl


class BarPlot:
    """
    Base class for Bar Plot data
    """
    def __init__(self, bismark: pl.DataFrame):
        """
        :param bismark: bismark polars.Dataframe from :class:`Bismark`
        """
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
