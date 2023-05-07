import polars as pl


class BoxPlot:
    """
    Base class for Box Plot data
    """
    def __init__(self, bismark: pl.DataFrame):
        """
        :param bismark: bismark polars.Dataframe from :class:`Bismark`
        """
        self.bismark = (
            bismark.lazy()
            .groupby(['chr', 'start', 'context', 'strand'])
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
            .groupby(['context', 'strand'])
            .agg(pl.col('density'))
        ).collect()

    def filter(self, context: str, strand: str = None) -> pl.DataFrame:
        """
        Get filtered data as DataFrame
        :param context: Context to filter
        :param strand: Strand to filter
        :return: Filtered Dataframe
        """

        if strand is not None:
            filter_expr = (pl.col('context') == context) & (pl.col('strand') == strand)
            return self.bismark.filter(filter_expr)
        else:
            filter_expr = pl.col('context') == context
            return self.bismark.groupby('context').agg(pl.sum('density')).filter(filter_expr)



    def filter_density(self, context: str, strand: str = None) -> list:
        """
        Get filtered density as list
        :param context: Context to filter
        :param strand: Strand to filter
        :return: List of density of filtered Dataframe
        """
        if strand is not None:
            filter_expr = (pl.col('context') == context) & (pl.col('strand') == strand)
            return self.bismark.filter(filter_expr)['density'].to_list()[0]
        else:
            filter_expr = pl.col('context') == context
            return self.bismark.groupby('context').agg(pl.sum('density')).filter(filter_expr)['density'].to_list()[0]

