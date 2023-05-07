from multiprocessing import cpu_count
import polars as pl


def read_bismark_batches(
        file: str,
        genome: pl.DataFrame,
        flank_windows: int   = 500,
        gene_windows: int    = 2000,
        batch_size: int      = 10 ** 6,
        cpu: int             = cpu_count()
) -> pl.DataFrame:
    """
    Method to read Bismark **genomeWide cytosine report**

    :param cpu: How many cores to use. Uses every physical core by default
    :param file: path to bismark genomeWide report
    :param genome: polars.Dataframe with gene ranges
    :param flank_windows: Number of windows flank regions to split
    :param gene_windows: Number of windows gene regions to split
    :param batch_size: Number of rows to read by one CPU core
    :return: | polars.Dataframe with windowed cytosine report with columns
             | ``[<chr>, <strand>, <start>, <context>, <fragment>, <density>]``
             | (density = count_methylated / count_unmethylated)
    """
    with pl.StringCache():
        total = None
        genome = genome.with_columns(
            [
                pl.col('strand').cast(pl.Categorical),
                pl.col('chr').cast(pl.Categorical)
            ]
        )
        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand', 'count_m', 'count_um', 'context'],
            columns=[0, 1, 2, 3, 4, 5],
            batch_size=batch_size
        )
        batches = bismark.next_batches(cpu)
        while batches:
            for df in batches:
                df = (
                    df.lazy()
                    .filter((pl.col('count_m') + pl.col('count_um') != 0))
                    # calculate density for each cytosine
                    .with_columns([
                        pl.col('position').cast(pl.Int32),
                        pl.col('count_m').cast(pl.Int8),
                        pl.col('count_um').cast(pl.Int8),
                        pl.col('chr').cast(pl.Categorical),
                        pl.col('strand').cast(pl.Categorical),
                        pl.col('context').cast(pl.Categorical),
                        ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density')
                    ])
                    # delete redundant columns
                    .drop(['count_m', 'count_um'])
                    # join on nearest start for every row
                    .sort('position')
                    # join on nearest start for every row
                    .join_asof(
                        genome.lazy().sort('upstream'),
                        left_on='position', right_on='upstream', by=['chr', 'strand']
                    )
                    # limit by end of gene
                    .filter(pl.col('position') <= pl.col('downstream'))
                    .with_columns(
                        # upstream
                        pl.when(
                            pl.col('position') < pl.col('start')
                        ).then(
                            (((pl.col('position') - pl.col('upstream')) /
                              (pl.col('start') - pl.col('upstream'))
                              ) * flank_windows).floor()
                        )
                        # gene body
                        .when(
                            (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
                        ).then(
                            (((pl.col('position') - pl.col('start'))
                              / (pl.col('end') - pl.col('start') + 1e-10)
                              ) * gene_windows).floor() + flank_windows
                        )
                        # downstream
                        .when(
                            (pl.col('position') > pl.col('end'))
                        ).then(
                            (((pl.col('position') - pl.col('end'))
                              / (pl.col('downstream') - pl.col('end') + 1e-10)
                              ) * flank_windows).floor() + flank_windows + gene_windows
                        )
                        .cast(pl.Int32).alias('fragment')
                    )
                    .groupby(
                        by=['chr', 'strand', 'start', 'context', 'fragment']
                    )
                    .agg([
                        pl.sum('density').alias('sum'),
                        pl.count('density').alias('count')
                    ])
                    .drop_nulls(subset=['sum'])
                ).collect()
                if total is None:
                    total = df
                else:
                    total = total.extend(df)
            batches = bismark.next_batches(cpu)
    return total
