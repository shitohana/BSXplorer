import polars as pl


def read_genome(
        file: str,
        columns: list     = None,
        flank_length: int = 0,
        min_length: int   = None,
        has_header: bool  = False,
        comment_char: str = '#'
) -> pl.DataFrame:
    """
    Method to **read gff (or any TSV) as reference genome**. Mandatory columns are chr, type, start, end, strand.

    :param has_header: Specify if there is a header
    :param comment_char: Comment character in genome file
    :param file: Path to a file.
    :param columns: Array with columns to use ``[<chr>, <type>, <start>, <end>, <strand>]``.
    :param flank_length: Length of flank regions. If not needed - set zero.
    :param min_length: Filter for length of selected regions. flank_length * 2 by default.
    :return: polars.Dataframe with selected regions ranges.
    """
    if columns is None:
        columns = [0, 2, 3, 4, 6]
    if min_length is None:
        min_length = flank_length * 2

    genes = pl.read_csv(
        file,
        comment_char = comment_char,
        columns      = columns,
        has_header   = has_header,
        separator    = '\t',
        new_columns  = ['chr', 'type', 'start', 'end', 'strand'],
        dtypes       = {'start': pl.Int32, 'end': pl.Int32}
    ).filter(pl.col('type') == 'gene').drop('type')

    if flank_length != 0:
        genes = genes.filter(pl.col('start') > 2000)
    if min_length is not None:
        genes = genes.filter(pl.col('end') - pl.col('start') > 4000)

    genes = (
        genes.lazy()
        # generate flank regions lengths
        .groupby(['chr', 'strand'], maintain_order=True).agg([
            pl.col('start'), pl.col('end'),
            # upstream shift
            (pl.col('start').shift(-1) - pl.col('end')).shift(1)
            .fill_null(flank_length)
            .alias('upstream'),
            # downstream shift
            (pl.col('start').shift(-1) - pl.col('end'))
            .fill_null(flank_length)
            .alias('downstream')
        ])
        .explode(['start', 'end', 'upstream', 'downstream'])
        .with_columns([
            (pl.col('start') - pl.when(
                pl.col('upstream') >= flank_length
            )
             .then(flank_length)
             .otherwise(
                (pl.col('upstream') - pl.col('upstream') % 2) // 2
            )).alias('upstream'),

            (pl.col('end') + pl.when(
                pl.col('downstream') >= flank_length
            )
             .then(flank_length)
             .otherwise(
                (pl.col('downstream') - pl.col('downstream') % 2) // 2
            )).alias('downstream')
        ])
    ).collect()
    return genes
