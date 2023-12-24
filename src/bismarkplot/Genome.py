from __future__ import annotations

import polars as pl
from pathlib import Path
from .utils import MetageneSchema

class Genome:
    def __init__(self, genome: pl.LazyFrame):
        """
        Class for storing and manipulating genome DataFrame.

        Genome Dataframe columns:

        +------+--------+-------+-------+----------+------------+
        | chr  | strand | start | end   | upstream | downstream |
        +======+========+=======+=======+==========+============+
        | Utf8 | Utf8   | Int32 | Int32 | Int32    | Int32      |
        +------+--------+-------+-------+----------+------------+

        :param genome: :class:`pl.LazyFrame` with genome data.
        """
        self.genome = genome

    @classmethod
    def from_custom(cls,
                    file: str | Path,
                    chr_col: int = 0,
                    start_col: int = 1,
                    end_col: int = 2,
                    id_col: int = None,
                    strand_col: int = 5,
                    type_col: int = None,
                    comment_char: str = "#",
                    has_header: bool = False):

        if sum([val is None for val in [chr_col, strand_col, start_col, end_col]]) > 0:
            raise Exception("All position columns need to be specified!")

        genes = (
            pl.scan_csv(
                file,
                comment_char=comment_char,
                has_header=has_header,
                separator='\t'
            )
        )
        cols = genes.columns
        select_cols = [
            pl.col(cols[chr_col]).alias("chr").cast(pl.Utf8),
            pl.col(cols[type_col]).alias("type") if type_col is not None else pl.lit(None).alias("type"),
            pl.col(cols[start_col]).alias("start").cast(MetageneSchema.position),
            pl.col(cols[end_col]).alias("end").cast(MetageneSchema.position),
            pl.col(cols[strand_col]).alias("strand"),
            pl.col(cols[id_col]).alias("id") if id_col is not None else pl.lit("").alias("id"),
        ]

        genes = genes.with_columns(select_cols).drop(cols)

        print(f"Genome read from {file}")
        return cls(genes)

    @classmethod
    def from_gff(cls, file: str):
        """
        Constructor with parameters for default gff file.

        :param file: path to genome.gff.
        """

        id_regex = "^ID=([^;]+)"

        genome = cls.from_custom(file,
                                 0, 3, 4, 8, 6, 2,
                                 "#", False)

        genome.genome = genome.genome.with_columns(
            pl.col("id").str.extract(id_regex)
        )
        return genome

    def all(self, min_length: int = 4000, flank_length: int = 2000) -> pl.DataFrame:
        genes = self.__filter_genes(
            self.genome, None, min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    def gene_body(self, min_length: int = 4000, flank_length: int = 2000) -> pl.DataFrame:
        """
        Filter type == gene from gff.

        :param min_length: minimal length of genes.
        :param flank_length: length of the flanking region.
        :return: :class:`pl.LazyFrame` with genes and their flanking regions.
        """
        genes = self.__filter_genes(
            self.genome, 'gene', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    def exon(self, min_length: int = 100) -> pl.DataFrame:
        """
        Filter type == exon from gff.

        :param min_length: minimal length of exons.
        :return: :class:`pl.LazyFrame` with exons.
        """
        flank_length = 0
        genes = self.__filter_genes(
            self.genome, 'exon', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    def cds(self, min_length: int = 100) -> pl.DataFrame:
        """
        Filter type == CDS from gff.

        :param min_length: minimal length of CDS.
        :return: :class:`pl.LazyFrame` with CDS.
        """
        flank_length = 0
        genes = self.__filter_genes(
            self.genome, 'CDS', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    def near_TSS(self, min_length: int = 4000, flank_length: int = 2000):
        """
        Get region near TSS - upstream and same length from TSS.

        :param min_length: minimal length of genes.
        :param flank_length: length of the flanking region.
        :return: :class:`pl.LazyFrame` with genes and their flanking regions.
        """

        # decided not to use this
        '''
        upstream_length = (
            # when before length is enough
            # we set upstream length to specified
            pl.when(pl.col('upstream') >= flank_length).then(flank_length)
            # when genes are intersecting (current start < previous end)
            # we don't take this as upstream region
            .when(pl.col('upstream') < 0).then(0)
            # when length between genes is not enough for full specified length
            # we divide it into half
            .otherwise((pl.col('upstream') - (pl.col('upstream') % 2)) // 2)
        )
        '''
        upstream_length = flank_length

        gene_type = "gene"
        genes = self.__filter_genes(
            self.genome, gene_type, min_length, flank_length)
        genes = (
            genes
            .groupby(['chr', 'strand'], maintain_order=True).agg([
                pl.col('start'),
                # upstream shift
                (pl.col('start').shift(-1) - pl.col('end')).shift(1)
                .fill_null(flank_length)
                .alias('upstream'),
                pl.col('id')
            ])
            .explode(['start', 'upstream', 'id'])
            .with_columns([
                (pl.col('start') - upstream_length).alias('upstream'),
                (pl.col("start") + flank_length).alias("end")
            ])
            .with_columns(pl.col("end").alias("downstream"))
        ).collect()

        return self.__check_empty(genes)

    def near_TES(self, min_length: int = 4000, flank_length: int = 2000):
        """
        Get region near TES - downstream and same length from TES.

        :param min_length: minimal length of genes.
        :param flank_length: length of the flanking region.
        :return: :class:`pl.LazyFrame` with genes and their flanking regions.
        """

        # decided not to use this
        '''
        downstream_length = (
            # when before length is enough
            # we set upstream length to specified
            pl.when(pl.col('downstream') >= flank_length).then(flank_length)
            # when genes are intersecting (current start < previous end)
            # we don't take this as upstream region
            .when(pl.col('downstream') < 0).then(0)
            # when length between genes is not enough for full specified length
            # we divide it into half
            .otherwise((pl.col('downstream') - pl.col('downstream') % 2) // 2)
        )
        '''
        downstream_length = flank_length

        gene_type = "gene"
        genes = self.__filter_genes(
            self.genome, gene_type, min_length, flank_length)
        genes = (
            genes
            .groupby(['chr', 'strand'], maintain_order=True).agg([
                pl.col('end'),
                # downstream shift
                (pl.col('start').shift(-1) - pl.col('end'))
                .fill_null(flank_length)
                .alias('downstream'),
                pl.col('id')
            ])
            .explode(['end', 'downstream', 'id'])
            .with_columns([
                (pl.col('end') + downstream_length).alias('downstream'),
                (pl.col("end") - flank_length).alias("start")
            ])
            .with_columns(pl.col("start").alias("upstream"))
        ).collect()

        return self.__check_empty(genes)

    def other(self, gene_type: str, min_length: int = 1000, flank_length: int = 100) -> pl.DataFrame:
        """
        Filter by selected type.

        :param gene_type: selected type from gff. Cases need to match.
        :param min_length: minimal length of genes.
        :param flank_length: length of the flanking region.
        :return: :class:`pl.LazyFrame` with genes and their flanking regions.
        """
        genes = self.__filter_genes(
            self.genome, gene_type, min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    @staticmethod
    def __filter_genes(genes, gene_type, min_length, flank_length):
        if gene_type is not None:
            genes = genes.filter(pl.col('type') == gene_type).drop('type')
        else:
            genes = genes.drop("type")

        # filter genes, which start < flank_length
        if flank_length > 0:
            genes = genes.filter(pl.col('start') > flank_length)
        # filter genes which don't pass length threshold
        if min_length > 0:
            genes = genes.filter((pl.col('end') - pl.col('start')) > min_length)

        return genes

    @staticmethod
    def __trim_genes(genes, flank_length) -> pl.LazyFrame:
        # upstream shift
        # calculates length to previous gene on same chr_strand
        length_before = (pl.col('start').shift(-1) - pl.col('end')).shift(1).fill_null(flank_length)
        # downstream shift
        # calculates length to next gene on same chr_strand
        length_after = (pl.col('start').shift(-1) - pl.col('end')).fill_null(flank_length)

        # decided not to use this conditions
        '''
        upstream_length_conditioned = (
            # when before length is enough
            # we set upstream length to specified
            pl.when(pl.col('upstream') >= flank_length).then(flank_length)
            # when genes are intersecting (current start < previous end)
            # we don't take this as upstream region
            .when(pl.col('upstream') < 0).then(0)
            # when length between genes is not enough for full specified length
            # we divide it into half
            .otherwise((pl.col('upstream') - (pl.col('upstream') % 2)) // 2)
        )

        downstream_length_conditioned = (
            # when before length is enough
            # we set upstream length to specified
            pl.when(pl.col('downstream') >= flank_length).then(flank_length)
            # when genes are intersecting (current start < previous end)
            # we don't take this as upstream region
            .when(pl.col('downstream') < 0).then(0)
            # when length between genes is not enough for full specified length
            # we divide it into half
            .otherwise((pl.col('downstream') - pl.col('downstream') % 2) // 2)
        )
        '''

        return (
            genes
            .groupby(['chr', 'strand'], maintain_order=True).agg([
                pl.col('start'),
                pl.col('end'),
                length_before.alias('upstream'),
                length_after.alias('downstream'),
                pl.col('id')
            ])
            .explode(['start', 'end', 'upstream', 'downstream', 'id'])
            .with_columns([
                # calculates length of region
                (pl.col('start') - flank_length).alias('upstream'),
                # calculates length of region
                (pl.col('end') + flank_length).alias('downstream')
            ])
        )

    @staticmethod
    def __check_empty(genes):
        if len(genes) > 0:
            return genes
        else:
            raise Exception(
                "Genome DataFrame is empty. Are you sure input file is valid?")
