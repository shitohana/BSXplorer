from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from pathlib import Path

from matplotlib import pyplot as plt

from .Plots import savgol_line
from .utils import MetageneSchema


class Genome:
    """
    Storing and manipulating annotation data.
    """

    def __init__(self, genome: pl.LazyFrame):
        """
        Constructor for :class:`Genome`. Shouldn't be called directly.

        Parameters
        ----------
        genome
            `polars.LazyFrame <https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/index.html>`_
        """
        self.genome = genome

        if "id" not in genome.columns:
            genome = genome.with_columns(pl.lit("").alias("id"))

    @classmethod
    def from_custom(cls,
                    file: str | Path,
                    chr_col: int = 0,
                    start_col: int = 1,
                    end_col: int = 2,
                    id_col: int = None,
                    strand_col: int | None = None,
                    type_col: int = None,
                    comment_char: str = "#",
                    has_header: bool = False,
                    read_filters: pl.Expr = None
                    ) -> Genome:
        """
        Create :class:`Genome` from custom tab separated file with genomic regions.

        Warnings
        --------

        Index for columns starts from 0!

        Parameters
        ----------
        file
            Path to file.
        chr_col
            Index of chromosome column.
        start_col
            Index of region start column.
        end_col
            Index of region end column.
        id_col
            Index of id column (if exists).
        strand_col
            Index of strand column.
        type_col
            Index of type column (if exists).
        comment_char
            Character for comments in file.
        has_header
            Does file have header.

        Returns
        -------
            Instance of :class:`Genome`
        """

        if any(val is None for val in [chr_col, start_col, end_col]):
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
            pl.col(cols[chr_col]).cast(pl.Utf8).alias("chr"),
            (pl.col(cols[type_col]) if type_col is not None else pl.lit(None)).alias("type"),
            pl.col(cols[start_col]).cast(MetageneSchema.position).alias("start"),
            pl.col(cols[end_col]).cast(MetageneSchema.position).alias("end"),
            (pl.col(cols[strand_col]) if strand_col is not None else pl.lit(".")).alias("strand"),
            (pl.col(cols[id_col]) if id_col is not None else pl.lit("")).alias("id"),
        ]

        genes = genes.with_columns(select_cols).select(["chr", "type", "start", "end", "strand", "id"]).sort(["chr", "start"])

        print(f"Genome read from {file}")
        return cls(genes)

    @classmethod
    def from_gff(cls, file: str | Path) -> Genome:
        """
        Constructor for :class:`Genome` class from .gff file.

        Parameters
        ----------
        file
            Path to .gff file.

        Returns
        -------
            Instance of :class:`Genome`.
        """

        id_regex = "^ID=([^;]+)"

        genome = cls.from_custom(file,
                                 0, 3, 4, 8, 6, 2,
                                 "#", False)

        genome.genome = genome.genome.with_columns(
            pl.col("id").str.extract(id_regex)
        )
        return genome

    def all(self, min_length: int = 0, flank_length: int = 0) -> pl.DataFrame:
        """
        Filter annotation and calculate positions of flanking regions.

        Parameters
        ----------
        min_length
            Region length threshold.
        flank_length
            Length of flanking regions.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.all(min_length = 0)
        shape: (710_650, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬─────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id              │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---             │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str             │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪═════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3631   ┆ 5899   ┆ 1631     ┆ 7899       ┆ gene-AT1G01010  │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …               │
        │ NC_000932.1 ┆ +      ┆ 153878 ┆ 154312 ┆ 151878   ┆ 156312     ┆ cds-NP_051123.1 │
        │ NC_000932.1 ┆ ?      ┆ 69611  ┆ 140650 ┆ 67611    ┆ 142650     ┆ rna-ArthCp047   │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴─────────────────┘
        """

        genes = self.__filter_genes(
            self.genome, None, min_length, flank_length).collect()
        genes = self.__trim_genes(genes, flank_length)
        return self.__check_empty(genes)

    def gene_body(self, min_length: int = 0, flank_length: int = 2000) -> pl.DataFrame:
        """
        Filter annotation by type == gene and calculate positions of flanking regions.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

        Parameters
        ----------
        min_length
            Region length threshold.
        flank_length
            Length of flanking regions.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.gene_body(min_length=2000, flank_length=2000)
        shape: (14_644, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id             │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---            │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str            │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3631   ┆ 5899   ┆ 1631     ┆ 7899       ┆ gene-AT1G01010 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …              │
        │ NC_000932.1 ┆ +      ┆ 104691 ┆ 107500 ┆ 102691   ┆ 109500     ┆ gene-ArthCr087 │
        │ NC_000932.1 ┆ +      ┆ 141485 ┆ 143708 ┆ 139485   ┆ 145708     ┆ gene-ArthCp086 │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴────────────────┘

        """
        genes = self.__filter_genes(self.genome, 'gene', min_length, flank_length).collect()
        genes = self.__trim_genes(genes, flank_length)
        return self.__check_empty(genes)

    def exon(self, min_length: int = 100) -> pl.DataFrame:
        """
        Filter annotation by type == exon and length threshold.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

        Parameters
        ----------
        min_length
            Region length threshold.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.exon(min_length=200)
        shape: (132_822, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬────────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id                 │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---                │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str                │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪════════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3631   ┆ 3913   ┆ 3631     ┆ 3913       ┆ exon-NM_099983.2-1 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …                  │
        │ NC_000932.1 ┆ +      ┆ 152806 ┆ 153195 ┆ 152806   ┆ 153195     ┆ exon-ArthCp085-1   │
        │ NC_000932.1 ┆ +      ┆ 153878 ┆ 154312 ┆ 153878   ┆ 154312     ┆ exon-ArthCp085-2   │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴────────────────────┘
        """
        flank_length = 0
        genes = self.__filter_genes(
            self.genome, 'exon', min_length, flank_length).collect()
        genes = self.__trim_genes(genes, flank_length)
        return self.__check_empty(genes)

    def cds(self, min_length: int = 100) -> pl.DataFrame:
        """
        Filter annotation by type == CDS and length threshold.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

        Parameters
        ----------
        min_length
            Region length threshold.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.cds(min_length=200)
        shape: (81_837, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬─────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id              │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---             │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str             │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪═════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3996   ┆ 4276   ┆ 3996     ┆ 4276       ┆ cds-NP_171609.1 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …               │
        │ NC_000932.1 ┆ +      ┆ 152806 ┆ 153195 ┆ 152806   ┆ 153195     ┆ cds-NP_051123.1 │
        │ NC_000932.1 ┆ +      ┆ 153878 ┆ 154312 ┆ 153878   ┆ 154312     ┆ cds-NP_051123.1 │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴─────────────────┘
        """
        flank_length = 0
        genes = self.__filter_genes(
            self.genome, 'CDS', min_length, flank_length).collect()
        genes = self.__trim_genes(genes, flank_length)
        return self.__check_empty(genes)

    def near_TSS(self, min_length: int = 4000, flank_length: int = 2000):
        """
        Filter annotation by type == gene and calculate positions of near TSS regions.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

            Use down_windows=0 for Metagene.

        Parameters
        ----------
        min_length
            Region length threshold.
        flank_length
            Length of flanking regions.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.near_TSS(min_length=2000, flank_length=2000)
        shape: (14_644, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id             │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---            │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str            │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3631   ┆ 5631   ┆ 1631     ┆ 5631       ┆ gene-AT1G01010 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …              │
        │ NC_000932.1 ┆ +      ┆ 104691 ┆ 106691 ┆ 102691   ┆ 106691     ┆ gene-ArthCr087 │
        │ NC_000932.1 ┆ +      ┆ 141485 ┆ 143485 ┆ 139485   ┆ 143485     ┆ gene-ArthCp086 │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴────────────────┘


        """

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
            .groupby(['chr', 'strand'], maintain_order=True)
            .agg([
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
            .select(['chr', 'strand', 'start', 'end', 'upstream', 'downstream', 'id'])
        ).collect()

        return self.__check_empty(genes)

    def near_TES(self, min_length: int = 4000, flank_length: int = 2000):
        """
        Filter annotation by type == gene and calculate positions of near TES regions.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

            Use down_windows=0 for Metagene.

        Parameters
        ----------
        min_length
            Region length threshold.
        flank_length
            Length of flanking regions.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.near_TES(min_length=2000, flank_length=2000)
        shape: (14_644, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id             │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---            │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str            │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3899   ┆ 5899   ┆ 3899     ┆ 7899       ┆ gene-AT1G01010 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …              │
        │ NC_000932.1 ┆ +      ┆ 105500 ┆ 107500 ┆ 105500   ┆ 109500     ┆ gene-ArthCr087 │
        │ NC_000932.1 ┆ +      ┆ 141708 ┆ 143708 ┆ 141708   ┆ 145708     ┆ gene-ArthCp086 │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴────────────────┘
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
            .select(['chr', 'strand', 'start', 'end', 'upstream', 'downstream', 'id'])
        ).collect()

        return self.__check_empty(genes)

    def other(self, region_type: str, min_length: int = 1000, flank_length: int = 100) -> pl.DataFrame:
        """
        Filter annotation by selected type and calculate positions of nflanking regions.

        Warnings
        --------
            This method will have empty output, if type is not specified in input file.

            If no flanking regions are needed – enter flank_length=0.

        Parameters
        ----------
        min_length
            Region length threshold.
        flank_length
            Length of flanking regions.

        Returns
        -------
            Return :class:`polars.DataFrame` for downstream usage.

        Examples
        --------

        >>> path = "/path/to/genome.gff"
        >>> genome = genome.from_gff(path)
        >>> genome.other("region")
        shape: (45_888, 7)
        ┌─────────────┬────────┬────────┬────────┬──────────┬────────────┬─────────────────┐
        │ chr         ┆ strand ┆ start  ┆ end    ┆ upstream ┆ downstream ┆ id              │
        │ ---         ┆ ---    ┆ ---    ┆ ---    ┆ ---      ┆ ---        ┆ ---             │
        │ str         ┆ str    ┆ u64    ┆ u64    ┆ u64      ┆ u64        ┆ str             │
        ╞═════════════╪════════╪════════╪════════╪══════════╪════════════╪═════════════════╡
        │ NC_003070.9 ┆ +      ┆ 3631   ┆ 5899   ┆ 3531     ┆ 5999       ┆ rna-NM_099983.2 │
        │ …           ┆ …      ┆ …      ┆ …      ┆ …        ┆ …          ┆ …               │
        │ NC_000932.1 ┆ +      ┆ 152806 ┆ 154312 ┆ 152706   ┆ 154412     ┆ rna-ArthCp085   │
        │ NC_000932.1 ┆ ?      ┆ 69611  ┆ 140650 ┆ 69511    ┆ 140750     ┆ rna-ArthCp047   │
        └─────────────┴────────┴────────┴────────┴──────────┴────────────┴─────────────────┘

        """
        genes = self.__filter_genes(
            self.genome, region_type, min_length, flank_length).collect()
        genes = self.__trim_genes(genes, flank_length)
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
    def __trim_genes(genes, flank_length) -> pl.DataFrame:
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
        if (genes["end"] < genes["start"]).sum() > 0:
            forward = (
                genes.filter(pl.col("start") <= pl.col("end"))
                .with_columns(pl.lit("+").alias("strand"))

            )
            reverse = (
                genes.filter(pl.col("start") > pl.col("end"))
                .with_columns(pl.lit("-").alias("strand"))
                .rename({"start": "end", "end": "start"})
                .select(forward.columns)
            )

            genes = pl.concat([forward, reverse]).sort(["chr", "start"])

        trimmed = (
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

        return trimmed

    @staticmethod
    def __check_empty(genes):
        if len(genes) > 0:
            return genes
        else:
            raise Exception("Genome DataFrame is empty. Are you sure input file is valid?")


@dataclass
class RegAlignResult:
    gene_body: pl.DataFrame
    upstream: pl.DataFrame
    downstream: pl.DataFrame
    intergene: pl.DataFrame
    flank_length: int

    def ref_positions(self):
        def ref_pos_expr(for_column: str):
            expr = (
                pl.when(
                    pl.col(for_column) < pl.col("areg_start")
                ).then(
                    (pl.col("areg_start") - pl.col(for_column)) / self.flank_length * -1
                ).when(
                    pl.col(for_column) > pl.col("areg_end")
                ).then(
                    (pl.col(for_column) - pl.col("areg_end")) / self.flank_length + 1
                ).otherwise(
                    (pl.col(for_column) - pl.col("areg_start")) / pl.col("length")
                )
            )
            return expr

        ref_positions = (
            self.near_reg()
            .cast(dict(areg_start=pl.Int64, areg_end=pl.Int64, start=pl.Int64, end=pl.Int64))
            .with_columns([
                (pl.col("areg_end") - pl.col("areg_start")).alias("length"),
                pl.when(pl.col("strand") == "-").then(0 - pl.col("end")).otherwise(pl.col("start")).alias("start"),
                pl.when(pl.col("strand") == "-").then(0 - pl.col("areg_end")).otherwise(pl.col("areg_start")).alias(
                    "areg_start"),
                pl.when(pl.col("strand") == "-").then(0 - pl.col("start")).otherwise(pl.col("end")).alias("end"),
                pl.when(pl.col("strand") == "-").then(0 - pl.col("areg_start")).otherwise(pl.col("areg_end")).alias(
                    "areg_end"),
            ])
            .with_columns([
                ref_pos_expr("start").alias("ref_start"),
                ref_pos_expr("end").alias("ref_end"),
            ])
            # .select(["ref_start", "ref_end"])
            .with_columns([
                pl.concat_list([pl.col("ref_start"), pl.col("ref_end")]).alias("ref_pos"),
                pl.lit([1, -1]).alias("is_start")
            ])
            .explode(["ref_pos", "is_start"])
            .sort("ref_pos")
            .with_columns(pl.col("is_start").cum_sum().alias("coverage"))
        )
        return ref_positions

    def metagene_coverage(self):

        ref_positions = self.ref_positions()
        pos = ref_positions["ref_pos"].to_numpy()
        cov = ref_positions["coverage"].to_numpy()
        interval_values = (-1 <= pos) & (pos <= 2)
        return pos[interval_values], cov [interval_values]

    def plot_density_mpl(
            self,
            fig_axes: tuple = None,
            flank_windows: int = None,
            body_windows: int = None,
            smooth: int = None,
            norm: bool = False,
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            label: str = None,
            **mpl_kwargs
    ):
        fig, axes = plt.subplots() if fig_axes is None else fig_axes
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        pos, cov = self.metagene_coverage()
        xticks = [-1, 0, .5, 1, 2]

        if flank_windows is not None and body_windows is not None:
            resampled_pos = []
            resampled_cov = []

            for start, stop, windows in zip([-1, 0, 1], [0, 1, 2], [flank_windows, body_windows, flank_windows]):
                points, step = np.linspace(start, stop, windows + 1, retstep=True)
                indexes = [(start <= pos) & (pos < (start + step)) for start in points[:-1]]
                resampled_pos.append(points[:-1])
                resampled_cov.append(np.array([cov[index].mean() if cov[index].size != 0 else 0 for index in indexes]))

            cov = np.concatenate(resampled_cov)
            pos = np.array(list(range(len(cov))))

            valid = np.isnan(cov)
            cov = np.interp(pos, pos[~valid], cov[~valid]) if (~valid).sum() > 1 else np.full_like(pos, 0)
            cov = savgol_line(cov, smooth)

            xticks = [0, flank_windows, flank_windows + body_windows / 2, flank_windows + body_windows, flank_windows * 2 + body_windows]

        else:
            new_cov = []
            new_pos = []
            for p_prev, p_next, c_prev, c_next in zip(pos[:-1], pos[1:], cov[:-1], cov[1:]):
                if c_next > c_prev:
                    new_pos += [p_prev, p_prev]
                    new_cov += [c_prev, c_next]
                elif c_next < c_prev:
                    new_pos += [p_next, p_next]
                    new_cov += [c_prev, c_next]
                else:
                    new_pos += [p_prev]
                    new_cov += [c_prev]
            pos = np.array(new_pos)
            cov = np.array(new_cov)

        if norm and cov.size > 0:
            cov = cov / cov.max()

        axes.plot(pos, cov, label=label, **mpl_kwargs)

        axes.set_xticks(xticks, labels=[minor_labels[0], major_labels[0], minor_labels[1], major_labels[1], minor_labels[2]])
        axes.set_title("Сoverage of genes and flanking regions by DMRs")
        axes.set_xlabel('Position')
        axes.set_ylabel('Density')

        axes.axvline(x=xticks[1], linestyle='--', color='k', alpha=.3)
        axes.axvline(x=xticks[3], linestyle='--', color='k', alpha=.3)

        return fig

    def near_reg(self) -> pl.DataFrame:
        return pl.concat([self.upstream, self.gene_body, self.downstream])

    def areg_ids(self) -> list[str]:
        return self.near_reg()["id"].unique().to_list()

    def areg_stats(self) -> pl.DataFrame:
        return (
            self.near_reg()
            .group_by(["chr", "areg_start", "areg_end"])
            .agg([
                pl.first("id"),
                pl.len().alias("count"),
                (pl.col("end") - pl.col("start")).mean().alias("mean_length")
            ])
            .sort("count", descending=True)
        )


def align_regions(regions: pl.DataFrame, along_regions: pl.DataFrame, flank_length: int = 2000):
    total = []
    # Join to middle
    for chrom in regions["chr"].unique():
        chr_left = (
            regions.filter(chr=chrom)
            .with_columns(((pl.col('end') + pl.col('start')) / 2).floor().cast(pl.UInt32).alias('mid'))
            .sort('mid')
        )
        chr_right = (
            along_regions.filter(chr=chrom)
            .with_columns(((pl.col('end') + pl.col('start')) / 2).floor().cast(pl.UInt32).alias('mid'))
            .sort('mid')
        )

        joined = (
            chr_left
            .join_asof(chr_right, on='mid', strategy='nearest')
            .with_columns(((pl.col('end') + pl.col('start')) / 2).floor().cast(pl.UInt32).alias('mid'))
            .select(["chr", "start", "mid", "end", pl.col("start_right").alias("areg_start"), pl.col("end_right").alias("areg_end"), "id", "strand"])
        )

        total.append(joined)

    total = pl.concat(total)

    in_gb = total.filter(
        (pl.col("mid") >= pl.col("areg_start")) &
        (pl.col("mid") <= pl.col("areg_end"))
    )
    in_up = total.filter(
        (
            (pl.col("strand") != "-") &
            (pl.col("end") >= (pl.col("areg_start") - flank_length)) &
            (pl.col("mid") < pl.col("areg_start"))
        ) |
        (
            (pl.col("strand") == "-") &
            (pl.col("mid") > pl.col("areg_end")) &
            (pl.col("start") <= (pl.col("areg_end") + flank_length))
        )
    )
    in_down = total.filter(
        (
                (pl.col("strand") != "+") &
                (pl.col("mid") > pl.col("areg_end")) &
                (pl.col("start") <= (pl.col("areg_end") + flank_length))
        ) |
        (
                (pl.col("strand") == "-") &
                (pl.col("end") >= (pl.col("areg_start") - flank_length)) &
                (pl.col("mid") < pl.col("areg_start"))
        )
    )
    intergene = total.filter(
        (pl.col("end") < (pl.col("areg_start") - flank_length)) |
        (pl.col("start") > (pl.col("areg_end") + flank_length))
    )

    return RegAlignResult(in_gb, in_up, in_down, intergene, flank_length)
