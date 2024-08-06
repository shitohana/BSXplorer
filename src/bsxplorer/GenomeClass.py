from __future__ import annotations

import itertools
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

        self.genome = self.validate(genome)

    def raw(self):
        """
        This method returns raw Genome DataFrame without any
        filtering and ranges manipulation.

        Returns
        -------
        ``pl.DataFrame``
        """
        return self.genome.collect()

    @classmethod
    def validate(cls, genome):
        if "id" not in genome.columns:
            genome = genome.with_columns(id=pl.lit(""))
        if "type" not in genome.columns:
            genome = genome.with_columns(type=pl.lit(""))
        if "strand" not in genome.columns:
            genome = genome.with_columns(strand=pl.lit("."))

        return genome.select(list(cls._schema.keys())).cast(cls._schema)

    _schema = {
        "chr": pl.Utf8,
        "type": pl.Utf8,
        "start": MetageneSchema.position,
        "end": MetageneSchema.position,
        "strand": pl.Utf8,
        "id": pl.Utf8,
    }

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
        read_filters
            Filter annotation by `polars.Expr
            <https://docs.pola.rs/py-polars/html/reference/expressions/index.html>`_

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
            pl.col(cols[chr_col]).alias("chr"),
            (pl.col(cols[type_col]) if type_col is not None else pl.lit(None)).alias("type"),
            pl.col(cols[start_col]).alias("start"),
            pl.col(cols[end_col]).alias("end"),
            (pl.col(cols[strand_col]) if strand_col is not None else pl.lit(".")).alias("strand"),
            (pl.col(cols[id_col]) if id_col is not None else pl.lit("")).alias("id"),
        ]

        genes = (
            genes
            .with_columns(select_cols)
            .select(["chr", "type", "start", "end", "strand", "id"])
            .sort(["chr", "start"])
            .filter(True if read_filters is None else read_filters)
        )

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
        region_type
            Filter annotation by region type from gff.
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
    """
    Class for storing regions aligned to outer set of regions

    Attributes
    ----------
    gene_body
        Regions, which middle coordinate is inside outer region.
    upstream
        Regions, which middle coordinate is in upstream of outer region.
    downstream
        Regions, which middle coordinate is in downstream of outer region.
    intergene
        Regions, which middle coordinate not in flanking regions nor gene body.
    """

    gene_body: pl.DataFrame
    upstream: pl.DataFrame
    downstream: pl.DataFrame
    intergene: pl.DataFrame
    flank_length: int

    def ref_positions(self):
        """
        This function calculates the coordinates normalized with respect to the length of the outer region

        Returns
        -------
            polars.DataFrame for upstream, gene_body and downstream relative positions.
        """

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
        """
        This function calculates the coverage of the metagene by the aligned regions.

        Returns
        -------
            tuple(relative_positions, coverage)
        """

        ref_positions = self.ref_positions()
        pos = ref_positions["ref_pos"].to_numpy()
        cov = ref_positions["coverage"].to_numpy()
        interval_values = (-1 <= pos) & (pos <= 2)
        return pos[interval_values], cov[interval_values]

    def plot_density_mpl(
            self,
            fig_axes: tuple = None,
            flank_windows: int = None,
            body_windows: int = None,
            smooth: int = None,
            norm: bool = False,
            tick_labels: list[str] = None,
            label: str = None,
            **mpl_kwargs
    ):
        """
        Plot coverage, returned by :func:`RegAlignResult.metagene_coverage`

        Parameters
        ----------
        fig_axes
            Tuple of (
            `matplotlib.pyplot.Figure
            <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_,
            `matplotlib.axes.Axes
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_).
            New are created if ``None``
        flank_windows
            Number of windows for flanking regions (set None for no resampling).
        body_windows
            Number of windows for body region (set None for no resampling).
        smooth
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows` params are specified.
        norm
            Should the output plot be normalized by maximum coverage.
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        label
            Label of line on line-plot
        mpl_kwargs
            Keyword arguments for
            `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_

        Returns
        -------
        ``matplotlib.pyplot.Figure``
        """
        fig, axes = plt.subplots() if fig_axes is None else fig_axes
        tick_labels = ["Upstream", "TSS", "Body", "TES", "Downstream"] if tick_labels is None else tick_labels

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

            xticks = [0, flank_windows, flank_windows + body_windows / 2, flank_windows + body_windows,
                      flank_windows * 2 + body_windows]

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

        axes.set_xticks(xticks, labels=tick_labels)
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


def align_regions(
        regions: pl.DataFrame,
        along_regions: pl.DataFrame,
        flank_length: int = 2000
):
    total = []

    DeprecationWarning("This method will be removed in future versions. Please use Enrichment class.")

    # Join by middle
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
            .select(["chr", "start", "mid", "end", pl.col("start_right").alias("areg_start"),
                     pl.col("end_right").alias("areg_end"), "id", "strand"])
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
                (pl.col("strand") != "-") &
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


class EnrichmentResult:
    """
    Class for storing and visualizing enrichment results.

    Warnings
    --------
    This class SHOULD NOT be called directly. To create
    it call :func:`Enrichment.enrich` instead.
    """

    def __init__(
            self,
            aligned: pl.DataFrame,
            enrich_stats: pl.DataFrame,
            is_gff: bool = False,
    ):
        self.aligned = aligned
        self.enrich_stats = enrich_stats
        self._is_gff = is_gff

    def ref_positions(self):
        """
        This method calculates relative normalized
        positions of region start and end to gene region
        respectively. Coordinates (-1, 0) refer to the
        upstream region, [0, 1] refer to the gene body,
        (1, 2) refer to the downstream region.

        Returns
        -------
            pl.DataFrame
        """
        ref_positions = (
            self.aligned.lazy()
            .filter(
                pl.col("type").is_in(["upstream", "gene", "downstream"])
                if self._is_gff
                else True
            )
            .select(["gstart", "gend", "afrag_start", "afrag_end", "strand", "type"])
            .cast({key: pl.Int64 for key in ["gstart", "gend", "afrag_start", "afrag_end"]})
            .with_columns(
                ref_start=(
                    pl.when(pl.col("strand") == "-")
                    .then(pl.col("gend") - pl.col("afrag_end"))
                    .otherwise(pl.col("afrag_start") - pl.col("gstart"))
                ),
                ref_end=(
                    pl.when(pl.col("strand") == "-")
                    .then(pl.col("gend") - pl.col("afrag_start"))
                    .otherwise(pl.col("afrag_end") - pl.col("gstart"))
                ),
                glength=(pl.col("gend") - pl.col("gstart"))
            )
            .with_columns(
                ref_start=pl.col("ref_start") / pl.col("glength"),
                ref_end=pl.col("ref_end") / pl.col("glength"),
            )
            .with_columns(
                ref_start=(
                    pl.when(pl.col("type") == "upstream")
                    .then(pl.col("ref_start") - 1)
                    .when(pl.col("type") == "downstream")
                    .then(pl.col("ref_start") + 1)
                    .otherwise(pl.col("ref_start"))
                ),
                ref_end=(
                    pl.when(pl.col("type") == "upstream")
                    .then(pl.col("ref_end") - 1)
                    .when(pl.col("type") == "downstream")
                    .then(pl.col("ref_end") + 1)
                    .otherwise(pl.col("ref_end"))
                )
            )
            .with_columns([
                pl.concat_list([pl.col("ref_start"), pl.col("ref_end")]).alias("ref_pos"),
                pl.lit([1, -1]).alias("is_start")
            ])
            .explode(["ref_pos", "is_start"])
            .sort("ref_pos")
            .filter(((pl.col("ref_start") < -1) | (pl.col("ref_end") > 2)).not_())
            .with_columns(pl.col("is_start").cum_sum().alias("coverage"))
            .collect()
        )

        return ref_positions

    def metagene_coverage(self):
        """
        This function calculates the coverage of the metagene by the aligned regions.

        Returns
        -------
            tuple(relative_positions, coverage)
        """

        ref_positions = self.ref_positions()
        pos = ref_positions["ref_pos"].to_numpy()
        cov = ref_positions["coverage"].to_numpy()
        interval_values = (-1 <= pos) & (pos <= 2)
        return pos[interval_values], cov[interval_values]

    def plot_enrich_mpl(
            self,
            fig_axes: tuple = None,
            exclude: list = None,
            label: str = "",
            **mpl_kwargs
    ):
        """
        Visualize enrichment results as a scatterplot, where
        x – genomic region type,
        y – enrichment value.

        Parameters
        ----------
        fig_axes
            Tuple of (
            `matplotlib.pyplot.Figure
            <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_,
            `matplotlib.axes.Axes
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_).
            New are created if ``None``

        exclude
            List of names of genomic region types to exclude from
            the final plot. Set None if no exclusion should be performed.

        label
            Label for the points on the scatterplot
            (useful, when several groups of points should
            be bisualized on the same plot).

        mpl_kwargs
            Keyword arguemtnts for plt.scatter function of the
            matplotlib API.

        Returns
        -------
        ``matplotlib.pyplot.Figure``
        """

        exclude = list() if exclude is None else exclude
        fig, axes = plt.subplots() if fig_axes is None else fig_axes

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

        plot_df = self.enrich_stats.filter(pl.col("type").is_in(exclude).not_()).sort("enrichment")

        axes.scatter(plot_df["type"].to_list(), plot_df["enrichment"].to_list(), label=label, marker='o', alpha=.90,
                     s=50, **mpl_kwargs)

        return fig

    def plot_density_mpl(
            self,
            fig_axes: tuple = None,
            flank_windows: int = None,
            body_windows: int = None,
            smooth: int = None,
            norm: bool = False,
            tick_labels: list[str] = None,
            label: str = None,
            **mpl_kwargs
    ):
        """
        Plot coverage, returned by :func:`RegAlignResult.metagene_coverage`

        Parameters
        ----------
        fig_axes
            Tuple of (
            `matplotlib.pyplot.Figure
            <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_,
            `matplotlib.axes.Axes
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_).
            New are created if ``None``
        flank_windows
            Number of windows for flanking regions (set None for no resampling).
        body_windows
            Number of windows for body region (set None for no resampling).
        smooth
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows` params are specified.
        norm
            Should the output plot be normalized by maximum coverage.
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        label
            Label of line on line-plot
        mpl_kwargs
            Keyword arguments for
            `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_

        Returns
        -------
        ``matplotlib.pyplot.Figure``
        """
        fig, axes = plt.subplots() if fig_axes is None else fig_axes
        tick_labels = ["Upstream", "TSS", "Body", "TES", "Downstream"] if tick_labels is None else tick_labels

        pos, cov = self.metagene_coverage()

        goodones = (pos > np.quantile(pos, q=.005)) & (pos < np.quantile(pos, q=.995))
        pos = pos[goodones]
        cov = cov[goodones]

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

            xticks = [0, flank_windows, flank_windows + body_windows / 2, flank_windows + body_windows,
                      flank_windows * 2 + body_windows]

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

        axes.set_xticks(xticks, labels=tick_labels)
        axes.set_title("Сoverage of genes and flanking regions by DMRs")
        axes.set_xlabel('Position')
        axes.set_ylabel('Density')

        axes.axvline(x=xticks[1], linestyle='--', color='k', alpha=.3)
        axes.axvline(x=xticks[3], linestyle='--', color='k', alpha=.3)

        return fig


class Enrichment:
    """
    Class for performing logFC enrichment of specified regions over genome.

    Parameters
    ----------
    regions
        polars.DataFrame of region coordinates, which is
        validated by :func:`Genome.validate`.
        Obligatory columns are: chr (chromosome name), start, end.

    genome
        polars.DataFrame of genome, which is generated by
        :func:`Genome.raw` method of an :class:`Genome` instance.

    flank_length
        Length in bp of flanking regions for genes.
        If no flanking regions should be added, set
        this parameter to 0.
    """

    def __init__(self, regions: pl.DataFrame, genome: pl.DataFrame, flank_length: int = 0):
        self.regions = Genome.validate(regions)
        genome = Genome.validate(genome)
        self.genome = genome if not flank_length else self._add_flank_regions(genome, flank_length)
        self._flank_length = flank_length

    @property
    def _type_lengths(self) -> pl.DataFrame:
        return (
            self.genome
            .group_by("type")
            .agg([(pl.col("end") - pl.col("start")).sum().alias("total")])
        )

    @property
    def _chr_lengths(self) -> pl.DataFrame:
        # Use predefined chromosome lengths if available
        if "region" not in self._type_lengths["type"]:
            return self.genome.group_by("chr").agg([(pl.last("end") - pl.first("start")).alias("length")])
        # Otherwise, take last gene coord for chromosome end
        else:
            return (self.genome.vstack(self.regions).filter(type="region").select(
                ["type", (pl.col("end")).alias("length")]))

    @property
    def _is_gff(self) -> bool:
        return "gene" in self._type_lengths["type"]

    @property
    def _total_rlen(self) -> int:
        return self.regions.with_columns(length=pl.col("end") - pl.col("start"))["length"].sum()

    @staticmethod
    def _add_flank_regions(genome, flank_length) -> pl.DataFrame:
        gene_bodies = Genome(genome.lazy()).gene_body(flank_length=flank_length)
        return pl.concat(
            [
                genome,
                Genome.validate(
                    gene_bodies
                    .select(["chr", pl.col("upstream").alias("start"), pl.col("start").alias("end"), "strand"])
                    .with_columns(
                        type=pl.when(pl.col("strand") == "-").then(pl.lit("downstream")).otherwise(pl.lit("upstream"))
                    )
                ),
                Genome.validate(
                    gene_bodies
                    .select(["chr", pl.col("end").alias("start"), pl.col("downstream").alias("end"), "strand"])
                    .with_columns(
                        type=pl.when(pl.col("strand") == "-").then(pl.lit("upstream")).otherwise(pl.lit("downstream"))
                    )
                )
            ]
        )

    def enrich(self) -> EnrichmentResult:
        """
        This method performs an alignment of regions to the genome,
        calculating genomic coordinates of genome regions and
        user-defined regions intersections, and runs calculates logFC
        enrichment metric.

        Returns
        -------
            :class:`EnrichmentResult`
        """
        if not self._is_gff:
            raise ValueError('"gene" region type must be included in the genome DataFrame!')

        type_lengths = self._type_lengths
        chr_lengths = self._chr_lengths

        # Add intergene type to type_lengths
        type_lengths.extend(
            pl.DataFrame({
                "type": "intergene",
                "total": chr_lengths["length"].sum() - type_lengths.filter(type="gene").row(0)[1]
            }).cast(type_lengths.schema)
        )

        res = []
        # Aligning
        for chrom in chr_lengths["chr"]:
            filtered_genome = self.genome.filter(chr=chrom).sort("start")
            filtered_dmrs = self.regions.filter(chr=chrom)

            gpos = list(filtered_genome[["start", "end"]].iter_rows())
            dpos = list(filtered_dmrs[["start", "end"]].iter_rows())

            for gstart, gend in gpos:
                aligned = []

                i = 0
                while dpos:
                    try:
                        dstart, dend = dpos[i]
                    except IndexError:
                        break
                    if dend < gstart:
                        dpos.pop(0)
                    else:
                        if dend > gstart > dstart:
                            aligned.append((chrom, gstart, gend, dstart, dend, gstart, dend))
                        elif dstart >= gstart and dend <= gend:
                            aligned.append((chrom, gstart, gend, dstart, dend, dstart, dend))
                        elif dstart < gend < dend:
                            aligned.append((chrom, gstart, gend, dstart, dend, dstart, gend))
                        else:
                            break

                        i += 1

                if aligned:
                    res.append(aligned)

        res_df = pl.DataFrame(
            list(itertools.chain(*res)),
            schema={
                "chr": self.genome.schema["chr"],
                "gstart": self.genome.schema["start"],
                "gend": self.genome.schema["end"],
                "dmr_start": self.genome.schema["start"],
                "dmr_end": self.genome.schema["end"],
                "afrag_start": self.genome.schema["start"],
                "afrag_end": self.genome.schema["end"],
            }
        )

        joined = res_df.join(self.genome, left_on=["chr", "gstart", "gend"], right_on=["chr", "start", "end"],
                             how="left")

        len_stats = (
            joined
            .group_by("type")
            .agg([
                (pl.col("afrag_end") - pl.col("afrag_start")).sum().alias("alen"),
            ])
        )

        # Intergene row
        len_stats.extend(
            self.regions
            .join(res_df, right_on=["chr", "dmr_start", "dmr_end"], left_on=["chr", "start", "end"], how="left")
            .filter(pl.col("gstart").is_null())
            .group_by(pl.lit(True))
            .agg((pl.col("end") - pl.col("start")).sum().alias("alen"))
            .with_columns(type=pl.lit("intergene"))
            .select(["type", "alen"])
        )

        len_stats = len_stats.join(type_lengths, on="type")
        # Non-CDS row
        len_stats.extend(
            pl.DataFrame({
                "type": "NCDS",
                "alen": len_stats.filter(type="gene").row(0)[1] - len_stats.filter(type="CDS").row(0)[1],
                "total": len_stats.filter(type="gene").row(0)[2] - len_stats.filter(type="CDS").row(0)[2],
            }).cast(len_stats.schema)
        )

        enrichment = (
            len_stats
            .with_columns(
                enrichment=(
                        (pl.col("alen") / self._total_rlen).log(2) -
                        (pl.col("total") / chr_lengths["length"].sum()).log(2)
                )
            )
        )

        return EnrichmentResult(aligned=joined, enrich_stats=enrichment, is_gff=self._is_gff)
