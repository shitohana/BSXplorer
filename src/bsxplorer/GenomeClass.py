from __future__ import annotations

import polars as pl
from pathlib import Path

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
                    has_header: bool = False) -> Genome:
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
            self.genome, None, min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
        return self.__check_empty(genes)

    def gene_body(self, min_length: int = 4000, flank_length: int = 2000) -> pl.DataFrame:
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
        genes = self.__filter_genes(
            self.genome, 'gene', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
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
            self.genome, 'exon', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
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
            self.genome, 'CDS', min_length, flank_length)
        genes = self.__trim_genes(genes, flank_length).collect()
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
            self.genome, region_type, min_length, flank_length)
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
