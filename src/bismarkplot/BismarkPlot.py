from __future__ import annotations

import multiprocessing
import os
import re
import typing
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pyarrow
import pyarrow.parquet as pq
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame
from pyreadr import write_rds
from scipy.signal import savgol_filter

from .ArrowReaders import CsvReader, BismarkOptions, ParquetReader
from .SeqMapper import Mapper, Sequence
from .Base import BismarkBase, BismarkFilesBase, PlotBase
from .Clusters import Clustering
from .utils import remove_extension, prepare_labels, interval, MetageneSchema
from .GenomeClass import Genome

pl.enable_string_cache(True)


class Metagene(BismarkBase):
    """
    Stores Metagene data.
    """
    def __init__(self, bismark_df: pl.DataFrame, **kwargs):

        super().__init__(bismark_df, **kwargs)

    @classmethod
    def from_bismark(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            block_size_mb: int = 100,
            use_threads: bool = True,
            sumfunc: str = "mean"
    ):
        """
        Constructor for Metagene class from Bismark ``coverage2cytosine`` report.

        Parameters
        ----------
        file
            Path to bismark genomeWide report.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        sumfunc
            Summary function to calculate density for window with.

        Returns
        -------
        Metagene

        Examples
        --------
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/bismark.CX_report.txt'
        >>> metagene = Metagene.from_bismark(path, genome, up_windows=500, body_windows=1000, down_windows=500)
        """

        up_windows, body_windows, down_windows = cls.__check_windows(up_windows, body_windows, down_windows)

        file = Path(file).expanduser().absolute()
        cls.__check_exists(file)

        print("Initializing CSV reader.")

        pool = pyarrow.default_memory_pool()
        block_size = (1024 ** 2) * block_size_mb
        reader = CsvReader(file, BismarkOptions(use_threads=use_threads, block_size=block_size), memory_pool=pool)
        pool.release_unused()

        print("Reading Bismark report from", file.absolute())

        file_size = os.stat(file).st_size
        read_batches = 0

        if isinstance(genome, Genome): raise TypeError("Genome class need to be converted into DataFrame (e.g. Genome.gene_body()).")
        def mutate(batch):
            return (
                pl
                .from_arrow(batch)
                .filter((pl.col('count_m') + pl.col('count_um') != 0))
                .with_columns(
                    ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density')
                    .cast(MetageneSchema.sum)
                )
            )

        bismark_df = None
        pl.enable_string_cache()
        for batch in reader:
            df = cls.__process_batch(batch, genome, mutate, up_windows, body_windows, down_windows, sumfunc)

            if bismark_df is None: bismark_df = df
            else: bismark_df = bismark_df.extend(df)

            read_batches += 1
            cls.__print_batch_stat(read_batches * block_size, file_size, bismark_df.estimated_size())

        cls.__print_batch_stat(file_size, file_size, bismark_df.estimated_size())
        print("\nDONE")

        return cls(bismark_df,
                   upstream_windows=up_windows,
                   gene_windows=body_windows,
                   downstream_windows=down_windows)

    @classmethod
    def from_parquet(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            sumfunc: str = "mean",
            use_threads=True
    ):
        """
        Constructor for Metagene class from converted ``.bedGraph`` or ``.cov`` (via :class:`Mapper`).

        Parameters
        ----------
        file
            Path to converted ``.parquet`` report.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        sumfunc
            Summary function to calculate density for window with.

        Returns
        -------
        Metagene

        Warnings
        --------
        If ``.parquet`` file is created with :meth:`BinomialData.preprocess` use :meth:`Metagene.from_binom` instead.

        Examples
        --------

        >>> save_name = "preprocessed.parquet"
        >>> sequence = Sequence.from_fasta("path/to/sequence.fasta")
        >>> Mapper.coverage("/path/to/report.cov", sequence, delete=False, name=save_name)
        >>> # Now our converted coverage file saved to preprocessed.parquet
        >>>
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>> metagene = Metagene.from_parquet(save_name, genome, up_windows=500, body_windows=1000, down_windows=500)
        """
        up_windows, body_windows, down_windows = cls.__check_windows(up_windows, body_windows, down_windows)

        file = Path(file).expanduser().absolute()
        cls.__check_exists(file)

        bismark_df = None

        # initialize batched reader
        reader = ParquetReader(file.absolute(), use_threads=use_threads)

        # batch approximation
        file_size = os.stat(file).st_size
        num_row_groups = len(reader)
        read_row_groups = 0

        if isinstance(genome, Genome): raise TypeError("Genome class need to be converted into DataFrame (e.g. Genome.gene_body()).")
        print(f"Reading from {file}")

        def mutate(batch):
            return (
                batch
                .from_arrow(df)
                .filter(pl.col("count_total") != 0)
                .with_columns(
                    (pl.col('count_m') / pl.col('count_total')).alias('density').cast(MetageneSchema.sum)
                )
                .drop("count_total")
            )

        for df in reader:
            df = cls.__process_batch(df, genome, mutate, up_windows, body_windows, down_windows, sumfunc)

            if bismark_df is None: bismark_df = df
            else: bismark_df = bismark_df.extend(df)

            read_row_groups += 1
            cls.__print_batch_stat((int(read_row_groups / num_row_groups) * file_size), file_size, bismark_df.estimated_size())

        cls.__print_batch_stat(file_size, file_size, bismark_df.estimated_size())
        print("DONE")

        return cls(bismark_df,
                   upstream_windows=up_windows,
                   gene_windows=body_windows,
                   downstream_windows=down_windows)

    @classmethod
    def from_binom(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            p_value: float = .05,
            use_threads=True
    ):
        """
        Constructor for Metagene class from :meth:`BinomialData.preprocess` ``.parquet`` file.

        Only ``"mean"`` summary function is supported for construction :class:`Metagene` from binom data.

        Parameters
        ----------
        p_value
            P-value of cytosine methylation for it to be considered methylated.
        file
            Path to preprocessed `.parquet` file.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`

        Returns
        -------
        Metagene

        Examples
        --------
        >>> save_name = "preprocessed.parquet"
        >>> BinomialData.preprocess("path/to/bismark.CX_report.txt", report_type="bismark", name=save_name)
        >>>
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>> metagene = Metagene.from_binom(save_name, genome, up_windows=500, body_windows=1000, down_windows=500)
        """
        up_windows, body_windows, down_windows = cls.__check_windows(up_windows, body_windows, down_windows)

        file = Path(file).expanduser().absolute()
        cls.__check_exists(file)

        bismark_df = None

        # initialize batched reader
        reader = ParquetReader(file.absolute(), use_threads=use_threads)

        # batch approximation
        file_size = file.stat().st_size
        num_row_groups = len(reader)
        read_row_groups = 0

        if isinstance(genome, Genome): raise TypeError("Genome class need to be converted into DataFrame (e.g. Genome.gene_body()).")
        print(f"Reading from {file}")

        def mutate(batch):
            return (
                pl.from_arrow(batch)
                .with_columns((pl.col("p_value") < p_value).cast(pl.Float32).alias("density"))
                .drop("count_total")
            )

        for df in reader:
            df = cls.__process_batch(df, genome, mutate, up_windows, body_windows, down_windows, "mean")

            if bismark_df is None:
                bismark_df = df
            else:
                bismark_df = bismark_df.extend(df)

            read_row_groups += 1

        cls.__print_batch_stat(file_size, file_size, bismark_df.estimated_size())
        print("DONE")

        return cls(bismark_df,
                   upstream_windows=up_windows,
                   gene_windows=body_windows,
                   downstream_windows=down_windows)

    @classmethod
    def from_bedGraph(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            sequence: str | Path,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            sumfunc: str = "mean",
            batch_size: int = 10**6,
            cpu: int = multiprocessing.cpu_count(),
            skip_rows: int = 1,
            save_preprocessed: str = None,
            temp_dir: str = "./"
    ):
        """
        Constructor for Metagene class from ``.bedGraph`` file.

        Parameters
        ----------
        file
            Path to ``.bedGraph`` file.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        sequence
            Path to FASTA genome sequence file.
        batch_size
            How many rows to read simultaneously.
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        sumfunc
            Summary function to calculate density for window with.
        temp_dir
            Directory for temporary files.
        save_preprocessed
            Does preprocessed file need to be saved
        skip_rows
            How many rows to skip from header.
        cpu
            How many CPU cores to use.

        Returns
        -------
        Metagene

        Examples
        --------
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/report.bedGraph'
        >>> metagene = Metagene.from_bedGraph(path, genome, up_windows=500, body_windows=1000, down_windows=500)
        """
        if isinstance(genome, Genome): raise TypeError("Genome class need to be converted into DataFrame (e.g. Genome.gene_body()).")

        sequence = Sequence.from_fasta(sequence, temp_dir)
        mapped = Mapper.bedGraph(file, sequence, temp_dir,
                                 save_preprocessed, True if save_preprocessed is None else False,
                                 batch_size, cpu, skip_rows)

        return cls.from_parquet(mapped.report_file, genome, up_windows, body_windows, down_windows, sumfunc)

    @classmethod
    def from_coverage(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            sequence: str | Path,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            sumfunc: str = "mean",
            batch_size: int = 10 ** 6,
            cpu: int = multiprocessing.cpu_count(),
            skip_rows: int = 1,
            save_preprocessed: str = None,
            temp_dir: str = "./"
    ):
        """
        Constructor for Metagene class from ``.cov`` file.

        Parameters
        ----------
        file
            Path to ``.cov`` file.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        sequence
            Path to FASTA genome sequence file.
        batch_size
            How many rows to read simultaneously.
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        sumfunc
            Summary function to calculate density for window with.
        temp_dir
            Directory for temporary files.
        save_preprocessed
            Does preprocessed file need to be saved
        skip_rows
            How many rows to skip from header.
        cpu
            How many CPU cores to use.


        Returns
        -------
        Metagene

        Examples
        --------

        >>> path = 'path/to/report.cov'
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> metagene = Metagene.from_coverage(path, genome, up_windows=500, body_windows=1000, down_windows=500)
        """
        if isinstance(genome, Genome): raise TypeError("Genome class need to be converted into DataFrame (e.g. Genome.gene_body()).")

        sequence = Sequence.from_fasta(sequence, temp_dir)
        mapped = Mapper.coverage(file, sequence, temp_dir,
                                 save_preprocessed, True if save_preprocessed is None else False,
                                 batch_size, cpu, skip_rows)

        return cls.from_parquet(mapped.report_file, genome, up_windows, body_windows, down_windows, sumfunc)

    @staticmethod
    def __check_windows(uw, gw, dw):
        return uw if uw > 0 else 0, gw if gw > 0 else 0, dw if dw > 0 else 0

    @staticmethod
    def __check_exists(file: Path):
        if not file.exists():
            raise FileNotFoundError(f"Specified file: {file.absolute()} – not found!")

    @staticmethod
    def __print_batch_stat(read_bytes, total_bytes, ram_bytes):
        print("Read {read_mb}/{total_mb}Mb | Metagene RAM usage - {ram_usage}Mb".format(
            read_mb=round(read_bytes / 1024 ** 2, 1),
            total_mb=round(total_bytes / 1024 ** 2, 1),
            ram_usage=round(ram_bytes / 1024 ** 2, 1)), end="\r")

    @staticmethod
    def __process_batch(df: pl.DataFrame, genome: pl.DataFrame, mutate_fun, up_win, gene_win, down_win, sumfunc):
        # *** POLARS EXPRESSIONS ***
        # cast genome columns to type to join
        GENE_COLUMNS = [
            pl.col('strand').cast(MetageneSchema.strand),
            pl.col('chr').cast(MetageneSchema.chr)
        ]
        # upstream region position check
        UP_REGION = pl.col('position') < pl.col('start')
        # body region position check
        BODY_REGION = (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
        # downstream region position check
        DOWN_REGION = (pl.col('position') > pl.col('end'))

        UP_FRAGMENT = (((pl.col('position') - pl.col('upstream')) / (pl.col('start') - pl.col('upstream'))) * up_win).floor()
        # fragment even for position == end needs to be rounded by floor
        # so 1e-10 is added (position is always < end)
        BODY_FRAGMENT = (((pl.col('position') - pl.col('start')) / (pl.col('end') - pl.col('start') + 1e-10)) * gene_win).floor() + up_win
        DOWN_FRAGMENT = (((pl.col('position') - pl.col('end')) / (pl.col('downstream') - pl.col('end') + 1e-10)) * down_win).floor() + up_win + gene_win

        # Firstly BismarkPlot was written so there were only one sum statistic - mean.
        # Sum and count of densities was calculated for further weighted mean analysis in respect to fragment size
        # For backwards compatibility, for newly introduces statistics, column names are kept the same.
        # Count is set to 1 and "sum" to actual statistics (e.g. median, min, e.t.c)
        if sumfunc == "median":
            AGG_EXPR = [pl.median("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "min":
            AGG_EXPR = [pl.min("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "max":
            AGG_EXPR = [pl.max("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "geometric":
            AGG_EXPR = [pl.col("density").log().mean().exp().alias("sum"),
                        pl.lit(1).alias("count")]
        elif sumfunc == "1pgeometric":
            AGG_EXPR = [(pl.col("density").log1p().mean().exp() - 1).alias("sum"),
                        pl.lit(1).alias("count")]
        else:
            AGG_EXPR = [pl.sum('density').alias('sum'), pl.count('density').alias('count')]

        df = mutate_fun(df)

        return (
            df.lazy()
            # assign types
            # calculate density for each cytosine
            .with_columns([
                pl.col('position').cast(MetageneSchema.position),
                pl.col('chr').cast(MetageneSchema.chr),
                pl.col('strand').cast(MetageneSchema.strand),
                pl.col('context').cast(MetageneSchema.context),
            ])
            # sort by position for joining
            .sort(['chr', 'strand', 'position'])
            # join with nearest
            .join_asof(
                genome.lazy().with_columns(GENE_COLUMNS),
                left_on='position', right_on='upstream', by=['chr', 'strand']
            )
            # limit by end of region
            .filter(pl.col('position') <= pl.col('downstream'))
            # calculate fragment ids
            .with_columns([
                pl.when(UP_REGION).then(UP_FRAGMENT)
                .when(BODY_REGION).then(BODY_FRAGMENT)
                .when(DOWN_REGION).then(DOWN_FRAGMENT)
                .alias('fragment'),
                pl.concat_str([
                    pl.col("chr"),
                    (pl.concat_str(pl.col("start"), pl.col("end"), separator="-"))
                ], separator=":").alias("gene")
            ])
            .with_columns([
                pl.col("fragment").cast(MetageneSchema.fragment),
                pl.col("gene").cast(MetageneSchema.gene),
                pl.col('id').cast(MetageneSchema.id)
            ])
            # gather fragment stats
            .groupby(by=['chr', 'strand', 'start', 'gene', 'context', 'id', 'fragment'])
            .agg(AGG_EXPR)
            .drop_nulls(subset=['sum'])
        ).collect()

    def filter(
            self,
            context: Literal["CG", "CHG", "CHH", None] = None,
            strand: Literal["+", "-", None] = None,
            chr: str = None,
            genome: pl.DataFrame = None
    ):
        """
        Method for filtering metagene.

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH) to filter (only one).
        strand
            Strand to filter (+ or -).
        chr
            Chromosome name to filter.
        genome
            DataFrame with annotation to filter with (e.g. from :class:`Genome`)

        Returns
        -------
            Filtered :class:`Metagene`.

        Examples
        --------
        >>> metagene
        shape: (3, 9)
        ┌─────────────┬────────┬───────┬────────────────┬─────────┬───────────────┬──────────┬─────┬───────┐
        │ chr         ┆ strand ┆ start ┆ gene           ┆ context ┆ id            ┆ fragment ┆ sum ┆ count │
        │ ---         ┆ ---    ┆ ---   ┆ ---            ┆ ---     ┆ ---           ┆ ---      ┆ --- ┆ ---   │
        │ cat         ┆ cat    ┆ u64   ┆ cat            ┆ cat     ┆ cat           ┆ u32      ┆ f32 ┆ u32   │
        ╞═════════════╪════════╪═══════╪════════════════╪═════════╪═══════════════╪══════════╪═════╪═══════╡
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CG      ┆ gene-AT1G0101 ┆ 4        ┆ 0.0 ┆ 1     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CHH     ┆ gene-AT1G0101 ┆ 19       ┆ 0.0 ┆ 2     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CHH     ┆ gene-AT1G0101 ┆ 20       ┆ 0.0 ┆ 1     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        └─────────────┴────────┴───────┴────────────────┴─────────┴───────────────┴──────────┴─────┴───────┘
        >>> metagene.filter(context = "CG", strand = "+")
        shape: (3, 9)
        ┌─────────────┬────────┬───────┬────────────────┬─────────┬───────────────┬──────────┬─────┬───────┐
        │ chr         ┆ strand ┆ start ┆ gene           ┆ context ┆ id            ┆ fragment ┆ sum ┆ count │
        │ ---         ┆ ---    ┆ ---   ┆ ---            ┆ ---     ┆ ---           ┆ ---      ┆ --- ┆ ---   │
        │ cat         ┆ cat    ┆ u64   ┆ cat            ┆ cat     ┆ cat           ┆ u32      ┆ f32 ┆ u32   │
        ╞═════════════╪════════╪═══════╪════════════════╪═════════╪═══════════════╪══════════╪═════╪═══════╡
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CG      ┆ gene-AT1G0101 ┆ 4        ┆ 0.0 ┆ 1     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CG      ┆ gene-AT1G0101 ┆ 166      ┆ 0.0 ┆ 1     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        │ NC_003070.9 ┆ +      ┆ 3631  ┆ NC_003070.9:36 ┆ CG      ┆ gene-AT1G0101 ┆ 264      ┆ 0.0 ┆ 1     │
        │             ┆        ┆       ┆ 31-5899        ┆         ┆ 0             ┆          ┆     ┆       │
        └─────────────┴────────┴───────┴────────────────┴─────────┴───────────────┴──────────┴─────┴───────┘

        """

        context_filter = self.bismark["context"] == context if context is not None else True
        strand_filter = self.bismark["strand"] == strand if strand is not None else True
        chr_filter = self.bismark["chr"] == chr if chr is not None else True

        metadata = self.metadata
        metadata["context"] = context
        metadata["strand"] = strand

        if genome is not None:
            def genome_filter(df: pl.DataFrame):
                return df.join(genome.select(["chr", "strand", "start"]), on=["chr", "strand", "start"])
        else:
            genome_filter = lambda df: df

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(genome_filter(self.bismark.filter(context_filter & strand_filter & chr_filter)),
                                  **metadata)

    def resize(self, to_fragments: int = None):
        """
        Mutate DataFrame to fewer fragments.

        Parameters
        ----------
        to_fragments
            Number of TOTAL (including flanking regions) fragments per gene.

        Returns
        -------
            Resized :class:`Metagene`.

        Examples
        --------
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/bismark.CX_report.txt'
        >>> metagene = Metagene.from_bismark(path, genome, up_windows=500, body_windows=1000, down_windows=500)
        >>> metagene
        Metagene with 20223666 windows total.
        Filtered by None context and None strand.
        Upstream windows: 500.
        Body windows: 1000.
        Downstream windows: 500.
        >>> metagene.resize(100)
        Metagene with 4946800 windows total.
        Filtered by None context and None strand.
        Upstream windows: 25.
        Body windows: 50.
        Downstream windows: 25.
        """
        if self.upstream_windows is not None and self.gene_windows is not None and self.downstream_windows is not None:
            from_fragments = self.total_windows
        else:
            from_fragments = self.bismark["fragment"].max() + 1

        if to_fragments is None or from_fragments <= to_fragments:
            return self

        resized = (
            self.bismark.lazy()
            .with_columns(
                ((pl.col("fragment") / from_fragments) * to_fragments).floor()
                .cast(MetageneSchema.fragment)
            )
            .group_by(
                by=['chr', 'strand', 'start', 'gene', 'id', 'context', 'fragment']
            ).agg([
                pl.sum('sum').alias('sum'),
                pl.sum('count').alias('count')
            ])
        ).collect()

        metadata = self.metadata
        metadata["upstream_windows"] = metadata["upstream_windows"] // (from_fragments // to_fragments)
        metadata["downstream_windows"] = metadata["downstream_windows"] // (from_fragments // to_fragments)
        metadata["gene_windows"] = metadata["gene_windows"] // (from_fragments // to_fragments)

        return self.__class__(resized, **metadata)

    def trim_flank(self, upstream=True, downstream=True):
        """
        Trim Metagene flanking regions.

        Parameters
        ----------
        upstream
            Trim upstream region?
        downstream
            Trim downstream region?

        Returns
        -------
            Trimmed :class:`Metagene`

        Examples
        --------
        >>> metagene
        Metagene with 20223666 windows total.
        Filtered by None context and None strand.
        Upstream windows: 500.
        Body windows: 1000.
        Downstream windows: 500.
        >>> metagene.trim_flank()
        Metagene with 7750085 windows total.
        Filtered by None context and None strand.
        Upstream windows: 0.
        Body windows: 1000.
        Downstream windows: 0.

        Or if you do not need to trim one of flanking regions (e.g. upstream).

        >>> metagene.trim_flank(upstream=False)
        Metagene with 16288550 windows total.
        Filtered by None context and None strand.
        Upstream windows: 500.
        Body windows: 1000.
        Downstream windows: 0.
        """
        trimmed = self.bismark.lazy()
        metadata = self.metadata.copy()
        if downstream:
            trimmed = (
                trimmed
                .filter(pl.col("fragment") < self.upstream_windows + self.gene_windows)
            )
            metadata["downstream_windows"] = 0

        if upstream:
            trimmed = (
                trimmed
                .filter(pl.col("fragment") > self.upstream_windows - 1)
                .with_columns(pl.col("fragment") - self.upstream_windows)
            )
            metadata["upstream_windows"] = 0

        return self.__class__(trimmed.collect(), **metadata)

    # TODO finish annotation
    def clustering(
            self,
            count_threshold: int = 5,
            dist_method: str = "euclidean",
            clust_method: str = "average"):
        """
        Cluster regions with hierarchical clustering, by their methylation pattern.

        Parameters
        ----------
        count_threshold
            Minimum counts per window
        dist_method
            Distance method to use.
        clust_method
            Clustering method to use.

        Returns
        -------
            :class:`Clustering`

        Warnings
        --------
            Experimental Feature. May run very slow!

        See Also
        -------
        Clustering : For possible analysis options

        `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_ : For possible distance methods.

        `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_ : For possible clustering methods.
        """

        return Clustering(self.bismark, count_threshold, dist_method, clust_method, **self.metadata)

    def line_plot(
            self,
            resolution: int = None,
            stat="wmean",
            merge_strands: bool = True
    ):
        """
        Create :class:`LinePlot` method.

        Parameters
        ----------
        resolution
            Number of fragments to resize to. Keep None if no resize is needed.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``
        merge_strands
            Does negative strand need to be reversed

        Returns
        -------
        LinePlot

        Notes
        -----
        - ``mean`` – Default mean between bins, when count of cytosine residues in bin IS NOT taken into account

        - ``wmean`` – Weighted mean between bins. Cytosine residues in bin IS taken into account

        - ``log`` – NOT weighted geometric mean.

        - ``wlog`` - Weighted geometric mean.

        - ``qN`` – Return quantile by ``N`` percent (e.g. "``q50``")

        Examples
        --------
        Firstly we need to initialize Metagene class

        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/bismark.CX_report.txt'
        >>> metagene = Metagene.from_bismark(path, genome, up_windows=500, body_windows=1000, down_windows=500)

        Next we can optionally filter metagene by context and strand.

        >>> filtered = metagene.filter(context = "CG", strand = "-")

        And LinePlot can be created

        >>> lp = filtered.line_plot()
        >>> figure = lp.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_mpl.png

        No filtering is suitable too. Then LinePlot will visualize all methylation contexts.

        >>> lp = metagene.line_plot()
        >>> figure = lp.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_multi_mpl.png

        You can use Plotly version for all plots as well.

        >>> figure = lp.draw_ploty()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_multi_plotly.png

        See Also
        --------
        LinePlot : For more information about plottling parameters.
        """
        bismark = self.resize(resolution)
        return LinePlot(bismark.bismark, stat=stat, merge_strands=merge_strands, **bismark.metadata)

    def heat_map(
            self,
            nrow: int = 100,
            ncol: int = 100,
            stat="wmean"
    ):
        """
        Create :class:`HeatMap` method.

        Parameters
        ----------
        nrow
            Number of fragments to resize to. Keep None if no resize is needed.
        ncol
            Number of columns in the resulting heat-map.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``

        Returns
        -------
        HeatMap

        Examples
        --------

        Firstly we need to initialize Metagene class

        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/bismark.CX_report.txt'
        >>> metagene = Metagene.from_bismark(path, genome, up_windows=500, body_windows=1000, down_windows=500)

        Next we need to (in contrast with :meth:`Metagene.line_plot`) filter metagene by context and strand.

        >>> filtered = metagene.filter(context = "CG", strand = "-")

        And HeatMap can be created. Let it's width be 200 columns and height – 100 rows.

        >>> hm = filtered.heat_map(nrow=100, ncol=200)
        >>> figure = hm.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/heatmap/ara_mpl.png

        Plotly version is also available:

        >>> figure = hm.draw_plotly()
        >>> figure.show()

        .. image:: ../../images/heatmap/ara_plotly.png

        See Also
        --------
        Metagene.line_plot : For more information about `stat` parameter.
        HeatMap : For more information about plotting parameters.
        """

        bismark = self.resize(ncol)
        return HeatMap(bismark.bismark, nrow, order=None, stat=stat, **bismark.metadata)

    def __str__(self):
        representation = (f'Metagene with {len(self.bismark)} windows total.\n'
                          f'Filtered by {self.context} context and {self.strand} strand.\n'
                          f'Upstream windows: {self.upstream_windows}.\n'
                          f'Body windows: {self.gene_windows}.\n'
                          f'Downstream windows: {self.downstream_windows}.\n')

        return representation

    def __repr__(self):
        return self.__str__()


# TODO add other type constructors
class MetageneFiles(BismarkFilesBase):
    """
    Stores and plots multiple data for :class:`Metagene`.

    If you want to compare Bismark data with different genomes, create this class with a list of :class:`Bismark` classes.
    """
    @classmethod
    def from_list(
            cls,
            filenames: list[str | Path],
            genomes: pl.DataFrame | list[pl.DataFrame],
            labels: list[str] = None,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            report_type: Literal["bismark", "parquet", "binom", "bedGraph", "coverage"] = "bismark",
            block_size_mb: int = 50,
            use_threads: bool = True
    ) -> MetageneFiles:
        """
        Create istance of :class:`MetageneFiles` from list of paths.

        Parameters
        ----------
        filenames
            List of filenames to read from
        genomes
            Annotation DataFrame or list of DataFrames (may be different annotations)
        labels
            Labels for plots for Metagenes
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        report_type
            Type of input report. Possible options: ``bismark``, ``parquet``, ``binom``, ``bedGraph``, ``coverage``.
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`

        Returns
        -------
        MetageneFiles
            Instance of :class:`MetageneFiles`

        See Also
        ________
        Metagene

        Examples
        --------

        Initialization using :meth:`MetageneFiles.from_list`

        >>> ara_genome = Genome.from_gff("/path/to/arath.gff").gene_body(min_length=2000)
        >>> brd_genome = Genome.from_gff("/path/to/bradi.gff").gene_body(min_length=2000)
        >>>
        >>> metagenes = MetageneFiles.from_list(
        >>>     ["path/to/bradi.txt", "path/to/ara.txt"],
        >>>     [brd_genome, ara_genome],
        >>>     ["BraDi", "AraTh"],
        >>>     report_type="bismark",
        >>>     up_windows=250, body_windows=500, down_windows=250
        >>> )

        :class:`MetageneFiles` can be initialized explicitly:

        >>> metagene_ara = Metagene.from_bismark("path/to/ara.txt", ara_genome, up_windows=250, body_windows=500, down_windows=250)
        >>> metagene_brd = Metagene.from_bismark("path/to/ara.txt", ara_genome, up_windows=250, body_windows=500, down_windows=250)
        >>> metagenes = MetageneFiles(samples=[metagene_brd, metagene_ara], labels=["BraDi", "AraTh"])

        The resulting objects will be identical.

        Warnings
        --------
        When :class:`MetageneFiles` is initialized explicitly, number of windows needs ot be the same in evety sample
        """
        read_fnc: dict[str | typing.Callable] = {
            "bismark":  Metagene.from_bismark,
            "parquet":  Metagene.from_parquet,
            "binom":    Metagene.from_binom,
            "bedGraph": Metagene.from_bedGraph,
            "coverage": Metagene.from_coverage
        }

        if not isinstance(genomes, list):
            genomes = [genomes] * len(filenames)
        else:
            if len(genomes) != len(filenames):
                raise AttributeError("Number of genomes and filenames provided does not match")

        samples: list[Metagene] = []
        for file, genome in zip(filenames, genomes):
            samples.append(read_fnc[report_type](file, genome, up_windows, body_windows, down_windows, block_size_mb, use_threads))

        return cls(samples, labels)

    def filter(self, context: str = None, strand: str = None, chr: str = None):
        """
        :meth:`Metagene.filter` all metagenes.

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH) to filter (only one).
        strand
            Strand to filter (+ or -).
        chr
            Chromosome name to filter.
        genome
            DataFrame with annotation to filter with (e.g. from :class:`Genome`)

        Returns
        -------
            Filtered :class:`MetageneFiles`.

        See Also
        --------
        Metagene.filter : For examples.
        """
        return self.__class__([sample.filter(context, strand, chr) for sample in self.samples], self.labels)

    def trim_flank(self, upstream=True, downstream=True):
        """
        :meth:`Metagene.trim_flank` all metagenes.

        Parameters
        ----------
        upstream
            Trim upstream region?
        downstream
            Trim downstream region?

        Returns
        -------
            Trimmed :class:`Metagene`

        See Also
        --------
        Metagene.trim_flank : For examples.
        """

        return self.__class__([sample.trim_flank(upstream, downstream) for sample in self.samples], self.labels)

    def resize(self, to_fragments: int):
        """
        :meth:`Metagene.resize` all metagenes.

        Parameters
        ----------
        to_fragments
            Number of TOTAL (including flanking regions) fragments per gene.

        Returns
        -------
            Resized :class:`Metagene`.

        See Also
        --------
        Metagene.resize : For examples.
        """
        return self.__class__([sample.resize(to_fragments) for sample in self.samples], self.labels)

    def merge(self):
        """
        Merge :class:`MetageneFiles` into single :class:`Metagene`

        Warnings
        --------
        **ALL** metagenes in :class:`MetageneFiles` need to be aligned to the same annotation.

        Parameters
        ----------

        Returns
        -------
            Instance of :class:`Metagene`, with merged replicates.
        """
        pl.enable_string_cache()

        metadata = [sample.metadata for sample in self.samples]
        upstream_windows = set([md.get("upstream_windows") for md in metadata])
        gene_windows = set([md.get("gene_windows") for md in metadata])
        downstream_windows = set(
            [md.get("downstream_windows") for md in metadata])

        if len(upstream_windows) == len(downstream_windows) == len(gene_windows) == 1:
            merged = (
                pl.concat([sample.bismark for sample in self.samples]).lazy()
                .group_by(["strand", "context", "chr", "gene", "fragment"])
                .agg([pl.sum("sum").alias("sum"), pl.sum("count").alias("count")])
            ).collect()

            return Metagene(merged,
                            upstream_windows=list(upstream_windows)[0],
                            downstream_windows=list(downstream_windows)[0],
                            gene_windows=list(gene_windows)[0])
        else:
            raise Exception("Metadata for merge DataFrames does not match!")

    def line_plot(self, resolution: int = None, stat: str = "wmean", merge_strands: bool = True):
        """
        Create :class:`LinePlotFiles` method.

        Parameters
        ----------
        resolution
            Number of fragments to resize to. Keep None if no resize is needed.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``
        merge_strands
            Does negative strand need to be reversed

        Returns
        -------
            Instance of :class:`LinePlotFiles`

        See Also
        --------
        Metagene.line_plot : for more information about `stat` parameter.

        Examples
        --------

        >>> filtered = metagenes.filter("CG")
        >>> lp = metagenes.line_plot()

        Matplotlib version:

        >>> figure = lp.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_bradi_mpl.png

        Plotly version

        >>> figure = lp.draw_plotly()
        >>> figure.show()

        Now when we have metagene initlaized

        .. image:: ../../images/lineplot/ara_bradi_plotly.png
        """
        return LinePlotFiles([sample.line_plot(resolution, stat, merge_strands) for sample in self.samples], self.labels)

    def heat_map(self, nrow: int = 100, ncol: int = None, stat: str = "wmean"):
        """
        Create :class:`HeatMapFiles` method.

        Parameters
        ----------
        nrow
            Number of fragments to resize to. Keep None if no resize is needed.
        ncol
            Number of columns in the resulting heat-map.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``

        Returns
        -------
            Instance of :class:`HeatMapFiles`

        See Also
        --------
        Metagene.line_plot : for more information about `stat` parameter.

        Metagene.heat_map

        Examples
        --------

        >>> filtered = metagenes.filter("CG")
        >>> hm = metagenes.heat_map(nrow=100, ncol=100)

        Matplotlib version:

        >>> figure = hm.draw_mpl(major_labels=None)
        >>> figure.show()

        .. image:: ../../images/heatmap/ara_bradi_mpl.png

        Plotly version

        >>> figure = hm.draw_plotly(major_labels=None)
        >>> figure.show()

        Now when we have metagene initlaized

        .. image:: ../../images/heatmap/ara_bradi_plotly.png
        """
        return HeatMapFiles([sample.heat_map(nrow, ncol, stat) for sample in self.samples], self.labels)

    def violin_plot(self, fig_axes: tuple = None, title: str = ""):
        """
        Draws violin plot for Metagenes with matplotlib.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_) (e.g. created by `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_)
        title
            Title for plot.

        Returns
        -------
            `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        Examples
        --------

        Assuming we need to compare only gene body methylation, we need to trim flank regions from metagenes (this step is **OPTIONAL**):

        >>> trimmed = metagenes.trim_flank()
        >>>
        >>> figure = trimmed.violin_plot()
        >>> figure.show()

        .. image:: ../../images/boxplot/vp_ara_bradi_mpl.png
        """
        data = LinePlotFiles([sample.line_plot()
                             for sample in self.samples], self.labels)
        data = [sample.plot_data.sort(
            "fragment")["density"].to_numpy() for sample in data.samples]

        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        axes.violinplot(data, showmeans=False, showmedians=True)
        axes.set_xticks(np.arange(1, len(self.labels) + 1), labels=self.labels)
        axes.set_title(title)
        axes.set_ylabel('Methylation density')

        return fig

    def violin_plot_plotly(self, title="", points=None):
        """
        Draws violin plot for Metagenes with plotly.

        Parameters
        ----------
        points
            How many points should be plotted. For variants see `plotly.express.box <https://plotly.com/python-api-reference/generated/plotly.express.box>`_
        title
            Title for plot.

        Returns
        -------
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_

        Examples
        --------

        Assuming we need to compare only gene body methylation, we need to trim flank regions from metagenes (this step is **OPTIONAL**):

        >>> trimmed = metagenes.trim_flank()
        >>>
        >>> figure = trimmed.violin_plot_plotly()
        >>> figure.show()

        .. image:: ../../images/boxplot/vp_ara_bradi_plotly.png
        """

        def sample_convert(sample, label):
            return (
                sample
                .line_plot()
                .plot_data
                .with_columns([
                    pl.col("density") * 100,  # convert to %
                    pl.lit(label).alias("label")
                ])
            )

        data = pl.concat([sample_convert(sample, label) for sample, label in zip(self.samples, self.labels)])
        data = data.to_pandas()

        labels = dict(
            context="Context",
            label="",
            density="Methylation density, %"
        )

        figure = px.violin(data, x="label", y="density",
                           color="context", points=points,
                           labels=labels, title=title)

        return figure

    def box_plot(self, fig_axes: tuple = None, showfliers=False, title: str = None):
        """
        Draws box plot for Metagenes with matplotlib.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_) (e.g. created by `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_)
        title
            Title for plot.
        showfliers
            Do fliers need to be shown.

        Returns
        -------
            `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        Examples
        --------

        Assuming we need to compare only gene body methylation, we need to trim flank regions from metagenes (this step is **OPTIONAL**):

        >>> trimmed = metagenes.trim_flank()
        >>>
        >>> figure = trimmed.box_plot()
        >>> figure.show()

        .. image:: ../../images/boxplot/bp_ara_bradi_mpl.png
        """
        data = LinePlotFiles([sample.line_plot()
                             for sample in self.samples], self.labels)
        data = [sample.plot_data.sort(
            "fragment")["density"].to_numpy() for sample in data.samples]

        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        axes.boxplot(data, showfliers=showfliers)
        axes.set_xticks(np.arange(1, len(self.labels) + 1), labels=self.labels)
        axes.set_title(title)
        axes.set_ylabel('Methylation density')

        return fig

    def box_plot_plotly(self, title="", points=None):
        """
        Draws box plot for Metagenes with plotly.

        Parameters
        ----------
        points
            How many points should be plotted. For variants see `plotly.express.box <https://plotly.com/python-api-reference/generated/plotly.express.box>`_
        title
            Title for plot.

        Returns
        -------
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_

        Examples
        --------

        Assuming we need to compare only gene body methylation, we need to trim flank regions from metagenes (this step is **OPTIONAL**):

        >>> trimmed = metagenes.trim_flank()
        >>>
        >>> figure = trimmed.box_plot_plotly()
        >>> figure.show()

        .. image:: ../../images/boxplot/bp_ara_bradi_plotly.png
        """
        def sample_convert(sample, label):
            return (
                sample
                .line_plot()
                .plot_data
                .with_columns([
                    pl.col("density") * 100,  # convert to %
                    pl.lit(label).alias("label")
                ])
            )

        data = pl.concat([sample_convert(sample, label) for sample, label in zip(self.samples, self.labels)])
        data = data.to_pandas()

        labels = dict(
            context="Context",
            label="",
            density="Methylation density, %"
        )

        figure = px.box(data, x="label", y="density",
                        color="context", points=points,
                        labels=labels, title=title)

        return figure

    def __dendrogram(self, stat="mean"):
        # get intersecting regions
        gene_sets = [set(sample.bismark["gene"].to_list()) for sample in self.samples]
        intersecting = list(set.intersection(*gene_sets))

        if len(intersecting) < 1:
            print("No regions with same labels were found. Exiting.")
            return

        # TODO check options setter for stat (limited set of options)
        # Lazy
        def region_levels(bismark: pl.DataFrame, stat="mean"):
            if stat == "median":
                expr = pl.median("density")
            elif stat == "min":
                expr = pl.min("density")
            elif stat == "max":
                expr = pl.max("density")
            else:
                expr = pl.mean("density")

            levels = (
                bismark.lazy()
                .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
                .group_by(["gene"])
                .agg(expr.alias("stat"))
                .sort("gene")
            )

            return levels

        levels = [region_levels(sample.bismark, stat).rename({"stat": str(label)})
                  for sample, label in zip(self.samples, self.labels)]

        data = pl.concat(levels, how="align").collect()

        matrix = data.select(pl.exclude("gene")).to_numpy()
        genes = data["gene"].to_numpy()

        # get intersected
        matrix = matrix[np.isin(genes, intersecting), :]

        return


class LinePlot(PlotBase):
    """Line-plot single metagene"""

    def __init__(self, bismark_df: pl.DataFrame, stat="wmean", merge_strands: bool = True, **kwargs):
        """
        Calculates plot data for line-plot.
        """
        super().__init__(bismark_df, **kwargs)

        self.stat = stat

        if merge_strands:
            bismark_df = self.__merge_strands(bismark_df)
        plot_data = self.__calculate_plot_data(bismark_df, stat)

        if not merge_strands:
            if self.strand == '-':
                plot_data = self.__strand_reverse(plot_data)
        self.plot_data = plot_data

    @staticmethod
    def __calculate_plot_data(df: pl.DataFrame, stat):
        if stat == "log":
            stat_expr = (pl.col("sum") / pl.col("count")).log1p().mean().exp() - 1
        elif stat == "wlog":
            stat_expr = (((pl.col("sum") / pl.col("count")).log1p() * pl.col("count")).sum() / pl.sum("count")).exp() - 1
        elif stat == "mean":
            stat_expr = (pl.col("sum") / pl.col("count")).mean()
        elif re.search("^q(\d+)", stat):
            quantile = re.search("q(\d+)", stat).group(1)
            stat_expr = (pl.col("sum") / pl.col("count")).quantile(int(quantile) / 100)
        else:
            stat_expr = pl.sum("sum") / pl.sum("count")

        res = (
            df
            .group_by(["context", "fragment"]).agg([
                pl.col("sum"),
                pl.col("count").cast(MetageneSchema.count),
                stat_expr.alias("density")
            ])
            .sort("fragment")
        )

        return res

    @staticmethod
    def __strand_reverse(df: pl.DataFrame):
        max_fragment = df["fragment"].max()
        return df.with_columns((max_fragment - pl.col("fragment")).alias("fragment")).sort("fragment")

    def __merge_strands(self, df: pl.DataFrame):
        return df.filter(pl.col("strand") == "+").extend(self.__strand_reverse(df.filter(pl.col("strand") == "-")))


    @staticmethod
    def __get_x_y(df, smooth, confidence):
        if 0 < confidence < 1:
            df = (
                df
                .with_columns(
                    pl.struct(["sum", "count"]).map_elements(
                        lambda x: interval(x["sum"], x["count"], confidence)
                    ).alias("interval")
                )
                .unnest("interval")
                .select(["fragment", "lower", "density", "upper"])
            )

        data = df["density"]

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            data = savgol_filter(data, window, 3, mode='nearest')

        lower, upper = None, None
        data = data * 100  # convert to percents

        if 0 < confidence < 1:
            upper = df["upper"].to_numpy() * 100  # convert to percents
            lower = df["lower"].to_numpy() * 100  # convert to percents

            upper = savgol_filter(upper, window, 3, mode="nearest") if smooth else upper
            lower = savgol_filter(lower, window, 3, mode="nearest") if smooth else lower

        return lower, data, upper

    def save_plot_rds(self, path, compress: bool = False):
        """
        Saves plot data in a rds DataFrame with columns:

        +----------+---------+
        | fragment | density |
        +==========+=========+
        | Int      | Float   |
        +----------+---------+
        """
        df = self.bismark.group_by("fragment").agg(
            (pl.sum("sum") / pl.sum("count")).alias("density")
        )
        write_rds(path, df.to_pandas(),
                  compress="gzip" if compress else None)

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            smooth: int = 50,
            label: str = "",
            confidence: int = 0,
            linewidth: float = 1.0,
            linestyle: str = '-',
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border:bool = True
    ) -> Figure:
        """
        Draws line-plot on given :class:`matplotlib.Axes` or makes them itself.

        :param fig_axes: Tuple with (fig, axes) from :meth:`matplotlib.plt.subplots`
        :param smooth: Window for SavGol filter. (see :meth:`scipy.signal.savgol`)
        :param label: Label of the plot
        :param confidence: Probability for confidence bands. 0 for disabled.
        :param linewidth: See matplotlib documentation.
        :param linestyle: See matplotlib documentation.
        :return:
        """
        fig, axes = plt.subplots() if fig_axes is None else fig_axes
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        contexts = self.plot_data["context"].unique().to_list()

        for context in contexts:
            df = self.plot_data.filter(pl.col("context") == context)

            lower, data, upper = self.__get_x_y(df, smooth, confidence)

            x = np.arange(len(data))

            axes.plot(x, data,
                      label=f"{context}" if not label else f"{label}_{context}",
                      linestyle=linestyle, linewidth=linewidth)

            if 0 < confidence < 1:
                axes.fill_between(x, lower, upper, alpha=.2)

        self.flank_lines(axes, major_labels, minor_labels, show_border)

        axes.legend()

        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        return fig

    def draw_plotly(
            self,
            figure: go.Figure = None,
            smooth: int = 50,
            label: str = "",
            confidence: int = .0,
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ):
        figure = go.Figure() if figure is None else figure
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        contexts = self.plot_data["context"].unique().to_list()

        for context in contexts:
            df = self.plot_data.filter(pl.col("context") == context)

            lower, data, upper = self.__get_x_y(df, smooth, confidence)

            x = np.arange(len(data))

            traces = [go.Scatter(x=x, y=data, name=f"{context}" if not label else f"{label}_{context}", mode="lines")]

            if 0 < confidence < 1:
                traces += [
                    go.Scatter(x=x, y=upper, mode="lines", line_color='rgba(0,0,0,0)', showlegend=False,
                               name=f"{context}_{confidence}CI" if not label else f"{label}_{context}_{confidence}CI"),
                    go.Scatter(x=x, y=lower, mode="lines", line_color='rgba(0,0,0,0)', showlegend=True,
                               fill="tonexty", fillcolor='rgba(0, 0, 0, 0.2)',
                               name=f"{context}_{confidence}CI" if not label else f"{label}_{context}_{confidence}CI"),
                ]

            figure.add_traces(data=traces)

        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %"
        )

        figure = self.flank_lines_plotly(figure, major_labels, minor_labels, show_border)

        return figure


class LinePlotFiles(BismarkFilesBase):
    """Line-plot multiple metagenes"""

    def draw_mpl(
        self,
        smooth: int = 50,
        linewidth: float = 1.0,
        linestyle: str = '-',
        confidence: int = 0,
        major_labels: list[str] = None,
        minor_labels: list[str] = None,
        show_border: bool = True
    ):
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        plt.clf()
        fig, axes = plt.subplots()
        for lp, label in zip(self.samples, self.labels):
            assert isinstance(lp, LinePlot)
            lp.draw_mpl((fig, axes), smooth, label, confidence, linewidth, linestyle, major_labels, minor_labels, show_border)

        return fig

    def draw_plotly(self,
                    smooth: int = 50,
                    confidence: int = 0,
                    major_labels: list[str] = None,
                    minor_labels: list[str] = None,
                    show_border: bool = True
                    ):

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        figure = go.Figure()

        for lp, label in zip(self.samples, self.labels):
            assert isinstance(lp, LinePlot)
            lp.draw_plotly(figure, smooth, label, confidence, major_labels, minor_labels, show_border)

        return figure

    def save_plot_rds(self, base_filename, compress: bool = False, merge: bool = False):
        if merge:
            merged = pl.concat(
                [sample.plot_data.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
            )
            write_rds(base_filename, merged.to_pandas(),
                      compress="gzip" if compress else None)
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_plot_rds(f"{remove_extension(base_filename)}_{label}.rds",
                                     compress="gzip" if compress else None)


class HeatMap(PlotBase):
    """Heat-map single metagene"""

    def __init__(self, bismark_df: pl.DataFrame, nrow, order=None, stat="wmean", **kwargs):
        super().__init__(bismark_df, **kwargs)

        plot_data = self.__calculcate_plot_data(bismark_df, nrow, order, stat)
        plot_data = self.__strand_reverse(plot_data)

        self.plot_data = plot_data

    def __calculcate_plot_data(self, df, nrow, order=None, stat="wmean"):
        if stat == "log":
            stat_expr = (pl.col("sum") / pl.col("count")).log1p().mean().exp() - 1
        elif stat == "wlog":
            stat_expr = (((pl.col("sum") / pl.col("count")).log1p() * pl.col("count")).sum() / pl.sum("count")).exp() - 1
        elif stat == "mean":
            stat_expr = (pl.col("sum") / pl.col("count")).mean()
        elif re.search("^q(\d+)", stat):
            quantile = re.search("q(\d+)", stat).group(1)
            stat_expr = (pl.col("sum") / pl.col("count")).quantile(int(quantile) / 100)
        else:
            stat_expr = pl.sum("sum") / pl.sum("count")

        order = (
            df.lazy()
            .groupby(['chr', 'strand', "gene"])
            .agg(
                stat_expr.alias("order")
            )
        ).collect()["order"] if order is None else order

        # sort by rows and add row numbers
        hm_data = (
            df.lazy()
            .groupby(['chr', 'strand', "gene"])
            .agg([pl.col('fragment'), pl.col('sum'), pl.col('count')])
            .with_columns(
                pl.lit(order).alias("order")
            )
            .sort('order', descending=True)
            # add row count
            .with_row_count(name='row')
            # round row count
            .with_columns(
                (pl.col('row') / (pl.col('row').max() + 1) * nrow).floor().alias('row').cast(pl.UInt16)
            )
            .explode(['fragment', 'sum', 'count'])
            # calc sum count for row|fragment
            .groupby(['row', 'fragment'])
            .agg(
                stat_expr.alias('density')
            )
        )

        # prepare full template
        template = (
            pl.LazyFrame(data={"row": list(range(nrow))})
            .with_columns(
                pl.lit([list(range(0, self.total_windows))]).alias("fragment")
            )
            .explode("fragment")
            .with_columns([
                pl.col("fragment").cast(MetageneSchema.fragment),
                pl.col("row").cast(pl.UInt16)
            ])
        )
        # join template with actual data
        hm_data = (
            # template join with orig
            template.join(hm_data, on=['row', 'fragment'], how='left')
            .fill_null(0)
            .sort(['row', 'fragment'])
        ).collect()

        # convert to matrix
        plot_data = np.array(
            hm_data.groupby('row', maintain_order=True).agg(
                pl.col('density'))['density'].to_list(),
            dtype=np.float32
        )

        return plot_data

    def __strand_reverse(self, df: np.ndarray):
        if self.strand == '-':
            return np.fliplr(df)
        return df

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            title: str = None,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ):
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        fig, axes = plt.subplots() if fig_axes is None else fig_axes

        vmin = 0 if vmin is None else vmin
        vmax = np.max(np.array(self.plot_data)) if vmax is None else vmax

        image = axes.imshow(
            self.plot_data,
            interpolation="nearest", aspect='auto',
            cmap=colormaps[color_scale.lower()],
            vmin=vmin, vmax=vmax
        )

        axes.set_title(title)
        axes.set_xlabel('Position')
        axes.set_ylabel('')

        self.flank_lines(axes, major_labels, minor_labels, show_border)
        axes.set_yticks([])

        plt.colorbar(image, ax=axes, label='Methylation density')

        return fig

    def draw_plotly(
            self,
            title: str = None,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ):

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        labels = dict(
            x="Position",
            y="Rank",
            color="Methylation density"
        )

        figure = px.imshow(
            self.plot_data,
            zmin=vmin, zmax=vmax,
            labels=labels,
            title=title,
            aspect="auto",
            color_continuous_scale=color_scale
        )

        # disable y ticks
        figure.update_layout(
            yaxis=dict(
                showticklabels=False
            )
        )

        figure = self.flank_lines_plotly(figure, major_labels, minor_labels, show_border)

        return figure

    def save_plot_rds(self, path, compress: bool = False):
        """
        Save heat-map data in a matrix (ncol:nrow)
        """
        write_rds(path, pdDataFrame(self.plot_data),
                  compress="gzip" if compress else None)


class HeatMapFiles(BismarkFilesBase):
    """Heat-map multiple metagenes"""

    def __add_flank_lines_plotly(self, figure: go.Figure, major_labels: list, minor_labels: list, show_border=True):
        """
        Add flank lines to the given axis (for line plot)
        """
        labels = prepare_labels(major_labels, minor_labels)

        if self.samples[0].downstream_windows < 1:
            labels["down_mid"], labels["body_end"] = [""] * 2

        if self.samples[0].upstream_windows < 1:
            labels["up_mid"], labels["body_start"] = [""] * 2

        ticks = self.samples[0].tick_positions

        names = list(ticks.keys())
        x_ticks = [ticks[key] for key in names]
        x_labels = [labels[key] for key in names]

        figure.for_each_xaxis(lambda x: x.update(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_labels)
        )

        if show_border:
            for tick in [ticks["body_start"], ticks["body_end"]]:
                figure.add_vline(x=tick, line_dash="dash", line_color="rgba(0,0,0,0.2)")

        return figure

    def draw_mpl(
            self,
            title: str = None,
            color_scale: str = "Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True,
    ):
        plt.clf()
        if len(self.samples) > 3:
            subplots_y = 2
        else:
            subplots_y = 1

        if len(self.samples) > 1 and subplots_y > 1:
            subplots_x = (len(self.samples) + len(self.samples) % 2) // subplots_y
        elif len(self.samples) > 1:
            subplots_x = len(self.samples)
        else:
            subplots_x = 1

        fig, axes = plt.subplots(subplots_y, subplots_x)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        vmin = 0
        vmax = np.max(np.array([sample.plot_data for sample in self.samples]))

        for i in range(subplots_y):
            for j in range(subplots_x):
                number = i * subplots_x + j
                if number > len(self.samples) - 1:
                    break

                if subplots_y > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                assert isinstance(ax, Axes)

                hm = self.samples[number]
                assert isinstance(hm, HeatMap)
                hm.draw_mpl((fig, ax), self.labels[number], vmin, vmax, color_scale, major_labels, minor_labels, show_border)

        fig.suptitle(title, fontstyle='italic')
        fig.set_size_inches(6 * subplots_x, 5 * subplots_y)
        return fig

    def draw_plotly(
            self,
            title: str = None,
            color_scale: str = "Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True,
            facet_cols: int = 3,
    ):
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels
        samples_matrix = np.stack([sample.plot_data for sample in self.samples])

        labels = dict(
            x="Position",
            y="Rank",
            color="Methylation density"
        )

        facet_col = 0
        figure = px.imshow(
            samples_matrix,
            labels=labels,
            title=title,
            aspect="auto",
            color_continuous_scale=color_scale,
            facet_col=facet_col,
            facet_col_wrap=facet_cols if len(self.samples) > facet_cols else len(self.samples)
        )

        # set facet annotations
        figure.for_each_annotation(lambda l: l.update(text=self.labels[int(l.text.split("=")[1])]))

        # disable y ticks
        figure.update_layout(
            yaxis=dict(
                showticklabels=False
            )
        )

        figure = self.__add_flank_lines_plotly(figure, major_labels, minor_labels, show_border)

        return figure

    def save_plot_rds(self, base_filename, compress: bool = False):
        for sample, label in zip(self.samples, self.labels):
            sample.save_plot_rds(f"{remove_extension(base_filename)}_{label}.rds",
                                 compress="gzip" if compress else None)

def test():
    pass