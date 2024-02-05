from __future__ import annotations

import multiprocessing
import os
import typing
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl
import seaborn as sns

from .Plots import LinePlot, LinePlotFiles, HeatMap, HeatMapFiles
from .SeqMapper import Mapper, Sequence
from .Base import (
    MetageneBase, MetageneFilesBase,
    BismarkReportReader, ParquetReportReader, BinomReportReader
)
from .Clusters import ClusterSingle, ClusterMany
from .utils import MetageneSchema
from .GenomeClass import Genome

pl.enable_string_cache(True)


class Metagene(MetageneBase):
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

        report_reader = BismarkReportReader(
            report_file=file,
            genome=genome,
            upstream_windows=up_windows,
            body_windows=body_windows,
            downstream_windows=down_windows,
            use_threads=use_threads,
            sumfunc=sumfunc,
            block_size_mb=block_size_mb
        )

        report_df = report_reader.read()

        return cls(report_df,
                   upstream_windows=report_reader.upstream_windows,
                   gene_windows=report_reader.body_windows,
                   downstream_windows=report_reader.downstream_windows)

    @classmethod
    def from_parquet(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            use_threads=True,
            sumfunc: str = "mean"
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

        report_reader = ParquetReportReader(
            report_file=file,
            genome=genome,
            upstream_windows=up_windows,
            body_windows=body_windows,
            downstream_windows=down_windows,
            use_threads=use_threads,
            sumfunc=sumfunc,
        )

        report_df = report_reader.read()

        return cls(report_df,
                   upstream_windows=report_reader.upstream_windows,
                   gene_windows=report_reader.body_windows,
                   downstream_windows=report_reader.downstream_windows)

    @classmethod
    def from_binom(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            p_value: float = .05,
            use_threads=True,
            sumfunc: None = None
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
        report_reader = BinomReportReader(
            report_file=file,
            genome=genome,
            upstream_windows=up_windows,
            body_windows=body_windows,
            downstream_windows=down_windows,
            use_threads=use_threads,
            sumfunc="mean",
            p_value=p_value
        )

        report_df = report_reader.read()

        return cls(report_df,
                   upstream_windows=report_reader.upstream_windows,
                   gene_windows=report_reader.body_windows,
                   downstream_windows=report_reader.downstream_windows)

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
            block_size_mb: int = 30,
            use_threads: bool = True,
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
        if isinstance(genome, Genome):
            raise TypeError("Genome must be converted into DataFrame (e.g. via Genome.gene_body()).")

        sequence = Sequence.from_fasta(sequence, temp_dir)
        mapped = Mapper.bedGraph(file, sequence, temp_dir,
                                 save_preprocessed, True if save_preprocessed is None else False,
                                 block_size_mb, use_threads)

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
            block_size_mb: int = 30,
            use_threads: bool = True,
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
        if isinstance(genome, Genome):
            raise TypeError("Genome must be converted into DataFrame (e.g. via Genome.gene_body()).")

        sequence = Sequence.from_fasta(sequence, temp_dir)
        mapped = Mapper.coverage(file, sequence, temp_dir,
                                 save_preprocessed, True if save_preprocessed is None else False,
                                 block_size_mb, use_threads)

        return cls.from_parquet(mapped.report_file, genome, up_windows, body_windows, down_windows, sumfunc)

    def filter(
            self,
            context: Literal["CG", "CHG", "CHH", None] = None,
            strand: Literal["+", "-", None] = None,
            chr: str = None,
            genome: pl.DataFrame = None,
            id: list[str] = None
    ) -> Metagene:
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

        if id is not None:
            def id_filter(df: pl.DataFrame):
                return df.filter(pl.col("id").is_in(id))
        else:
            id_filter = lambda df: df

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(
                id_filter(genome_filter(self.bismark.filter(context_filter & strand_filter & chr_filter))),
                **metadata)

    def resize(self, to_fragments: int = None) -> Metagene:
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

    def trim_flank(self, upstream=True, downstream=True) -> Metagene:
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
    def cluster(
            self,
            count_threshold: int = 5,
            na_rm: float | None = None) -> ClusterSingle:
        """
        Cluster regions with hierarchical clustering, by their methylation pattern.

        Parameters
        ----------
        count_threshold
            Minimum counts per window

        Returns
        -------
            :class:`ClusterSingle`

        See Also
        -------
        Clustering : For possible analysis options
        """

        return ClusterSingle(self, count_threshold, na_rm)

    def line_plot(
            self,
            resolution: int = None,
            stat="wmean",
            merge_strands: bool = True
    ) -> LinePlot:
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
        src.bismarkplot.LinePlot

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

        .. image:: ../../images/lineplot/lp_ara_mpl.png

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
    ) -> HeatMap:
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
        src.bismarkplot.HeatMap

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
# todo add fastcluster to dependencies
class MetageneFiles(MetageneFilesBase):
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
            use_threads: bool = True,
            sumfunc: str = "wmean"
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
            "bismark": Metagene.from_bismark,
            "parquet": Metagene.from_parquet,
            "binom": Metagene.from_binom,
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
            args = dict(
                file=file, genome=genome, up_windows=up_windows, body_windows=body_windows, down_windows=down_windows,
                block_size_mb=block_size_mb, use_threads=use_threads, sumfunc=sumfunc
            )
            samples.append(
                read_fnc[report_type](**args))

        return cls(samples, labels)

    def filter(self, context: str = None, strand: str = None, chr: str = None, genome: pl.DataFrame = None,
               id: list[str] = None):
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
        return self.__class__([sample.filter(context, strand, chr, genome, id) for sample in self.samples], self.labels)

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
        downstream_windows = set([md.get("downstream_windows") for md in metadata])

        if len(upstream_windows) == len(downstream_windows) == len(gene_windows) == 1:
            merged = (
                pl.concat([sample.bismark for sample in self.samples]).lazy()
                .group_by(["strand", "context", "chr", "gene", "start", "id", "fragment"], maintain_order=True)
                .agg([pl.sum("sum").alias("sum"), pl.sum("count").alias("count")])
                .select(self.samples[0].bismark.columns)
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
        return LinePlotFiles([sample.line_plot(resolution, stat, merge_strands) for sample in self.samples],
                             self.labels)

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
        data = data.to_pandas().dropna()

        labels = dict(
            context="Context",
            label="",
            density="Methylation density, %"
        )
        figure = px.box(data, x="label", y="density",
                        color="context", points=points,
                        labels=labels, title=title)

        return figure

    def dendrogram(self, q: float = .75):
        gene_stats = []
        for sample, label in zip(self.samples, self.labels):
            gene_stats.append(
                sample.bismark
                .group_by(["chr", "strand", "start", "gene"])
                .agg((pl.sum("sum") / pl.sum("count")).alias("density"))
                .with_columns(pl.lit(label).alias("label"))
            )

        gene_set = set.intersection(*[set(stat["gene"].to_list()) for stat in gene_stats])

        if len(gene_set) < 1:
            raise ValueError("Region set intersection is empty. Are Metagenes read with same genome?")

        gene_stats = [stat.filter(pl.col("gene").is_in(list(gene_set))) for stat in gene_stats]

        dendro_data = pl.concat(gene_stats).pivot(values="density", columns="label", index="gene", aggregate_function="mean")

        matrix = dendro_data.select(pl.all().exclude("gene")).to_pandas()

        # Filter by variance
        if q > 0:
            var = matrix.to_numpy().var(1)
            matrix = matrix[var > np.quantile(var, q)]

        fig = sns.clustermap(matrix, row_cluster=True)
        return fig

    def cluster(self, count_threshold=5, na_rm: float | None = None):
        return ClusterMany(self, count_threshold, na_rm)
