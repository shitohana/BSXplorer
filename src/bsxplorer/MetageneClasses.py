from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

from .Plots import (
    BoxPlot, savgol_line, LinePlotData, LinePlot, plot_stat_expr, HeatMapData, HeatMap,
    BoxPlotData
)
from .SeqMapper import CytosinesFileCM, SequenceFile
from .Base import (
    MetageneBase,
    MetageneFilesBase,
    read_metagene,
    validate_metagene_args
)
from .Clusters import ClusterSingle, ClusterMany
from .UniversalReader_batches import ReportTypes
from .utils import MetageneSchema, AvailableSumfunc, CONTEXTS
from .GenomeClass import Genome
from .UniversalReader_classes import UniversalReader


class Metagene(MetageneBase):
    """
    Stores Metagene data.
    """

    def __init__(self, report_df: pl.DataFrame, **kwargs):

        super().__init__(report_df, **kwargs)

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
            sumfunc: AvailableSumfunc = "wmean"
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

        reader = UniversalReader(**(locals() | dict(report_type="bismark")))

        args = validate_metagene_args(genome, up_windows, body_windows, down_windows, sumfunc)
        report_df = read_metagene(**(locals() | args))

        return cls(report_df,
                   upstream_windows=args["upstream_windows"],
                   gene_windows=args["body_windows"],
                   downstream_windows=args["downstream_windows"])

    @classmethod
    def from_cgmap(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            block_size_mb: int = 100,
            use_threads: bool = True,
            sumfunc: AvailableSumfunc = "wmean"
    ):
        """
        Constructor for Metagene class from BSSeeker2 CGmap file.

        Parameters
        ----------
        file
            Path to CGmap file report.
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
        >>> path = 'path/to/CGmap.txt'
        >>> metagene = Metagene.from_cgmap(path, genome, up_windows=500, body_windows=1000, down_windows=500)
        """

        reader = UniversalReader(**(locals() | dict(report_type="cgmap")))

        args = validate_metagene_args(genome, up_windows, body_windows, down_windows, sumfunc)
        report_df = read_metagene(**(locals() | args))

        return cls(report_df,
                   upstream_windows=args["upstream_windows"],
                   gene_windows=args["body_windows"],
                   downstream_windows=args["downstream_windows"])

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
        reader = UniversalReader(file, "binom", methylation_pvalue=p_value, use_threads=use_threads)

        args = validate_metagene_args(genome, up_windows, body_windows, down_windows, "wmean")
        report_df = read_metagene(**(locals() | args))

        return cls(report_df,
                   upstream_windows=args["upstream_windows"],
                   gene_windows=args["body_windows"],
                   downstream_windows=args["downstream_windows"])

    @classmethod
    def from_bedGraph(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            fasta: str | Path,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            sumfunc: AvailableSumfunc = "wmean",
            block_size_mb: int = 30,
            use_threads: bool = True,
            save_preprocessed: bool = False,
            temp_dir: str = Path.cwd()
    ):
        """
        Constructor for Metagene class from ``.bedGraph`` file.

        Parameters
        ----------
        file
            Path to ``.bedGraph`` file.
        genome
            ``polars.Dataframe`` with gene ranges (from :class:`Genome`)
        fasta
            Path to FASTA genome sequence file or path to preprocessed with :class:`Sequence` cytosine file.
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
            Does preprocessed sequence file need to be saved
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`

        Returns
        -------
        Metagene

        Examples
        --------
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = 'path/to/report.bedGraph'
        >>> fasta = 'path/to/sequence.fa'
        >>> metagene = Metagene.from_bedGraph(path, genome, fasta, up_windows=500, body_windows=1000, down_windows=500)
        """
        with CytosinesFileCM(fasta) as cm:
            cytosine_file = cm.cytosine_path
            if not cm.is_cytosine:
                SequenceFile(fasta).preprocess_cytosines(cytosine_file)

            reader = UniversalReader(file, report_type="bedgraph", use_threads=use_threads, cytosine_file=cytosine_file,
                                     block_size_mb=block_size_mb)

            args = validate_metagene_args(genome, up_windows, body_windows, down_windows, sumfunc)
            report_df = read_metagene(reader, genome, up_windows, body_windows, down_windows, sumfunc)

        return cls(report_df,
                   upstream_windows=args["upstream_windows"],
                   gene_windows=args["body_windows"],
                   downstream_windows=args["downstream_windows"])

    @classmethod
    def from_coverage(
            cls,
            file: str | Path,
            genome: pl.DataFrame,
            fasta: str | Path,
            up_windows: int = 0,
            body_windows: int = 2000,
            down_windows: int = 0,
            sumfunc: AvailableSumfunc = "wmean",
            block_size_mb: int = 30,
            use_threads: bool = True,
            save_preprocessed: bool = False,
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
        fasta
            Path to FASTA genome sequence file or path to preprocessed with :class:`Sequence` cytosine file.
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
            Does preprocessed sequence file need to be saved
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`


        Returns
        -------
        Metagene

        Examples
        --------

        >>> path = 'path/to/report.cov'
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> fasta = 'path/to/sequence.fa'
        >>> metagene = Metagene.from_coverage(path, genome, fasta, up_windows=500, body_windows=1000, down_windows=500)
        """

        with CytosinesFileCM(fasta) as cm:
            cytosine_file = cm.cytosine_path
            if not cm.is_cytosine:
                SequenceFile(fasta).preprocess_cytosines(cytosine_file)

            reader = UniversalReader(file, report_type="bedgraph", use_threads=use_threads, cytosine_file=cytosine_file,
                                     block_size_mb=block_size_mb)

            args = validate_metagene_args(genome, up_windows, body_windows, down_windows, sumfunc)
            report_df = read_metagene(reader, genome, up_windows, body_windows, down_windows, sumfunc)

        return cls(report_df,
                   upstream_windows=args["upstream_windows"],
                   gene_windows=args["body_windows"],
                   downstream_windows=args["downstream_windows"])

    # Todo Check and update
    def filter(
            self,
            context: Literal["CG", "CHG", "CHH", None] = None,
            strand: Literal["+", "-", None] = None,
            chr: str = None,
            genome: pl.DataFrame = None,
            id: list[str] = None,
            coords: list[str] = None
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
            DataFrame with annotation to filter with (from :class:`Genome`)

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

        context_filter = self.report_df["context"] == context if context is not None else True
        strand_filter = self.report_df["strand"] == strand if strand is not None else True
        chr_filter = self.report_df["chr"] == chr if chr is not None else True

        metadata = self.metadata
        metadata["context"] = context
        metadata["strand"] = strand

        if genome is not None:
            def genome_filter(df: pl.DataFrame):
                cast_dtypes = {"chr": pl.Utf8, "strand": pl.Utf8}
                cast_cat_dtypes = {"chr": pl.Categorical, "strand": pl.Categorical}
                return df.cast(cast_dtypes).join(genome.cast(cast_dtypes).select(["chr", "start"]), on=["chr", "start"]).cast(cast_cat_dtypes)
        else:
            genome_filter = lambda df: df

        if id is not None:
            def id_filter(df: pl.DataFrame):
                return df.filter(pl.col("id").cast(pl.String).is_in(id))
        else:
            id_filter = lambda df: df

        if coords is not None:
            def coords_filter(df: pl.DataFrame):
                return df.filter(pl.col("gene").is_in(coords))
        else:
            coords_filter = lambda df: df

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(
                coords_filter(id_filter(genome_filter(self.report_df.filter(context_filter & strand_filter & chr_filter)))),
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
            from_fragments = self.report_df["fragment"].max() + 1

        if to_fragments is None or from_fragments <= to_fragments:
            return self

        resized = (
            self.report_df.lazy()
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
        trimmed = self.report_df.lazy()
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
        na_rm
            Replace empty windows by specified value.
        count_threshold
            Minimum counts per window

        Returns
        -------
            :class:`ClusterSingle`

        See Also
        -------
        ClusterSingle : For possible analysis options
        """

        return ClusterSingle(self, count_threshold, na_rm)

    @staticmethod
    def _reverse_strand(df, max_fragment):
        return (
                df
                .filter(pl.col("strand") == "-")
                .with_columns((max_fragment - pl.col("fragment")).alias("fragment"))
                .sort("fragment")
            )

    def line_plot_data(
            self,
            resolution: int = None,
            smooth: int = 50,
            confidence: float = 0.,
            stat: str = "wmean",
            merge_strands: bool = True,
            label=""
    ):
        """

        Parameters
        ----------
        confidence
            Probability for confidence bands (e.g. 0.95)
        smooth
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows` params are specified.
        resolution
            Number of fragments to resize to. Keep None if no resize is needed.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``
        merge_strands
            Does negative strand need to be reversed
        label
            Label for the data.

        Returns
        -------
        :class:`LinePlotData`
        """
        resized = self if resolution is None else self.resize(resolution)
        df = resized.report_df

        # Merge strands
        if merge_strands:
            df = df.filter(pl.col("strand") != "-").extend(self._reverse_strand(df, df["fragment"].max()))

        # Apply stats expr
        res = (
            df
            .group_by("fragment")
            .agg([
                plot_stat_expr(stat).alias("density"),
                pl.col("sum"),
                pl.col("count"),
                pl.sum("count").alias("n"),
                (pl.col("sum") / pl.col("count")).mean().alias("average"),
                (pl.col("sum") - (pl.col("sum") / pl.col("count"))).mean().pow(2).alias("variance")
            ])
            .sort("fragment")
        )

        if 0 < confidence < 1 and stat in ["mean", "wmean"]:
            res = (
                res
                .with_columns(
                    (pl.col("variance") / pl.col("n")).sqrt().alias("scale")
                )
                .with_columns(
                    pl.struct(["n", "average", "scale"])
                    .map_elements(
                        lambda field: stats.t.interval(confidence, df=field["n"] - 1, loc=field["average"], scale=field["scale"]),
                        return_dtype=pl.List(pl.Float64)
                    ).alias("interval")
                )
                .with_columns(pl.col("interval").list.to_struct(fields=["lower", "upper"]))
                .unnest("interval")
            )
        elif 0 < confidence < 1 and not (stat in ["mean", "wmean"]):
            raise ValueError("Confidence bands available only for mean and wmean stat parameters.")

        # Fill empty windows
        template = pl.DataFrame(
            dict(fragment=list(range(self.total_windows))),
            schema=dict(fragment=res.schema["fragment"])
        )

        joined = template.join(res, on="fragment", how="left")

        # Calculate CI
        lower = None
        upper = None
        if 0 < confidence < 1 and stat in ["mean", "wmean"]:
            upper = savgol_line(joined["upper"].to_numpy(), smooth) * 100
            lower = savgol_line(joined["lower"].to_numpy(), smooth) * 100

        # Smooth and convert to percents
        y = savgol_line(joined["density"].to_numpy(), smooth) * 100
        x = np.arange(len(y))

        return LinePlotData(x, y, resized._x_ticks, resized._borders, lower, upper, label,
                            ["Upstream", "", "Body", "", "Downstream"])

    def line_plot(
            self,
            resolution: int = None,
            stat="wmean",
            smooth: int = 50,
            confidence: float = 0.,
            merge_strands: bool = True
    ) -> LinePlot:
        """
        Create :class:`LinePlot` method.

        Parameters
        ----------
        confidence
            Probability for confidence bands (e.g. 0.95)
        smooth
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows` params are specified.
        resolution
            Number of fragments to resize to. Keep None if no resize is needed.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``, ``log``, ``wlog``, ``qN``
        merge_strands
            Does negative strand need to be reversed

        Returns
        -------
            :class:`LinePlot`

        Notes
        -----
        - ``mean`` – Default mean between bins, when count of cytosine residues in bin IS NOT taken into account

        - ``wmean`` – Weighted mean between bins. Cytosine residues in bin IS taken into account

        - ``log`` – NOT weighted geometric mean.

        - ``wlog`` - Weighted geometric mean.

        - ``qN`` – Return quantile by ``N`` percent (e.g. "``q50``")

        See Also
        --------
            :doc:`LinePlot example<../../markdowns/test>`

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

        >>> figure = lp.draw_plotly()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_multi_plotly.png

        See Also
        --------
        LinePlot : For more information about plottling parameters.
        """
        resized = self.resize(resolution)
        lp_data = resized.line_plot_data(resolution, smooth, confidence, stat, merge_strands)
        return LinePlot(lp_data)

    def contexts_line_plot(
            self,
            resolution: int = None,
            stat="wmean",
            smooth: int = 50,
            confidence: float = 0.,
            merge_strands: bool = True
    ) -> LinePlot:
        resized = self.resize(resolution)

        lp_data = [
            filtered.line_plot_data(resolution, smooth, confidence, stat, merge_strands, context) for context, filtered
            in zip(CONTEXTS, map(lambda context: resized.filter(context=context), CONTEXTS))
            if len(filtered) > 0
        ]

        return LinePlot(lp_data)

    def heat_map_data(
            self,
            nrow: int = 100,
            ncol: int = 100,
            label=None
    ):
        resized = self.resize(ncol)
        report_df = resized.report_df
        # Merge strands
        report_df = report_df.filter(pl.col("strand") != "-").extend(self._reverse_strand(report_df, report_df["fragment"].max()))

        # sort by rows and add row numbers
        hm_data = (
            report_df.lazy()
            .group_by("gene")
            .agg([
                pl.col("fragment"),
                (pl.col("sum") / pl.col("count")).alias("density")
            ])
            .with_columns(
                (pl.col("density").list.sum() / (resized.total_windows + 1)).alias("order")
            )
            .sort('order', descending=True)
            .with_row_count(name='row')
            .with_columns(
                (pl.col('row') / (pl.max("row") + 1) * nrow).floor().alias('row').cast(pl.UInt16)
            )
            .explode(['fragment', 'density'])
            .group_by(["row", "fragment"])
            .agg(pl.mean("density"))
        )

        # prepare full template
        template = (
            pl.LazyFrame(data={"row": list(range(nrow))})
            .with_columns(pl.lit(list(range(0, resized.total_windows))).alias("fragment"))
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
            .group_by('row', maintain_order=True)
            .agg(pl.col('density'))
            .collect()
            ['density']
            .to_list()
        )

        # convert to matrix and percents
        hm_data = np.array(hm_data, dtype=np.float32) * 100

        return HeatMapData(hm_data, resized._x_ticks, resized._borders, label if label is not None else "", ["Upstream", "", "Body", "", "Downstream"])

    def heat_map(
            self,
            nrow: int = 100,
            ncol: int = 100,
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
            :class:`HeatMap`

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

        hm_data = self.heat_map_data(nrow, ncol)
        return HeatMap(hm_data)

    def __str__(self):
        representation = (f'Metagene with {len(self)} windows total.\n'
                          f'Filtered by {self.context} context and {self.strand} strand.\n'
                          f'Upstream windows: {self.upstream_windows}.\n'
                          f'Body windows: {self.gene_windows}.\n'
                          f'Downstream windows: {self.downstream_windows}.\n')

        return representation

    def __repr__(self):
        return self.__str__()

    def box_plot_data(self, filter_context: Literal["CG", "CHG", "CHH"] | None = None, label=""):
        df = self.filter(context=filter_context).report_df if filter_context is not None else self.report_df

        if not df.is_empty():
            data = (
                df
                .group_by(["chr", "start"])
                .agg([
                    pl.first("gene").alias("locus"),
                    pl.first("strand"),
                    pl.first("id"),
                    (pl.sum("sum") / pl.sum("count")).alias("density")
                ])
            )

            return BoxPlotData(
                data["density"].to_list(),
                label,
                data["locus"].to_list(),
                data["id"].to_list()
            )
        else:
            return BoxPlotData.empty(label)

    def context_box_plot(self):
        data = [self.box_plot_data(context, context)  for context in CONTEXTS]

        return BoxPlot(data)


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
            report_type: ReportTypes = "bismark",
            block_size_mb: int = 50,
            use_threads: bool = True,
            sumfunc: AvailableSumfunc = "wmean",
            **kwargs
    ) -> MetageneFiles:
        """
        Create istance of :class:`MetageneFiles` from list of paths.

        Parameters
        ----------
        filenames
            List of filenames to read from
        genomes
            Annotation DataFrame or list of Annotations (may be different annotations)
        labels
            Labels for plots for Metagenes
        up_windows
            Number of windows upstream region to split into
        body_windows
            Number of windows body region to split into
        down_windows
            Number of windows downstream region to split into
        report_type
            Type of input report. Possible options: ``bismark``, ``cgmap``, ``binom``, ``bedgraph``, ``coverage``.
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        sumfunc
            Summary function to calculate density for window with.

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
        read_fnc = {
            "bismark": Metagene.from_bismark,
            "binom": Metagene.from_binom,
            "cgmap": Metagene.from_cgmap,
            "bedgraph": Metagene.from_bedGraph,
            "coverage": Metagene.from_coverage
        }

        if not isinstance(genomes, list):
            genomes = [genomes] * len(filenames)
        else:
            if len(genomes) != len(filenames):
                raise AttributeError("Number of genomes and filenames provided does not match")

        default_args = dict(
            up_windows=up_windows,
            body_windows=body_windows,
            down_windows=down_windows,
            use_threads=use_threads
        )

        if report_type not in ["binom"]:
            default_args |= dict(
                block_size_mb=block_size_mb,
                sumfunc=sumfunc
            )

        samples: list[Metagene] = []
        for file, genome in zip(filenames, genomes):
            sample = read_fnc[report_type](file=file, genome=genome, **default_args, **kwargs)

            samples.append(sample)

        return cls(samples, labels)

    def filter(self, context: str = None, strand: str = None, chr: str = None, genome: pl.DataFrame = None,
               id: list[str] = None, coords: list[str] = None) -> MetageneFiles:
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
        return self.__class__([sample.filter(context, strand, chr, genome, id, coords) for sample in self.samples], self.labels)

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
                pl.concat([sample.report_df for sample in self.samples]).lazy()
                .group_by(["strand", "context", "chr", "gene", "start", "id", "fragment"], maintain_order=True)
                .agg([pl.sum("sum").alias("sum"), pl.sum("count").alias("count")])
                .select(self.samples[0].report_df.columns)
            ).collect()

            return Metagene(merged,
                            upstream_windows=list(upstream_windows)[0],
                            downstream_windows=list(downstream_windows)[0],
                            gene_windows=list(gene_windows)[0])
        else:
            raise Exception("Metadata for merge DataFrames does not match!")

    def line_plot(
            self,
            resolution: int = None,
            smooth: int = 50,
            confidence: float = 0.,
            stat: str = "wmean",
            merge_strands: bool = True
    ):
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
        confidence
            Probability for confidence bands (e.g. 0.95)
        smooth
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows` params are specified.

        Returns
        -------
            :class:`LinePlot`

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
        lp_data = [
            sample.line_plot_data(resolution, smooth, confidence, stat, merge_strands, label)
            for sample, label in zip(self.samples, self.labels)
        ]
        return LinePlot(lp_data)

    def heat_map(self, nrow: int = 100, ncol: int = None) -> HeatMap:
        """
        Create :class:`HeatMapNew` method.

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
            :class:`HeatMap`

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
        hm_data = [
            sample.heat_map_data(nrow, ncol, label)
            for sample, label in zip(self.samples, self.labels)
        ]
        return HeatMap(hm_data)

    def box_plot_data(self, filter_context: Literal["CG", "CHG", "CHH"] | None = None):
        return [sample.box_plot_data(filter_context, label) for label, sample in zip(self.labels, self.samples)]

    # Todo add fixes from issue14-16#fix
    def box_plot(self):
        return BoxPlot(self.box_plot_data())

    def dendrogram(self, q: float = .75, **kwargs):
        """
        Cluster samples by total methylation level of regions and draw dendrogram.

        Parameters
        ----------
        q
            Quantile of regions, which will be clustered
        kwargs
            Keyword arguments for `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_

        Returns
        -------
            ``matplotlib.pyplot.Figure``
        """
        gene_stats = []
        for sample, label in zip(self.samples, self.labels):
            gene_stats.append(
                sample.report_df
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

        fig = sns.clustermap(matrix, **kwargs)
        return fig

    def cluster(self, count_threshold=5, na_rm: float | None = None):
        """
        Cluster samples regions by methylation pattern.

        Parameters
        ----------
        count_threshold
            Minimum number of reads per fragment to include it in analysis.
        na_rm
            Value to fill empty fragments. (None if not full regions need to be deleted)

        Returns
        -------
        ClusterMany
        """
        return ClusterMany(self, count_threshold, na_rm)
