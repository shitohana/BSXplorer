"""
This module implements Metagene and MetageneFiles classes - high-level interfaces
to read, store and analyze methylation report data.
"""

# TODO: 1. Check and fix clusterization of single or multiple metagenes
# TODO: 2. Revise MetageneFiles.dendrogram
# TODO: 3. Maybe move save_* methods to abstract class
# TODO: 4. Revise MetageneFiles.merge
from __future__ import annotations

import functools
import gzip
import sys
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
import polars as pl
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
    validate_call,
)
from pyreadr import write_rds

from .cluster import ClusterMany, ClusterSingle
from .IO import UniversalReader
from .misc.schemas import ReportSchema
from .misc.types import Allocator, Context, ExistentPath, GenomeDf, Strand
from .misc.utils import (
    CONTEXTS,
    AvailableSumfunc,
    MetageneSchema,
    PatchedModel,
    ReportTypes,
    remove_extension,
)
from .plot import (
    BoxPlot,
    BoxPlotData,
    HeatMap,
    HeatMapData,
    LinePlot,
    LinePlotData,
)
from .process_report import read_report
from .sequence import CytosinesFileCM, SequenceFile


class Metagene(PatchedModel):
    """
    Class for reading, storing, analyzing and visualizing
    methylation data

    Attributes
    ----------
    report_df: pl.DataFrame
        ``polars.DataFrame`` with cytosine methylation status.
    genome:
        ``polars.DataFrame`` of genome to which metagene was aligned to.
    upstream_windows: int, default=0
        Upstream windows number.
    body_windows
        Region body windows number.
    downstream_windows:
        Downstream windows number
    strand: {'+', '-', '.'}, optional
        Defines the strand if metagene was filtered by it.
    context: {'CG', 'CHG', 'CHH'}, optional
        Defines the context if metagene was filtered by it.
    reader: UniversalReader
        Model of universal reader, used to read report.
    sumfunc: str
        Summary function used to calculate density
    fasta: str | Path, optional
        Path to FASTA sequence, which was used to read bedGraph or
        coverage report.

    """

    # Private
    _report_df: pl.DataFrame = PrivateAttr()
    _genome: GenomeDf = PrivateAttr()

    # Reader
    reader: UniversalReader

    # Metagene specific reading
    sumfunc: AvailableSumfunc = Field(default="wmean")
    fasta: Optional[ExistentPath] = (None,)

    # Windows
    upstream_windows: int = Field(ge=0, default=0)
    body_windows: int = Field(gt=0)
    downstream_windows: int = Field(ge=0, default=0)

    # Filters
    strand: Optional[Literal["+", "-", "."]] = Field(
        default=None,
    )
    context: Optional[Literal["CG", "CHG", "CHH"]] = Field(
        default=None,
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @property
    def genome(self) -> pl.DataFrame:
        return self._genome

    @property
    def report_df(self) -> pl.DataFrame:
        return self._report_df

    @report_df.setter
    def report_df(self, value: pl.DataFrame) -> None:
        self._report_df = value

    @computed_field
    @functools.cached_property
    def total_windows(self) -> int:
        return self.upstream_windows + self.downstream_windows + self.body_windows

    @computed_field
    @property
    def _x_ticks(self) -> list[float]:
        return [
            self.upstream_windows / 2,
            self.upstream_windows,
            self.total_windows / 2,
            self.body_windows + self.upstream_windows,
            self.total_windows - (self.downstream_windows / 2),
        ]

    @computed_field
    @property
    def _borders(self) -> list[float]:
        return [
            self.upstream_windows,
            self.body_windows + self.upstream_windows,
        ]

    def __init__(self, genome: GenomeDf, report_df: pl.DataFrame = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_genome", genome)

        if report_df is None:
            report_df = read_report(
                self.reader,
                self.genome,
                self.upstream_windows,
                self.body_windows,
                self.downstream_windows,
                self.sumfunc,
            )
        object.__setattr__(self, "_report_df", report_df)

    def __len__(self):
        return len(self.report_df)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_report(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        schema: ReportSchema,
        up_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices("up_windows", "uw", "upstream_windows"),
                ge=0,
            ),
        ] = 100,
        body_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices("body_windows", "bw", "body_windows"),
                gt=0,
            ),
        ] = 100,
        down_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices(
                    "down_windows", "dw", "downstream_windows"
                ),
                ge=0,
            ),
        ] = 100,
        use_threads: bool = True,
        *,
        sumfunc: AvailableSumfunc = "wmean",
        block_size_mb: Optional[Annotated[int, Field(gt=0)]] = 100,
        p_value: Optional[Annotated[float, Field(gt=0, lt=1)]] = 0.05,
        fasta: Optional[ExistentPath] = None,
        cytosine_file: Optional[ExistentPath] = None,
        allocator: Allocator = "system",
    ):
        """
        Read methylation report file and construct an instance
        of Metagene.

        Parameters
        ----------
        file: str or pathlib.Path
            Path to methylation report file.
        genome: pl.DataFrame
            ``polars.Dataframe`` with genomic coordinates
            (generated with :class:`Genome`).
        schema: ReportSchema
            One of ReportSchema values to specify report format.
        up_windows: int
            Number of windows upstream region to split into.
            Aliases: uw, upstream_windows.
        body_windows: int
            Number of windows body region to split into
            Aliases: bw, body_windows.
        down_windows: int
            Number of windows downstream region to split into
            Aliases: dw, downstream_windows
        use_threads: bool, default=True
            Will reading be multi-threaded or single-threaded.
            If multi-threaded option is used, number of threads
            is defined by `multiprocessing.cpu_count()`
        sumfunc: {'wmean', 'mean', 'min', 'max', 'median', '1pgeom'}
            Summary function to calculate density for window with.
        block_size_mb: int, optional
            Block size for reading. (Block size ≠ amount of RAM used.
            Reader allocates approximately block size * 20
            memory for reading.)
        p_value: float, optional
            P-value of cytosine methylation for it to be
            considered methylated.
        fasta: str or Path, optional
            Path to FASTA genome sequence file.
        cytosine_file: str or Path, optional
            Path to preprocessed with :class:`Sequence` cytosine
            file.
        allocator: {'system', 'default', 'mimalloc', 'jemalloc'}, optional
            Memory allocation method used for reading.
            Performance depends on your system and hardware and
            should be estimated individually.

        Returns
        -------
        Metagene
            Instance of :class:`Metagene`.

        Raises
        ------
        `pydantic.ValidationError`
            When parameters are incompatible or inappropriate.

        See Also
        --------
        :class:`MetageneFiles` :
            Read and analyze several methylation reports.
        :class:`ChrLevels` :
            Read and analyze chromosome methylation levels.
        :class:`UniversalReader` :
            Read methylation reports and convert them to universal
            schema for further customized analysis or converting
            into another format.

        Notes
        -----
        If both `fasta` and `cytosine_file` are specified, cytosine
        data will be extracted from `fasta` file.

        Examples
        --------

        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>> path = "path/to/CGmap.txt"
        >>> metagene = Metagene.from_report(
        ...     path,
        ...     genome,
        ...     schema=ReportSchema.CGMAP,
        ...     up_windows=100,
        ...     body_windows=200,
        ...     down_windows=100,
        ... )

        """
        if fasta is not None:
            cm = CytosinesFileCM(fasta)
            cm.__enter__()
            cytosine_file = cm.cytosine_path
            SequenceFile(fasta).preprocess_cytosines(cytosine_file)
        # Todo add model validation for universal reader
        # (if bedgraph or coverage, check if cytosine file is present
        reader = UniversalReader(
            file=file,
            report_schema=schema,
            use_threads=use_threads,
            allocator=allocator,
            cytosine_file=cytosine_file,
            methylation_pvalue=p_value,
            block_size_mb=block_size_mb,
        )

        return cls(
            reader=reader,
            genome=genome,
            sumfunc=sumfunc,
            fasta=fasta,
            upstream_windows=up_windows,
            body_windows=body_windows,
            downstream_windows=down_windows,
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def filter(
        self,
        context: Context = None,
        strand: Strand = None,
        chr: Optional[str] = None,
        genome: GenomeDf = None,
        id: Optional[list[str]] = None,
        coords: Optional[list[str]] = None,
    ):
        """
        Filter metagene by predicates or selected regions.

        Parameters
        ----------
        context: {'CG', 'CHG', 'CHH'}, optional
            Methylation context (CG, CHG, CHH) to filter (only one).
        strand: {'+', '-'}, optional
            Strand to filter (+ or -).
        chr: str, optional
            Chromosome name to filter.
        genome: polars.DataFrame, optional
            DataFrame with annotation to filter with (from :class:`Genome`)
        id: list[str], optional
            List of gene ids to filter (should be specified in annotation file)
        coords: list[str], optional
            List of genomic locuses to filter (column 'gene' of report_df)

        Returns
        -------
        Metagene
        """
        model = self.model_dump()
        model["context"] = context
        model["strand"] = strand

        df = self.report_df

        if genome is not None:
            cast_dtypes = {"chr": pl.Utf8, "strand": pl.Utf8}
            cast_cat_dtypes = {"chr": pl.Categorical, "strand": pl.Categorical}
            df = (
                df.cast(cast_dtypes)
                .join(
                    genome.cast(cast_dtypes).select(["chr", "start"]),
                    on=["chr", "start"],
                )
                .cast(cast_cat_dtypes)
            )
        if id is not None:
            df = df.filter(pl.col("id").cast(pl.String).is_in(id))
        if coords is not None:
            df = df.filter(pl.col("gene").is_in(coords))
        if context is not None:
            df = df.filter(context=context)
        if strand is not None:
            df = df.filter(strand=strand)
        if context is not None:
            df = df.filter(chr=chr)
        copy = self.model_copy(deep=True)
        copy.report_df = df
        return copy

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def resize(self, to_fragments: Annotated[Optional[int], Field(gt=0)] = None):
        """
        Mutate DataFrame to fewer fragments.

        Parameters
        ----------
        to_fragments: int, optional
            Number of TOTAL (upstream + body + downstream) fragments per gene.
            The ratio of flanking regions to body region fragment number is
            preserved in this operation.

        Returns
        -------
        Metagene

        Notes
        -----
        To resize the metagene data, mean value of methylation for fragments
        is calculated. For example, if there were 100 windows and the metagene
        is being resized to 20 - mean density of methylation for each sequential
        fragments is being calculated.

        Examples
        --------
        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = "path/to/bismark.CX_report.txt"
        >>> metagene = Metagene.from_bismark(
        ...     path,
        ...     genome,
        ...     up_windows=500,
        ...     body_windows=1000,
        ...     down_windows=500,
        ... )
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
        if (
            self.upstream_windows is not None
            and self.body_windows is not None
            and self.downstream_windows is not None
        ):
            from_fragments = self.total_windows
        else:
            from_fragments = self.report_df["fragment"].max() + 1

        if to_fragments is None or from_fragments <= to_fragments:
            return self

        resized = (
            self.report_df.lazy()
            .with_columns(
                ((pl.col("fragment") / from_fragments) * to_fragments)
                .floor()
                .cast(MetageneSchema.fragment)
            )
            .group_by(["chr", "strand", "start", "gene", "id", "context", "fragment"])
            .agg([pl.sum("sum").alias("sum"), pl.sum("count").alias("count")])
        ).collect()

        metadata = self.model_dump()
        metadata["upstream_windows"] = metadata["upstream_windows"] // (
            from_fragments // to_fragments
        )
        metadata["downstream_windows"] = metadata["downstream_windows"] // (
            from_fragments // to_fragments
        )
        metadata["body_windows"] = metadata["body_windows"] // (
            from_fragments // to_fragments
        )

        copy = self.model_copy(deep=True)
        copy.report_df = resized
        return copy

    def trim_flank(self, upstream=True, downstream=True) -> Metagene:
        """
        Trim Metagene flanking regions.

        Parameters
        ----------
        upstream: bool
            Trim upstream region.
        downstream: bool
            Trim downstream region.

        Returns
        -------
        Metagene

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
        metadata = self.model_dump()
        if downstream:
            trimmed = trimmed.filter(
                pl.col("fragment") < self.upstream_windows + self.body_windows
            )
            metadata["downstream_windows"] = 0

        if upstream:
            trimmed = (
                trimmed.filter(pl.col("fragment") > self.upstream_windows - 1)
                .with_columns(pl.col("fragment") - self.upstream_windows)
                .collect()
            )
            metadata["upstream_windows"] = 0

        copy = self.model_copy(deep=True)
        copy.report_df = trimmed
        return copy

    # TODO finish annotation
    def cluster(
        self, count_threshold: int = 5, na_rm: float | None = None
    ) -> ClusterSingle:
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
        --------
        ClusterSingle : For possible analysis options
        """
        return ClusterSingle(self, count_threshold, na_rm)

    def save_rds(self, filename, compress: bool = False) -> None:
        """
        Save Metagene in RDS format.

        Parameters
        ----------
        filename
            Path for file.
        compress
            Whether to compress to gzip or not.
        """
        write_rds(
            filename, self.report_df.to_pandas(), compress="gzip" if compress else None
        )

    def save_tsv(self, filename, compress=False) -> None:
        """
        Save Metagene as TSV file.

        Parameters
        ----------
        filename
            Path for file.
        compress
            Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.report_df.write_csv(file, separator="\t")
        else:
            self.report_df.write_csv(filename, separator="\t")

    def line_plot(
        self,
        resolution: int = None,
        stat="wmean",
        smooth: int = 50,
        confidence: float = 0.0,
        merge_strands: bool = True,
    ) -> LinePlot:
        """
        Create :class:`LinePlot`. To generate values for plot
        weightened mean (or specified by ``stat``) density of each fragments
        between all regions is calculated

        Parameters
        ----------
        confidence: float, default=0.0
            Probability for confidence bands (e.g. 0.95). Set 0
            to disable confidence bands.
        smooth: int, default=0
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing).
        resolution: int, optional, default=None
            Number of fragments to resize to. Keep None if no resize is needed.
        stat: {'mean', 'wmean', 'log', 'wlog', 'qN'}, default='wmean'
            Summary function to use for plot.
        merge_strands: bool, default=True
            Does negative strand need to be reversed

        Returns
        -------
        LinePlot

        Notes
        -----
        -   ``mean`` – Default mean between bins, when count of cytosine residues in bin
            IS NOT taken into account
        -   ``wmean`` – Weighted mean between bins. Cytosine residues in bin IS taken
            into account
        -   ``log`` – NOT weighted geometric mean.
        -   ``wlog`` - Weighted geometric mean.
        -   ``qN`` – Return quantile by ``N`` percent (e.g. "``q50``")

        See Also
        --------
            :doc:`LinePlot example<../../markdowns/test>`
            LinePlot : For more information about plotting parameters.
            contexts_line_plot : Comparative plot of methylation contexts

        Examples
        --------
        Firstly we need to initialize Metagene class

        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = "path/to/bismark.CX_report.txt"
        >>> metagene = Metagene.from_bismark(
        ...     path,
        ...     genome,
        ...     up_windows=500,
        ...     body_windows=1000,
        ...     down_windows=500,
        ... )

        Next we can optionally filter metagene by context and strand.

        >>> filtered = metagene.filter(context="CG", strand="-")

        And LinePlot can be created

        >>> lp = filtered.line_plot()
        >>> figure = lp.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/lineplot/lp_ara_mpl.png

        No filtering is suitable too. Then LinePlot will visualize all methylation
        contexts.

        >>> lp = metagene.line_plot()
        >>> figure = lp.draw_mpl()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_multi_mpl.png

        You can use Plotly version for all plots as well.

        >>> figure = lp.draw_plotly()
        >>> figure.show()

        .. image:: ../../images/lineplot/ara_multi_plotly.png

        """  # noqa: E501
        lp_data = LinePlotData.from_metagene_df(
            self.report_df, self.model_dump(), smooth, confidence, stat, merge_strands
        )
        return LinePlot(lp_data)

    def contexts_line_plot(
        self,
        resolution: int = None,
        stat="wmean",
        smooth: int = 50,
        confidence: float = 0.0,
        merge_strands: bool = True,
    ) -> LinePlot:
        """
        Generate comparative line plot of methylation contexts

        Parameters
        ----------
        confidence: float, default=0.0
            Probability for confidence bands (e.g. 0.95). Set 0
            to disable confidence bands.
        smooth: int, default=0
            Number of windows for
            `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter
            (set 0 for no smoothing).
        resolution: int, optional, default=None
            Number of fragments to resize to. Keep None if no resize is needed.
        stat: {'mean', 'wmean', 'log', 'wlog', 'qN'}, default='wmean'
            Summary function to use for plot.
        merge_strands: bool, default=True
            Does negative strand need to be reversed

        Returns
        -------
        LinePlot
        """
        lp_data = [
            filtered.line_plot_data(
                resolution, smooth, confidence, stat, merge_strands, context
            )
            for context, filtered in zip(
                CONTEXTS, map(lambda context: self.filter(context=context), CONTEXTS)
            )
            if len(filtered) > 0
        ]

        return LinePlot(lp_data)

    def heat_map(
        self,
        nrow: int = 100,
        ncol: int = 100,
    ) -> HeatMap:
        """
        Create :class:`HeatMap`. To generate heatmap values, genes
        are ranked by their total methylation density value; split into
        ``nrow`` groups; for each row group weighted average of methylation
        density for each fragment is calculated.


        Parameters
        ----------
        nrow: int
            Number of rows in the resulting heat-map.
        ncol: int
            Number of fragments to resize to. Keep None if no resize is needed.


        Returns
        -------
        HeatMap

        Examples
        --------
        Firstly we need to initialize Metagene class

        >>> genome = Genome.from_gff("path/to/genome.gff").gene_body()
        >>>
        >>> path = "path/to/bismark.CX_report.txt"
        >>> metagene = Metagene.from_bismark(
        ...     path,
        ...     genome,
        ...     up_windows=500,
        ...     body_windows=1000,
        ...     down_windows=500,
        ... )

        Next we need to (in contrast with :meth:`Metagene.line_plot`) filter metagene by
         context and strand.

        >>> filtered = metagene.filter(context="CG", strand="-")

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
        hm_data = HeatMapData.from_metagene_df(
            self.report_df, self.model_dump(), nrow, ncol
        )
        return HeatMap(hm_data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def box_plot_data(
        self,
        filter_context: Optional[Literal["CG", "CHG", "CHH"]] = None,
        label: str = "",
    ):
        df = (
            self.filter(context=filter_context).report_df
            if filter_context is not None
            else self.report_df
        )

        if not df.is_empty():
            data = df.group_by(["chr", "start"]).agg(
                [
                    pl.first("gene").alias("locus"),
                    pl.first("strand"),
                    pl.first("id"),
                    (pl.sum("sum") / pl.sum("count")).alias("density"),
                ]
            )

            return BoxPlotData(
                data["density"].to_list(),
                label,
                data["locus"].to_list(),
                data["id"].to_list(),
            )
        else:
            return BoxPlotData.empty(label)

    def context_box_plot(self):
        data = [
            BoxPlotData.from_metagene_df(self.report_df, context, context)
            for context in CONTEXTS
        ]
        return BoxPlot(data)

    @classmethod
    def from_bismark(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        up_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        down_windows: Annotated[int, Field(ge=0)] = 100,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = True,
        sumfunc: AvailableSumfunc = "wmean",
    ):
        """
        .. deprecated:: 1.3
            Will be removed and replaced by `from_report`
            for easier implementation of custom methylation report formats.
        """
        return cls.from_report(
            file,
            genome,
            ReportSchema.BISMARK,
            up_windows,
            body_windows,
            down_windows,
            use_threads,
            sumfunc=sumfunc,
            block_size_mb=block_size_mb,
        )

    @classmethod
    def from_cgmap(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        up_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        down_windows: Annotated[int, Field(ge=0)] = 100,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = True,
        sumfunc: AvailableSumfunc = "wmean",
    ):
        """
        .. deprecated:: 1.3
            Will be removed and replaced by `from_report`
            for easier implementation of custom methylation report formats.
        """
        return cls.from_report(
            file,
            genome,
            ReportSchema.CGMAP,
            up_windows,
            body_windows,
            down_windows,
            use_threads,
            sumfunc=sumfunc,
            block_size_mb=block_size_mb,
        )

    @classmethod
    def from_binom(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        up_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        down_windows: Annotated[int, Field(ge=0)] = 100,
        p_value: Annotated[float, Field(gt=0, lt=1)] = 0.05,
        use_threads: bool = True,
    ):
        """
        .. deprecated:: 1.3
            Will be removed and replaced by `from_report`
            for easier implementation of custom methylation report formats.
        """
        return cls.from_report(
            file,
            genome,
            ReportSchema.BINOM,
            up_windows,
            body_windows,
            down_windows,
            use_threads,
            p_value=p_value,
        )

    @classmethod
    def from_coverage(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        fasta: ExistentPath,
        up_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        down_windows: Annotated[int, Field(ge=0)] = 100,
        sumfunc: AvailableSumfunc = "wmean",
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = True,
    ):
        """
        .. deprecated:: 1.3
            Will be removed and replaced by `from_report`
            for easier implementation of custom methylation report formats.
        """
        return cls.from_report(
            file,
            genome,
            ReportSchema.COVERAGE,
            up_windows,
            body_windows,
            down_windows,
            use_threads,
            fasta=fasta,
            sumfunc=sumfunc,
            block_size_mb=block_size_mb,
        )

    @classmethod
    def from_bedgraph(
        cls,
        file: ExistentPath,
        genome: GenomeDf,
        fasta: ExistentPath,
        up_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        down_windows: Annotated[int, Field(ge=0)] = 100,
        sumfunc: AvailableSumfunc = "wmean",
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = True,
    ):
        """
        .. deprecated:: 1.3
            Will be removed and replaced by `from_report`
            for easier implementation of custom methylation report formats.
        """
        return cls.from_report(
            file,
            genome,
            ReportSchema.BEDGRAPH,
            up_windows,
            body_windows,
            down_windows,
            use_threads,
            fasta=fasta,
            sumfunc=sumfunc,
            block_size_mb=block_size_mb,
        )


class MetageneFiles(PatchedModel):
    """
    Class for reading, storing, analyzing and visualizing multiple
    methylation reports at once.

    Attributes
    ----------
    samples: list[Metagene]
        List of :class:`Metagene` instances.
    labels: list[str], optional
        List of sample labels.
    """

    samples: list[Metagene]
    labels: Optional[list[str]] = Field(default_factory=list)

    @field_validator("samples")
    @classmethod
    def _validate_saples(cls, samples: list[Metagene]):
        assert (
            functools.reduce(
                lambda a, b: a == b,
                map(lambda sample: sample.upstream_windows, samples),
            )
            and functools.reduce(
                lambda a, b: a == b, map(lambda sample: sample.body_windows, samples)
            )
            and functools.reduce(
                lambda a, b: a == b,
                map(lambda sample: sample.downstream_windows, samples),
            )
        ), ValueError("Samples have different number of windows!")
        return samples

    @model_validator(mode="after")
    def _validate_values(self) -> Any:
        if not self.labels:
            self.labels = list(map(str, range(len(self.samples))))
        assert len(self.labels) == self.samples, ValueError(
            "Samples and labels lists should be equal length!"
        )
        return self

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_list(
        cls,
        filenames: list[ExistentPath],
        genomes: Union[GenomeDf, list[GenomeDf]],
        schema: Union[ReportSchema, list[ReportSchema]],
        labels: Optional[list[str]] = None,
        up_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices("up_windows", "uw", "upstream_windows"),
                ge=0,
            ),
        ] = 100,
        body_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices("body_windows", "bw", "body_windows"),
                gt=0,
            ),
        ] = 100,
        down_windows: Annotated[
            int,
            Field(
                validation_alias=AliasChoices(
                    "down_windows", "dw", "downstream_windows"
                ),
                ge=0,
            ),
        ] = 100,
        use_threads: bool = True,
        *,
        report_type: Optional[ReportTypes] = None,
        block_size_mb: Optional[Annotated[int, Field(gt=0)]] = 100,
        sumfunc: AvailableSumfunc = "wmean",
        p_value: Optional[Annotated[float, Field(gt=0, lt=1)]] = 0.05,
        fasta: Optional[ExistentPath] = None,
        cytosine_file: Optional[ExistentPath] = None,
        allocator: Allocator = "system",
    ) -> MetageneFiles:
        """
        Create istance of :class:`MetageneFiles` from list of paths.

        Parameters
        ----------
        filenames: list[str or pathlib.Path]
            List of methylation report paths to read from
        genomes: pl.DataFrame or list[pl.DataFrame]
            Annotation DataFrame or list of Annotations (may be different annotations)
        schema: ReportSchema or list[ReportSchema]
            One or list of ReportSchema(s) values to specify report format(s).
        labels: list[str]
            List of sample labels.
        up_windows: int
            Number of windows upstream region to split into.
            Aliases: uw, upstream_windows.
        body_windows: int
            Number of windows body region to split into
            Aliases: bw, body_windows.
        down_windows: int
            Number of windows downstream region to split into
            Aliases: dw, downstream_windows
        report_type
            .. deprecated:: 1.3
               Will be removed and replaced by ``schema`` param
            Type of input report. Possible options: ``bismark``, ``cgmap``, ``binom``,
            ``bedgraph``, ``coverage``.
        sumfunc: {'wmean', 'mean', 'min', 'max', 'median', '1pgeom'}
            Summary function to calculate density for window with.
        block_size_mb: int, optional
            Block size for reading. (Block size ≠ amount of RAM used.
            Reader allocates approximately block size * 20
            memory for reading.)
        p_value: float, optional
            P-value of cytosine methylation for it to be
            considered methylated.
        fasta: str or Path, optional
            Path to FASTA genome sequence file.
        cytosine_file: str or Path, optional
            Path to preprocessed with :class:`Sequence` cytosine
            file.
        allocator: {'system', 'default', 'mimalloc', 'jemalloc'}, optional
            Memory allocation method used for reading.
            Performance depends on your system and hardware and
            should be estimated individually.

        Returns
        -------
        MetageneFiles
            Instance of :class:`MetageneFiles`

        See Also
        --------
        Metagene

        Examples
        --------
        Initialization using :meth:`MetageneFiles.from_list`

        >>> ara_genome = Genome.from_gff(
        ...     "/path/to/arath.gff"
        ... ).gene_body(min_length=2000)
        >>> brd_genome = Genome.from_gff(
        ...     "/path/to/bradi.gff"
        ... ).gene_body(min_length=2000)
        >>>
        >>> metagenes = MetageneFiles.from_list(
        ...     ["path/to/bradi.txt", "path/to/ara.txt"],
        ...     [brd_genome, ara_genome],
        ...     [ReportSchema.BISMARK, ReportSchema.CGMAP]
        ...     ["BraDi", "AraTh"],
        ...     up_windows=250,
        ...     body_windows=500,
        ...     down_windows=250,
        ... )

        :class:`MetageneFiles` can be initialized explicitly:

        >>> metagene_ara = Metagene.from_bismark(
        ...     "path/to/ara.txt",
        ...     ara_genome,
        ...     up_windows=250,
        ...     body_windows=500,
        ...     down_windows=250,
        ... )
        >>> metagene_brd = Metagene.from_bismark(
        ...     "path/to/ara.txt",
        ...     ara_genome,
        ...     up_windows=250,
        ...     body_windows=500,
        ...     down_windows=250,
        ... )
        >>> metagenes = MetageneFiles(
        ...     samples=[metagene_brd, metagene_ara],
        ...     labels=["BraDi", "AraTh"],
        ... )

        The resulting objects will be identical.

        Warnings
        --------
        When :class:`MetageneFiles` is initialized explicitly, number of windows needs
        or be the same in evety sample
        """
        if report_type is not None:
            schema = ReportSchema.__getitem__(report_type.upper())

        if not isinstance(genomes, list):
            genomes = [genomes] * len(filenames)
        else:
            if len(genomes) != len(filenames):
                raise AttributeError(
                    "Number of genomes and filenames provided does not match"
                )
        if not isinstance(schema, list):
            schema = [schema] * len(filenames)

        samples: list[Metagene] = []
        for file, genome, schema in zip(filenames, genomes, schema):
            sample = Metagene.from_report(
                file,
                genome,
                schema,
                up_windows,
                body_windows,
                down_windows,
                use_threads,
                sumfunc=sumfunc,
                block_size_mb=block_size_mb,
                p_value=p_value,
                fasta=fasta,
                cytosine_file=cytosine_file,
                allocator=allocator,
            )
            samples.append(sample)

        return cls(samples=samples, labels=labels)

    def filter(
        self,
        context: str = None,
        strand: str = None,
        chr: str = None,
        genome: pl.DataFrame = None,
        id: list[str] = None,
        coords: list[str] = None,
    ) -> MetageneFiles:
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
        id
            List of gene ids to filter (should be specified in annotation file)
        coords
            List of genomic locuses to filter (column 'gene' of report_df)

        Returns
        -------
            Filtered :class:`MetageneFiles`.

        See Also
        --------
        Metagene.filter : For examples.
        """
        return self.__class__(
            [
                sample.filter(context, strand, chr, genome, id, coords)
                for sample in self.samples
            ],
            self.labels,
        )

    def trim_flank(self, upstream=True, downstream=True):
        """
        Trim Metagenes flanking regions.

        Parameters
        ----------
        upstream: bool
            Trim upstream region.
        downstream: bool
            Trim downstream region.

        Returns
        -------
            Trimmed :class:`Metagene`

        See Also
        --------
        Metagene.trim_flank : For examples.
        """
        return self.__class__(
            [sample.trim_flank(upstream, downstream) for sample in self.samples],
            self.labels,
        )

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
        return self.__class__(
            [sample.resize(to_fragments) for sample in self.samples], self.labels
        )

    def merge(self):
        """
        Merge :class:`MetageneFiles` into single :class:`Metagene`.

        Warnings
        --------
        **ALL** metagenes in :class:`MetageneFiles` need to be aligned to the same
        annotation.


        Returns
        -------
            Instance of :class:`Metagene`, with merged replicates.
        """
        pl.enable_string_cache()

        merged = (
            pl.concat([sample.report_df for sample in self.samples])
            .lazy()
            .group_by(
                ["strand", "context", "chr", "start", "fragment"], maintain_order=True
            )
            .agg(
                [
                    pl.sum("sum").alias("sum"),
                    pl.first("gene"),
                    pl.first("id"),
                    pl.sum("count").alias("count"),
                ]
            )
            .select(self.samples[0].report_df.columns)
        ).collect()

        return Metagene(
            merged,
            upstream_windows=self.samples[0].upstream_windows,
            downstream_windows=self.samples[0].downstream_windows,
            body_windows=self.samples[0].body_windows,
        )

    def save_rds(self, base_filename, compress: bool = False, merge: bool = False):
        """
        Save Metagene in Rds.

        Parameters
        ----------
        base_filename
            Base path for file (final path will be ``base_filename+label.rds``).
        compress
            Whether to compress to gzip or not.
        merge
            Do samples need to be merged into single :class:`Metagene` before saving.
        """
        if merge:
            merged = pl.concat(
                [
                    sample.report_df.lazy().with_columns(pl.lit(label))
                    for sample, label in zip(self.samples, self.labels)
                ]
            )
            write_rds(
                base_filename, merged.to_pandas(), compress="gzip" if compress else None
            )
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_rds(
                    f"{remove_extension(base_filename)}_{label}.rds",
                    compress="gzip" if compress else None,
                )

    def save_tsv(self, base_filename, compress: bool = False, merge: bool = False):
        """
        Save Metagenes in TSV.

        Parameters
        ----------
        base_filename
            Base path for file (final path will be ``base_filename+label.tsv``).
        compress
            Whether to compress to gzip or not.
        merge
            Do samples need to be merged into single :class:`Metagene` before saving.
        """
        if merge:
            merged = pl.concat(
                [
                    sample.report_df.lazy().with_columns(pl.lit(label))
                    for sample, label in zip(self.samples, self.labels)
                ]
            )
            if compress:
                with gzip.open(base_filename + ".gz", "wb") as file:
                    # noinspection PyTypeChecker
                    merged.write_csv(file, separator="\t")
            else:
                merged.write_csv(base_filename, separator="\t")
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_tsv(
                    f"{remove_extension(base_filename)}_{label}.tsv", compress=compress
                )

    def line_plot(
        self,
        resolution: int = None,
        smooth: int = 50,
        confidence: float = 0.0,
        stat: str = "wmean",
        merge_strands: bool = True,
    ):
        """
        Create :class:`LinePlotFiles` method.

        Parameters
        ----------
        resolution
            Number of fragments to resize to. Keep None if no resize is needed.
        stat
            Summary function to use for plot. Possible options: ``mean``, ``wmean``,
            ``log``, ``wlog``, ``qN``
        merge_strands
            Does negative strand need to be reversed
        confidence
            Probability for confidence bands (e.g. 0.95)
        smooth
            Number of windows for
            `SavGol <
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
            >`_ filter
            (set 0 for no smoothing). Applied only if `flank_windows` and `body_windows`
            params are specified.

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
            sample.line_plot_data(
                resolution, smooth, confidence, stat, merge_strands, label
            )
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
            HeatMapData.from_metagene_df(
                sample.report_df, sample.dump_model(), nrow, ncol, label
            )
            for sample, label in zip(self.samples, self.labels)
        ]
        return HeatMap(hm_data)

    def box_plot_data(self, filter_context: Literal["CG", "CHG", "CHH"] | None = None):
        return [
            BoxPlotData.from_metagene_df(sample.report_df, filter_context, label)
            for label, sample in zip(self.labels, self.samples)
        ]

    def box_plot(self):
        return BoxPlot(self.box_plot_data())

    def dendrogram(self, q: float = 0.75, **kwargs):
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
                sample.report_df.group_by(["chr", "strand", "start", "gene"])
                .agg((pl.sum("sum") / pl.sum("count")).alias("density"))
                .with_columns(pl.lit(label).alias("label"))
            )

        gene_set = set.intersection(
            *[set(stat["gene"].to_list()) for stat in gene_stats]
        )

        if len(gene_set) < 1:
            raise ValueError(
                "Region set intersection is empty. Are Metagenes read with same genome?"
            )

        gene_stats = [
            stat.filter(pl.col("gene").is_in(list(gene_set))) for stat in gene_stats
        ]

        dendro_data = pl.concat(gene_stats).pivot(
            values="density", columns="label", index="gene", aggregate_function="mean"
        )

        matrix = dendro_data.select(pl.all().exclude("gene")).to_pandas()

        # Filter by variance
        if q > 0:
            var = matrix.to_numpy().var(1)
            matrix = matrix[var > np.quantile(var, q)]
        if "seaborn" not in sys.modules:
            import seaborn as sns
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
