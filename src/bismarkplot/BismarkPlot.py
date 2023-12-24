from __future__ import annotations

import multiprocessing
import os
import re
from pathlib import Path

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

pl.enable_string_cache(True)


class Metagene(BismarkBase):
    def __init__(self, bismark_df: pl.DataFrame, **kwargs):

        super().__init__(bismark_df, **kwargs)

    """
    Stores metagene coverage2cytosine data.
    """
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
        Constructor from Bismark coverage2cytosine output.

        :param file: Path to bismark genomeWide report
        :param genome: polars.Dataframe with gene ranges
        :param up_windows: Number of windows flank regions to split
        :param down_windows: Number of windows flank regions to split
        :param body_windows: Number of windows gene regions to split
        """
        up_windows, body_windows, down_windows = cls.__check_windows(up_windows, body_windows, down_windows)

        file = Path(file)
        cls.__check_exists(file)

        print("Initializing CSV reader.")

        pool = pyarrow.default_memory_pool()
        block_size = (1024 ** 2) * block_size_mb
        reader = CsvReader(file, BismarkOptions(use_threads=use_threads, block_size=block_size), memory_pool=pool)
        pool.release_unused()

        print("Reading Bismark report from", file.absolute())

        file_size = os.stat(file).st_size
        read_batches = 0

        mutate_cols = [
            # density for CURRENT cytosine
            ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(MetageneSchema.sum)
        ]

        bismark_df = None

        pl.enable_string_cache()
        for df in reader:
            df = pl.from_arrow(df)
            df = df.filter((pl.col('count_m') + pl.col('count_um') != 0))

            df = cls.__process_batch(df, genome, mutate_cols, up_windows, body_windows, down_windows, sumfunc)

            if bismark_df is None:
                bismark_df = df
            else:
                bismark_df = bismark_df.extend(df)

            read_batches += 1
            print(
                "Read {read_mb}/{total_mb}Mb | Total RAM usage - {ram_usage}Mb".format(
                    read_mb=round(read_batches * block_size / 1024 ** 2, 1),
                    total_mb=round(file_size / 1024 ** 2, 1),
                    ram_usage=round(bismark_df.estimated_size('mb'), 1)
                ),
                end="\r"
            )

        print(
            "Read {read_mb}/{total_mb}Mb | Total RAM usage - {ram_usage}Mb".format(
                read_mb=round(file_size / 1024 ** 2, 1),
                total_mb=round(file_size / 1024 ** 2, 1),
                ram_usage=round(bismark_df.estimated_size('mb'), 1)
            ),
            end="\r"
        )
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
        up_windows, body_windows, down_windows = cls.__check_windows(up_windows, body_windows, down_windows)

        file = Path(file)
        cls.__check_exists(file)

        bismark_df = None

        # initialize batched reader
        reader = ParquetReader(file.absolute(), use_threads=use_threads)
        pq_file = pq.ParquetFile(file.absolute())

        # batch approximation
        num_row_groups = pq_file.metadata.num_row_groups
        read_row_groups = 0

        df_columns = [
            # density for CURRENT cytosine
            (pl.col('count_m') / pl.col('count_total')).alias('density').cast(MetageneSchema.sum)
        ]

        print(f"Reading from {file}")

        for df in reader:
            df = pl.from_arrow(df)
            df = df.filter(pl.col("count_total") != 0)

            df = cls.__process_batch(df, genome, df_columns, up_windows, body_windows, down_windows, sumfunc)

            if bismark_df is None:
                bismark_df = df
            else:
                bismark_df = bismark_df.extend(df)

            read_row_groups += 1
            print(
                f"\tRead {read_row_groups}/{num_row_groups} batch | Total size - {round(bismark_df.estimated_size('mb'), 1)}Mb RAM",
                end="\r")

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
    def __process_batch(df: pl.DataFrame, genome: pl.DataFrame, df_columns, up_win, gene_win, down_win, sumfunc):
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

        return (
            df.lazy()
            # assign types
            # calculate density for each cytosine
            .with_columns(df_columns)
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
            .groupby(by=['chr', 'strand', 'gene', 'context', 'id', 'fragment'])
            .agg(AGG_EXPR)
            .drop_nulls(subset=['sum'])
        ).collect()

    def filter(self, context: str = None, strand: str = None, chr: str = None):
        """
        :param context: Methylation context (CG, CHG, CHH) to filter (only one).
        :param strand: Strand to filter (+ or -).
        :param chr: Chromosome name to filter.
        :return: Filtered :class:`Bismark`.
        """
        context_filter = self.bismark["context"] == context if context is not None else True
        strand_filter = self.bismark["strand"] == strand if strand is not None else True
        chr_filter = self.bismark["chr"] == chr if chr is not None else True

        metadata = self.metadata
        metadata["context"] = context
        metadata["strand"] = strand

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.bismark.filter(context_filter & strand_filter & chr_filter),
                                  **metadata)

    def resize(self, to_fragments: int = None):
        """
        Modify DataFrame to fewer fragments.

        :param to_fragments: Number of final fragments.
        :return: Resized :class:`Bismark`.
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
                by=['chr', 'strand', 'gene', 'context', 'fragment']
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
        Trim fragments

        :param upstream: Keep upstream region?
        :param downstream: Keep downstream region?
        :return: Trimmed :class:`Bismark`.
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

    def clustering(self, count_threshold = 5, dist_method="euclidean", clust_method="average"):
        """
        Gives an order for genes in specified method.

        *WARNING* - experimental function. May be very slow!

        :param count_threshold: Minimum counts per window
        :param dist_method: Distance method to use. See :meth:`scipy.spatial.distance.pdist`
        :param clust_method: Clustering method to use. See :meth:`scipy.cluster.hierarchy.linkage`
        :return: List of indexes of ordered rows.
        """

        return Clustering(self.bismark, count_threshold, dist_method, clust_method, **self.metadata)

    def line_plot(self, resolution: int = None, stat="wmean"):
        """
        :param resolution: Number of fragments to resize to. Keep None if not needed.
        :return: :class:`LinePlot`.
        """
        bismark = self.resize(resolution)
        return LinePlot(bismark.bismark, stat=stat, **bismark.metadata)

    def heat_map(self, nrow: int = 100, ncol: int = 100, stat="wmean"):
        """
        :param nrow: Number of fragments to resize to. Keep None if not needed.
        :param ncol: Number of columns in the resulting heat-map.
        :return: :class:`HeatMap`.
        """
        bismark = self.resize(ncol)
        return HeatMap(bismark.bismark, nrow, order=None, stat=stat, **bismark.metadata)


# TODO add other type constructors
class MetageneFiles(BismarkFilesBase):
    """
    Stores and plots multiple Bismark data.

    If you want to compare Bismark data with different genomes, create this class with a list of :class:`Bismark` classes.
    """
    @classmethod
    def from_list(
            cls,
            filenames: list[str],
            genome: pl.DataFrame,
            labels: list[str] = None,
            upstream_windows: int = 0,
            gene_windows: int = 2000,
            downstream_windows: int = 0,
            block_size_mb: int = 50,
            use_threads: bool = True
    ):
        """
        Constructor for BismarkFiles. See :meth:`Bismark.from_file`

        :param filenames: List of filenames of files
        :param genome: Same genome file for Bismark files to be aligned to.
        """
        samples = [Metagene.from_bismark(file, genome, upstream_windows, gene_windows,
                                         downstream_windows, block_size_mb, use_threads) for file in filenames]
        return cls(samples, labels)

    def filter(self, context: str = None, strand: str = None, chr: str = None):
        """
        :meth:`Bismark.filter` all BismarkFiles
        """
        return self.__class__([sample.filter(context, strand, chr) for sample in self.samples], self.labels)

    def trim_flank(self, upstream=True, downstream=True):
        """
        :meth:`Bismark.trim_flank` all BismarkFiles
        """
        return self.__class__([sample.trim_flank(upstream, downstream) for sample in self.samples], self.labels)

    def resize(self, to_fragments: int):
        """
        :meth:`Bismark.resize` all BismarkFiles
        """
        return self.__class__([sample.resize(to_fragments) for sample in self.samples], self.labels)

    @pl.StringCache()
    def merge(self):
        """
        If data comes from replicates, this method allows to merge them into single DataFrame by grouping them by position.
        """
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

    def line_plot(self, resolution: int = None, stat: str = "wmean"):
        """
        :class:`LinePlot` for all files.
        """
        return LinePlotFiles([sample.line_plot(resolution, stat) for sample in self.samples], self.labels)

    def heat_map(self, nrow: int = 100, ncol: int = None, stat: str = "wmean"):
        """
        :class:`HeatMap` for all files.
        """
        return HeatMapFiles([sample.heat_map(nrow, ncol, stat) for sample in self.samples], self.labels)

    def violin_plot(self, fig_axes: tuple = None):
        """
        Draws violin plot for Bismark DataFrames.
        :param fig_axes: see :meth:`LinePlot.__init__`
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
        axes.set_ylabel('Methylation density')

        return fig

    def violin_plot_plotly(self, title="", points=None):
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

    def box_plot(self, fig_axes: tuple = None, showfliers=False):
        """
        Draws box plot for Bismark DataFrames.
        :param fig_axes: see :meth:`LinePlot.__init__`
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
        axes.set_ylabel('Methylation density')

        return fig

    def box_plot_plotly(self, title="", points=None):
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
    def __init__(self, bismark_df: pl.DataFrame, stat="wmean", **kwargs):
        """
        Calculates plot data for line-plot.
        """
        super().__init__(bismark_df, **kwargs)

        self.stat = stat

        plot_data = self.__calculate_plot_data(bismark_df, stat)
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

    def __strand_reverse(self, df: pl.DataFrame):
        if self.strand == '-':
            max_fragment = df["fragment"].max()
            return df.with_columns((max_fragment - pl.col("fragment")).alias("fragment")).sort("fragment")
        else:
            return df

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

    def draw(
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
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

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
        figure = go.Figure if figure is None else figure
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

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

            figure.add_traces(traces)

        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %"
        )

        figure = self.flank_lines_plotly(figure, major_labels, minor_labels, show_border)

        return figure


class LinePlotFiles(BismarkFilesBase):
    def draw(
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
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

        plt.clf()
        fig, axes = plt.subplots()
        for lp, label in zip(self.samples, self.labels):
            assert isinstance(lp, LinePlot)
            lp.draw((fig, axes), smooth, label, confidence, linewidth, linestyle, major_labels, minor_labels, show_border)

        return fig

    def draw_plotly(self,
                    smooth: int = 50,
                    confidence: int = 0,
                    major_labels: list[str] = None,
                    minor_labels: list[str] = None,
                    show_border: bool = True
                    ):

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

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

    def draw(
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
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

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
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels

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

    def draw(
            self,
            title: str = None
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
                hm.draw((fig, ax), self.labels[number], vmin, vmax)

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
        minor_labels = ["TSS", "TES"] if minor_labels is None else minor_labels
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
