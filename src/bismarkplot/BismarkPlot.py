import re
from multiprocessing import cpu_count

import polars as pl

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.signal import savgol_filter
from scipy import stats

from pandas import DataFrame as pdDataFrame
from pyreadr import write_rds

from src.bismarkplot.base import BismarkBase, BismarkFilesBase
from src.bismarkplot.clusters import Clustering
from src.bismarkplot.utils import remove_extension, approx_batch_num, hm_flank_lines


class Metagene(BismarkBase):
    """
    Stores metagene coverage2cytosine data.
    """
    @classmethod
    def from_file(
            cls,
            file: str,
            genome: pl.DataFrame,
            upstream_windows: int = 0,
            gene_windows: int = 2000,
            downstream_windows: int = 0,
            batch_size: int = 10 ** 6,
            cpu: int = cpu_count(),
            sumfunc: str = "mean"
    ):
        """
        Constructor from Bismark coverage2cytosine output.

        :param cpu: How many cores to use. Uses every physical core by default
        :param file: Path to bismark genomeWide report
        :param genome: polars.Dataframe with gene ranges
        :param upstream_windows: Number of windows flank regions to split
        :param downstream_windows: Number of windows flank regions to split
        :param gene_windows: Number of windows gene regions to split
        :param batch_size: Number of rows to read by one CPU core
        """
        if upstream_windows < 1:
            upstream_windows = 0
        if downstream_windows < 1:
            downstream_windows = 0
        if gene_windows < 1:
            gene_windows = 0

        bismark_df = cls.__read_bismark_batches(file, genome,
                                                upstream_windows, gene_windows, downstream_windows,
                                                batch_size, cpu, sumfunc)

        return cls(bismark_df,
                   upstream_windows=upstream_windows,
                   gene_windows=gene_windows,
                   downstream_windows=downstream_windows)

    @staticmethod
    def __read_bismark_batches(
            file: str,
            genome: pl.DataFrame,
            upstream_windows: int = 500,
            gene_windows: int = 2000,
            downstream_windows: int = 500,
            batch_size: int = 10 ** 7,
            cpu: int = cpu_count(),
            sumfunc: str = "mean"
    ) -> pl.DataFrame:
        cpu = cpu if cpu is not None else cpu_count()

        # enable string cache for categorical comparison
        pl.enable_string_cache(True)

        # *** POLARS EXPRESSIONS ***
        # cast genome columns to type to join
        GENE_COLUMNS = [
            pl.col('strand').cast(pl.Categorical),
            pl.col('chr').cast(pl.Categorical)
        ]
        # cast report columns to optimized type
        DF_COLUMNS = [
            pl.col('position').cast(pl.Int32),
            pl.col('chr').cast(pl.Categorical),
            pl.col('strand').cast(pl.Categorical),
            pl.col('context').cast(pl.Categorical),
            # density for CURRENT cytosine
            ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(pl.Float32)
        ]

        # upstream region position check
        UP_REGION = pl.col('position') < pl.col('start')
        # body region position check
        BODY_REGION = (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
        # downstream region position check
        DOWN_REGION = (pl.col('position') > pl.col('end'))

        UP_FRAGMENT = ((
            (pl.col('position') - pl.col('upstream')) / (pl.col('start') - pl.col('upstream'))
        ) * upstream_windows).floor()

        # fragment even for position == end needs to be rounded by floor
        # so 1e-10 is added (position is always < end)
        BODY_FRAGMENT = ((
            (pl.col('position') - pl.col('start')) / (pl.col('end') - pl.col('start') + 1e-10)
         ) * gene_windows).floor() + upstream_windows

        DOWN_FRAGMENT = ((
            (pl.col('position') - pl.col('end')) / (pl.col('downstream') - pl.col('end') + 1e-10)
        ) * downstream_windows).floor() + upstream_windows + gene_windows

        # batch approximation
        read_approx = approx_batch_num(file, batch_size)
        read_batches = 0

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

        # *** READING START ***
        # output dataframe
        total = None
        # initialize batched reader
        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand',
                         'count_m', 'count_um', 'context'],
            columns=[0, 1, 2, 3, 4, 5],
            batch_size=batch_size,
            n_threads=cpu
        )
        batches = bismark.next_batches(cpu)

        def process_batch(df: pl.DataFrame):
            return (
                df.lazy()
                # filter empty rows
                .filter((pl.col('count_m') + pl.col('count_um') != 0))
                # assign types
                # calculate density for each cytosine
                .with_columns(DF_COLUMNS)
                # drop redundant columns, because individual cytosine density has already been calculated
                # individual counts do not matter because every cytosine is equal
                .drop(['count_m', 'count_um'])
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
                    .cast(pl.Int32).alias('fragment'),
                    pl.concat_str(
                        pl.col("chr"),
                        (pl.concat_str(pl.col("start"), pl.col("end"), separator="-")),
                        separator=":").alias("gene").cast(pl.Categorical),
                    pl.col('id').cast(pl.Categorical)
                ])
                # gather fragment stats
                .groupby(by=['chr', 'strand', 'gene', 'context', 'id', 'fragment'])
                .agg(AGG_EXPR)
                .drop_nulls(subset=['sum'])
            ).collect()

        print(f"Reading from {file}")
        while batches:
            for df in batches:
                df = process_batch(df)
                if total is None and len(df) == 0:
                    raise Exception(
                        "Error reading Bismark file. Check format or genome. No joins on first batch.")
                elif total is None:
                    total = df
                else:
                    total = total.extend(df)

                read_batches += 1
                print(
                    f"\tRead {read_batches}/{read_approx} batch | Total size - {round(total.estimated_size('mb'), 1)}Mb RAM",
                    end="\r")
            batches = bismark.next_batches(cpu)
        print("DONE")
        return total

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
                ((pl.col("fragment") / from_fragments)
                 * to_fragments).floor().cast(pl.Int32)
            )
            .group_by(
                by=['chr', 'strand', 'gene', 'context', 'fragment']
            ).agg([
                pl.sum('sum').alias('sum'),
                pl.sum('count').alias('count')
            ])
        ).collect()

        metadata = self.metadata
        metadata["upstream_windows"] = metadata["upstream_windows"] // (
            from_fragments // to_fragments)
        metadata["downstream_windows"] = metadata["downstream_windows"] // (
            from_fragments // to_fragments)
        metadata["gene_windows"] = metadata["gene_windows"] // (
            from_fragments // to_fragments)

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


class LinePlot(BismarkBase):
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
                pl.col("sum"), pl.col("count"),
                (stat_expr).alias("density")
            ])
            .sort("fragment")
        )

        return res

    def __strand_reverse(self, df: pl.DataFrame):
        if self.strand == '-':
            max_fragment = self.plot_data["fragment"].max()
            return df.with_columns((max_fragment - pl.col("fragment")).alias("fragment"))
        else:
            return df

    @staticmethod
    def __interval(sum_density: list[int], sum_counts: list[int], alpha=.95):
        """
        Evaluate confidence interval for point

        :param sum_density: Sums of methylated counts in fragment
        :param sum_counts: Sums of all read cytosines in fragment
        :param alpha: Probability for confidence band
        """
        sum_density, sum_counts = np.array(sum_density), np.array(sum_counts)
        average = sum_density.sum() / sum_counts.sum()

        normalized = np.divide(sum_density, sum_counts)

        variance = np.average((normalized - average) ** 2, weights=sum_counts)

        n = sum(sum_counts) - 1

        i = stats.t.interval(alpha, df=n, loc=average, scale=np.sqrt(variance / n))

        return {"lower": i[0], "upper": i[1]}

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

    def __get_x_y(self, df, smooth, confidence):
        if 0 < confidence < 1:
            df = (
                df
                .with_columns(
                    pl.struct(["sum", "count"]).map_elements(
                        lambda x: self.__interval(x["sum"], x["count"], confidence)
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


    def draw(
            self,
            fig_axes: tuple = None,
            smooth: int = 50,
            label: str = "",
            confidence: int = 0,
            linewidth: float = 1.0,
            linestyle: str = '-',
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
        if fig_axes is None:
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

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

        self.__add_flank_lines(axes)

        axes.legend()

        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        return fig

    def __add_flank_lines(self, axes: plt.Axes):
        """
        Add flank lines to the given axis (for line plot)
        """
        x_ticks = []
        x_labels = []
        if self.upstream_windows > 0:
            x_ticks.append(self.upstream_windows - 1)
            x_labels.append('TSS')
        if self.downstream_windows > 0:
            x_ticks.append(self.gene_windows + self.upstream_windows)
            x_labels.append('TES')

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)
        for tick in x_ticks:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)


class HeatMap(BismarkBase):
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
                (stat_expr).alias("order")
            )
        ).collect()["order"] if order is None else order

        # sort by rows and add row numbers
        hm_data = (
            df.lazy()
            .groupby(['chr', 'strand', "gene"])
            .agg(
                pl.col('fragment'), pl.col('sum'), pl.col('count')
            )
            .with_columns(
                pl.lit(order).alias("order")
            )
            .sort('order', descending=True)
            # add row count
            .with_row_count(name='row')
            # round row count
            .with_columns(
                (pl.col('row') / (pl.col('row').max() + 1)
                 * nrow).floor().alias('row').cast(pl.Int16)
            )
            .explode(['fragment', 'sum', 'count'])
            # calc sum count for row|fragment
            .groupby(['row', 'fragment'])
            .agg(
                (stat_expr).alias('density')
            )
        )

        # prepare full template
        template = (
            pl.LazyFrame(data={"row": list(range(nrow))})
            .with_columns(
                pl.lit([list(range(0, self.total_windows))]).alias("fragment")
            )
            .explode("fragment")
            .with_columns(
                pl.col("fragment").cast(pl.Int32),
                pl.col("row").cast(pl.Int16)
            )
        )
        # join template with actual data
        hm_data = (
            # template
            template
            # join with orig
            .join(
                hm_data,
                on=['row', 'fragment'],
                how='left'
            )
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
            vmin: float = None, vmax: float = None
    ) -> Figure:
        """
        Draws heat-map on given :class:`matplotlib.Axes` or makes them itself.

        :param fig_axes: Tuple with (fig, axes) from :meth:`matplotlib.plt.subplots`.
        :param title: Title of the plot.
        :param vmin: Minimum for colormap.
        :param vmax: Maximum for colormap.
        :return:
        """
        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        vmin = 0 if vmin is None else vmin
        vmax = np.max(np.array(self.plot_data)) if vmax is None else vmax

        image = axes.imshow(
            self.plot_data, interpolation="nearest", aspect='auto', cmap=colormaps['cividis'], vmin=vmin, vmax=vmax
        )
        axes.set_title(title)
        axes.set_xlabel('Position')
        axes.set_ylabel('')
        hm_flank_lines(axes, self.upstream_windows, self.gene_windows, self.downstream_windows)
        axes.set_yticks([])
        plt.colorbar(image, ax=axes, label='Methylation density')

        return fig

    def save_plot_rds(self, path, compress: bool = False):
        """
        Save heat-map data in a matrix (ncol:nrow)
        """
        write_rds(path, pdDataFrame(self.plot_data),
                  compress="gzip" if compress else None)


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
            batch_size: int = 10 ** 6,
            cpu: int = cpu_count()
    ):
        """
        Constructor for BismarkFiles. See :meth:`Bismark.from_file`

        :param filenames: List of filenames of files
        :param genome: Same genome file for Bismark files to be aligned to.
        """
        samples = [Metagene.from_file(file, genome, upstream_windows, gene_windows,
                                     downstream_windows, batch_size, cpu) for file in filenames]
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

    def __dendrogram(self, groups, stat="mean"):
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

        constant = .1
        log2matrix = np.log2(matrix + constant)

        groups = np.array(groups)
        logFC = np.mean(log2matrix[:, groups == 1], axis=1) - np.mean(log2matrix[:, groups == 2], axis=1)

        return


class LinePlotFiles(BismarkFilesBase):
    def draw(
        self,
        smooth: int = 50,
        linewidth: float = 1.0,
        linestyle: str = '-',
        confidence: int = 0
    ):
        plt.clf()
        fig, axes = plt.subplots()
        for lp, label in zip(self.samples, self.labels):
            assert isinstance(lp, LinePlot)
            lp.draw((fig, axes), smooth, label, confidence, linewidth, linestyle)

        return fig

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


class HeatMapFiles(BismarkFilesBase):
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

    def save_plot_rds(self, base_filename, compress: bool = False):
        for sample, label in zip(self.samples, self.labels):
            sample.save_plot_rds(f"{remove_extension(base_filename)}_{label}.rds",
                                 compress="gzip" if compress else None)

