import gzip
import re
from functools import cache
from collections import Counter
from multiprocessing import cpu_count
from os.path import getsize

import polars as pl

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import stats

from pandas import DataFrame as pdDataFrame
from pyreadr import write_rds

from dynamicTreeCut import cutreeHybrid


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


def approx_batch_num(path, batch_size, check_lines=1000):
    size = getsize(path)

    length = 0
    with open(path, "rb") as file:
        for _ in range(check_lines):
            length += len(file.readline())

    return round(np.ceil(size / (length / check_lines * batch_size)))


def hm_flank_lines(axes: Axes, upstream_windows: int, gene_windows: int, downstream_windows: int):
    """
    Add flank lines to the given axis (for line plot)
    """
    x_ticks = []
    x_labels = []
    if upstream_windows > 0:
        x_ticks.append(upstream_windows - .5)
        x_labels.append('TSS')
    if downstream_windows > 0:
        x_ticks.append(gene_windows + downstream_windows - .5)
        x_labels.append('TES')

    if x_ticks and x_labels:
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)
        for tick in x_ticks:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)


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
    def from_gff(cls, file: str):
        """
        Constructor with parameters for default gff file.

        :param file: path to genome.gff.
        """
        comment_char = '#'
        has_header = False

        id_regex = "^ID=([^;]+)"

        genes = (
            pl.scan_csv(
                file,
                comment_char=comment_char,
                has_header=has_header,
                separator='\t',
                new_columns=['chr', 'source', 'type', 'start',
                             'end', 'score', 'strand', 'frame', 'attribute'],
                dtypes={'start': pl.Int32, 'end': pl.Int32, 'chr': pl.Utf8}
            )
            .with_columns(
                pl.col("attribute").str.extract(id_regex).alias("id")
            )
            .select(['chr', 'type', 'start', 'end', 'strand', "id"])
        )

        print(f"Genome read from {file}")
        return cls(genes)

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
        genes = genes.filter(pl.col('type') == gene_type).drop('type')

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


class BismarkBase:
    """
    Base class for :class:`Metagene` and plots.
    """

    def __init__(self, bismark_df: pl.DataFrame, **kwargs):
        """
        Base class for Bismark data.

        DataFrame Structure:

        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+
        | chr             | strand      | context             | gene                 | fragment         | sum            | count                                   |
        +=================+=============+=====================+======================+==================+================+=========================================+
        | Categorical     | Categorical | Categorical         | Categorical          | Int32            | Int32          | Int32                                   |
        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+
        | chromosome name | strand      | methylation context | position of cytosine | fragment in gene | sum methylated | count of all cytosines in this position |
        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+


        :param bismark_df: pl.DataFrame with cytosine methylation status.
        :param upstream_windows: Number of upstream windows. Required.
        :param gene_windows: Number of gene windows. Required.
        :param downstream_windows: Number of downstream windows. Required.
        :param strand: Strand if filtered.
        :param context: Methylation context if filtered.
        :param plot_data: Data for plotting.
        """
        self.bismark: pl.DataFrame = bismark_df

        self.upstream_windows: int = kwargs.get("upstream_windows")
        self.downstream_windows: int = kwargs.get("downstream_windows")
        self.gene_windows: int = kwargs.get("gene_windows")
        self.plot_data: pl.DataFrame = kwargs.get("plot_data")
        self.context: str = kwargs.get("context")
        self.strand: str = kwargs.get("strand")

    @property
    def metadata(self) -> dict:
        """
        :return: Bismark metadata in dict
        """
        return {
            "upstream_windows": self.upstream_windows,
            "downstream_windows": self.downstream_windows,
            "gene_windows": self.gene_windows,
            "plot_data": self.plot_data,
            "context": self.context,
            "strand": self.strand
        }

    def save_rds(self, filename, compress: bool = False):
        """
        Save Bismark DataFrame in Rds.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        write_rds(filename, self.bismark.to_pandas(),
                  compress="gzip" if compress else None)

    def save_tsv(self, filename, compress=False):
        """
        Save Bismark DataFrame in TSV.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.bismark.write_csv(file, separator="\t")
        else:
            self.bismark.write_csv(filename, separator="\t")

    @property
    def total_windows(self):
        return self.upstream_windows + self.downstream_windows + self.gene_windows

    def __len__(self):
        return len(self.bismark)


class Clustering(BismarkBase):
    """
    Class for clustering genes within sample
    """

    def __init__(self, bismark_df: pl.DataFrame, count_threshold=5, dist_method="euclidean", clust_method="average", **kwargs):
        """
        :param bismark_df: :class:polars.DataFrame with genes data
        :param count_threshold: Minimum counts per fragment
        :param dist_method: Method for evaluating distance
        :param clust_method: Method for hierarchical clustering
        """
        super().__init__(bismark_df, **kwargs)

        if self.bismark["fragment"].max() > 50:
            print(f"WARNING: too many windows ({self.bismark['fragment'].max() + 1}), clusterisation may take very long time")

        grouped = (
            self.bismark.lazy()
            .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
            .group_by(["chr", "strand", "gene", "context"])
            .agg([pl.col("density"),
                  pl.col("fragment"),
                  pl.sum("count").alias("gene_count"),
                  pl.count("fragment").alias("count")])
        ).collect()

        print(f"Starting with:\t{len(grouped)}")

        by_count = grouped.filter(pl.col("gene_count") > (count_threshold * pl.col("count")))

        print(f"Left after count theshold filtration:\t{len(by_count)}")

        by_count = by_count.filter(pl.col("count") == self.total_windows)

        print(f"Left after empty windows filtration:\t{len(by_count)}")

        if len(by_count) == 0:
            print("All genes have empty windows, exiting")
            raise ValueError("All genes have empty windows")

        by_count = by_count.explode(["density", "fragment"]).drop(["gene_count", "count"]).fill_nan(0)

        unpivot = by_count.pivot(
            index=["chr", "strand", "gene"],
            values="density",
            columns="fragment",
            aggregate_function="sum"
        ).select(
            ["chr", "strand", "gene"] + list(map(str, range(self.total_windows)))
        ).with_columns(
            pl.col("gene").alias("label")
        )

        self.gene_labels = unpivot.with_columns(pl.col("label").cast(pl.Utf8))["label"].to_numpy()
        self.matrix = unpivot[list(map(str, range(self.total_windows)))].to_numpy()

        self.gene_labels = self.gene_labels[~np.isnan(self.matrix).any(axis=1)]
        self.matrix = self.matrix[~np.isnan(self.matrix).any(axis=1), :]

        # dist matrix
        print("Distances calculation")
        self.dist = pdist(self.matrix, metric=dist_method)
        # linkage matrix
        print("Linkage calculation and minimizing distances")
        self.linkage = linkage(self.dist, method=clust_method, optimal_ordering=True)

        self.order = leaves_list(self.linkage)

    def modules(self, **kwargs):
        return Modules(self.gene_labels, self.matrix, self.linkage, self.dist,
                       windows={
                           key: self.metadata[key] for key in ["upstream_windows", "gene_windows", "downstream_windows"]
                       },
                       **kwargs)

    # TODO: rewrite save_rds, save_tsv

    def draw(
            self,
            fig_axes: tuple = None,
            title: str = None
    ) -> Figure:
        """
        Draws heat-map on given :class:`matplotlib.Axes` or makes them itself.

        :param fig_axes: Tuple with (fig, axes) from :meth:`matplotlib.plt.subplots`.
        :param title: Title of the plot.
        :return:
        """
        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        vmin = 0
        vmax = np.max(np.array(self.plot_data))

        image = axes.imshow(
            self.matrix[self.order, :],
            interpolation="nearest", aspect='auto',
            cmap=colormaps['cividis'],
            vmin=vmin, vmax=vmax
        )
        axes.set_title(title)
        axes.set_xlabel('Position')
        axes.set_ylabel('')

        hm_flank_lines(axes, self.upstream_windows, self.gene_windows, self.downstream_windows)

        axes.set_yticks([])
        plt.colorbar(image, ax=axes, label='Methylation density')

        return fig


class Modules:
    """
    Class for module construction and visualization of clustered genes
    """
    def __init__(self, labels: list, matrix: np.ndarray, linkage, distance, windows, **kwargs):
        if not len(labels) == len(matrix):
            raise ValueError("Length of labels and methylation matrix labels don't match")

        self.labels, self.matrix = labels, matrix
        self.linkage, self.distance = linkage, distance

        self.__windows = windows

        self.tree = self.__dynamic_tree_cut(**kwargs)

    def recalculate(self, **kwargs):
        """
        Recalculate tree with another params

        :param kwargs: any kwargs to cutreeHybrid from dynamicTreeCut
        """
        self.tree = self.__dynamic_tree_cut(**kwargs)

    @cache
    def __dynamic_tree_cut(self, **kwargs):
        return cutreeHybrid(self.linkage, self.distance, **kwargs)

    @property
    def __format__table(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"gene_labels": list(self.labels)} |
            {key: list(self.tree[key]) for key in ["labels", "cores", "smallLabels", "onBranch"]}
        )

    def save_rds(self, filename, compress: bool = False):
        """
        Save module data in Rds.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        write_rds(filename, self.__format__table.to_pandas(),
                  compress="gzip" if compress else None)

    def save_tsv(self, filename, compress=False):
        """
        Save module data in TSV.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.__format__table.write_csv(file, separator="\t")
        else:
            self.__format__table.write_csv(filename, separator="\t")

    def draw(
            self,
            fig_axes: tuple = None,
            title: str = None,
            show_labels=True,
            show_size=False
    ) -> Figure:
        """
        Method for visualization of moduled genes. Every row of heat-map represents an average methylation
        profile of genes of the module.

        :param fig_axes: tuple(Fig, Axes) to plot
        :param title: Title of the plot
        :param show_labels: Enable/disable module number labels
        :param show_size: Enable/disable module size labels (in brackets)
        """

        me_matrix, me_labels = [], []
        label_stats = Counter(self.tree["labels"])

        # iterate every label
        for label in label_stats.keys():
            # select genes from module
            module_genes = self.tree["labels"] == label
            # append mean module pattern
            me_matrix.append(self.matrix[module_genes, :].mean(axis=0))

            me_labels.append(f"{label} ({label_stats[label]})" if show_size else str(label))

        me_matrix, me_labels = np.stack(me_matrix), np.stack(me_labels)
        # sort matrix to minimize distances between modules
        order = leaves_list(linkage(
            y=pdist(me_matrix, metric="euclidean"),
            method="average",
            optimal_ordering=True
        ))

        me_matrix, me_labels = me_matrix[order, :], me_labels[order]

        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        vmin = 0
        vmax = np.max(np.array(me_matrix))

        image = axes.imshow(
            me_matrix,
            interpolation="nearest", aspect='auto',
            cmap=colormaps['cividis'],
            vmin=vmin, vmax=vmax
        )

        if show_labels:
            axes.set_yticks(np.arange(.5, len(me_labels), 1))
            axes.set_yticklabels(me_labels)
        else:
            axes.set_yticks([])

        axes.set_title(title)
        axes.set_xlabel('Position')
        axes.set_ylabel('Module')
        # axes.yaxis.tick_right()

        hm_flank_lines(
            axes,
            self.__windows["upstream_windows"],
            self.__windows["gene_windows"],
            self.__windows["downstream_windows"],
        )

        plt.colorbar(image, ax=axes, label='Methylation density',
                     # orientation="horizontal", location="top"
                     )

        return fig




class ChrLevels:
    def __init__(self, df: pl.DataFrame) -> None:
        self.bismark = df
        self.plot_data = (
            df
            .sort(["chr", "window"])
            .with_row_count("fragment")
            .group_by(["chr", "fragment"], maintain_order=True)
            .agg([pl.sum("sum"), pl.sum("count")])
            .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
        )

    @classmethod
    def from_file(
        cls,
        file: str,
        chr_min_length = 10**6,
        window_length: int = 10**6,
        batch_size: int = 10 ** 6,
        cpu: int = cpu_count()
    ):
        """
        Initialize ChrLevels with CX_report file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        :param cpu: How many cores to use. Uses every physical core by default
        :param batch_size: Number of rows to read by one CPU core
        """
        cpu = cpu if cpu is not None else cpu_count()

        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand',
                         'count_m', 'count_um', 'context'],
            columns=[0, 1, 2, 3, 4, 5],
            batch_size=batch_size,
            n_threads=cpu
        )
        read_approx = approx_batch_num(file, batch_size)
        read_batches = 0

        total = None

        batches = bismark.next_batches(cpu)
        print(f"Reading from {file}")
        while batches:
            for df in batches:
                df = (
                    df.lazy()
                    .filter((pl.col('count_m') + pl.col('count_um') != 0))
                    .group_by(["strand", "chr"])
                    .agg([
                        pl.col("context"),
                        (pl.col("position") / window_length).floor().alias("window").cast(pl.Int32),
                        ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(pl.Float32),
                        (pl.max("position") - pl.min("position")).alias("length")
                    ])
                    .filter(pl.col("length") > chr_min_length)
                    .explode(["context", "window", "density"])
                    .group_by(by=['chr', 'strand', 'context', 'window'])
                    .agg([
                        pl.sum('density').alias('sum'),
                        pl.count('density').alias('count')
                    ])
                    .drop_nulls(subset=['sum'])
                ).collect()
                if total is None and len(df) == 0:
                    raise Exception(
                        "Error reading Bismark file. Check format or genome. No joins on first batch.")
                elif total is None:
                    total = df
                else:
                    total = total.extend(df)

                read_batches += 1
                print(
                    f"\tRead {read_batches}/{read_approx} batch | Total size - {round(total.estimated_size('mb'), 1)}Mb RAM", end="\r")
            batches = bismark.next_batches(cpu)
        print("DONE")

        return cls(total)

    def save_plot_rds(self, path, compress: bool = False):
        """
        Saves plot data in a rds DataFrame with columns:

        +----------+---------+
        | fragment | density |
        +==========+=========+
        | Int      | Float   |
        +----------+---------+
        """
        write_rds(path, self.plot_data.to_pandas(),
                  compress="gzip" if compress else None)

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

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.bismark.filter(context_filter & strand_filter & chr_filter))

    def draw(
            self,
            fig_axes: tuple = None,
            smooth: int = 10,
            label: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
    ) -> Figure:

        if fig_axes is None:
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        ticks_data = self.plot_data.group_by("chr", maintain_order = True).agg(pl.min("fragment"))

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()

        data = self.plot_data["density"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            data = savgol_filter(data, window, 3, mode='nearest')

        x = np.arange(len(data))
        data = data * 100  # convert to percents
        axes.plot(x, data, label=label,
                  linestyle=linestyle, linewidth=linewidth)

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)

        axes.legend()
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        for tick in x_lines:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.1)

        fig.set_size_inches(12, 5)

        return fig


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
            cpu: int = cpu_count()
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
                                                batch_size, cpu)

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
            cpu: int = cpu_count()
    ) -> pl.DataFrame:
        cpu = cpu if cpu is not None else cpu_count()

        # enable string cache for categorical comparison
        pl.enable_string_cache(True)

        # POLARS EXPRESSIONS
        # cast genome columns to type to join
        gene_columns = [
            pl.col('strand').cast(pl.Categorical),
            pl.col('chr').cast(pl.Utf8).cast(pl.Categorical)
        ]
        # cast report columns to optimized type
        df_columns = [
            pl.col('position').cast(pl.Int32),
            pl.col('chr').cast(pl.Utf8).cast(pl.Categorical),
            pl.col('strand').cast(pl.Categorical),
            pl.col('context').cast(pl.Categorical),
            # density for CURRENT cytosine
            ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(pl.Float32)
        ]

        # upstream region position check
        upstream_region = pl.col('position') < pl.col('start')
        # body region position check
        body_region = (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
        # downstream region position check
        downstream_region = (pl.col('position') > pl.col('end'))

        upstream_fragment = ((
            (pl.col('position') - pl.col('upstream')) / (pl.col('start') - pl.col('upstream'))
        ) * upstream_windows).floor()

        # fragment even for position == end needs to be rounded by floor
        # so 1e-10 is added (position is always < end)
        body_fragment = ((
            (pl.col('position') - pl.col('start')) / (pl.col('end') - pl.col('start') + 1e-10)
         ) * gene_windows).floor() + upstream_windows

        downstream_fragment = ((
            (pl.col('position') - pl.col('end')) / (pl.col('downstream') - pl.col('end') + 1e-10)
        ) * downstream_windows).floor() + upstream_windows + gene_windows

        # batch approximation
        read_approx = approx_batch_num(file, batch_size)
        read_batches = 0

        # output dataframe
        total = None
        # initialize batched reader
        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand',
                         'count_m', 'count_um', 'context'],
            dtypes=[pl.Utf8, pl.Int64, pl.Utf8, pl.Int64, pl.Int64, pl.Utf8],
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
                .with_columns(df_columns)
                # drop redundant columns, because individual cytosine density has already been calculated
                # individual counts do not matter because every cytosine is equal
                .drop(['count_m', 'count_um'])
                # sort by position for joining
                .sort(['chr', 'strand', 'position'])
                # join with nearest
                .join_asof(
                    genome.lazy().with_columns(gene_columns),
                    left_on='position', right_on='upstream', by=['chr', 'strand']
                )
                # limit by end of region
                .filter(pl.col('position') <= pl.col('downstream'))
                # calculate fragment ids
                .with_columns([
                    pl.when(upstream_region).then(upstream_fragment)
                    .when(body_region).then(body_fragment)
                    .when(downstream_region).then(downstream_fragment)
                    .cast(pl.Int32).alias('fragment'),
                    pl.concat_str(
                        pl.col("chr"),
                        (pl.concat_str(pl.col("start"), pl.col("end"), separator="-")),
                        separator=":").alias("gene").cast(pl.Categorical),
                    pl.col('id').cast(pl.Categorical)
                ])
                # gather fragment stats
                .groupby(by=['chr', 'strand', 'gene', 'context', 'id', 'fragment'])
                .agg([
                    pl.sum('density').alias('sum'),
                    pl.count('density').alias('count')
                ])
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

    def line_plot(self, resolution: int = None):
        """
        :param resolution: Number of fragments to resize to. Keep None if not needed.
        :return: :class:`LinePlot`.
        """
        bismark = self.resize(resolution)
        return LinePlot(bismark.bismark, **bismark.metadata)

    def heat_map(self, nrow: int = 100, ncol: int = 100):
        """
        :param nrow: Number of fragments to resize to. Keep None if not needed.
        :param ncol: Number of columns in the resulting heat-map.
        :return: :class:`HeatMap`.
        """
        bismark = self.resize(ncol)
        return HeatMap(bismark.bismark, nrow, order=None, **bismark.metadata)


class LinePlot(BismarkBase):
    def __init__(self, bismark_df: pl.DataFrame, **kwargs):
        """
        Calculates plot data for line-plot.
        """
        super().__init__(bismark_df, **kwargs)

        self.plot_data = self.bismark.group_by(["context", "fragment"]).agg([
            pl.col("sum"), pl.col("count"),
            (pl.sum("sum") / pl.sum("count")).alias("density")
        ]).sort("fragment")

        if self.strand == '-':
            max_fragment = self.plot_data["fragment"].max()
            self.plot_data = self.plot_data.with_columns(
                (max_fragment - pl.col("fragment")).alias("fragment"))

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

    def draw(
            self,
            fig_axes: tuple = None,
            smooth: int = 50,
            label: str = "",
            confidence = 0,
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

        for context in self.plot_data["context"].unique().to_list():
            df = self.plot_data.filter(pl.col("context") == context)

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

            x = np.arange(len(data))
            data = data * 100  # convert to percents

            axes.plot(x, data,
                      label=f"{context}" if not label else f"{label}_{context}",
                      linestyle=linestyle, linewidth=linewidth)

            if 0 < confidence < 1:
                upper = df["upper"].to_numpy() * 100  # convert to percents
                lower = df["lower"].to_numpy() * 100  # convert to percents

                upper = savgol_filter(upper, window, 3, mode="nearest") if smooth else upper
                lower = savgol_filter(lower, window, 3, mode="nearest") if smooth else lower

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
    def __init__(self, bismark_df: pl.DataFrame, nrow, order=None, **kwargs):
        super().__init__(bismark_df, **kwargs)

        order = (
            self.bismark.lazy()
            .groupby(['chr', 'strand', "gene"])
            .agg(
                (pl.col('sum').sum() / pl.col('count').sum()).alias("order")
            )
        ).collect()["order"] if order is None else order

        # sort by rows and add row numbers
        hm_data = (
            self.bismark.lazy()
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
                (pl.sum('sum') / pl.sum('count')).alias('density')
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
        self.plot_data = np.array(
            hm_data.groupby('row', maintain_order=True).agg(
                pl.col('density'))['density'].to_list(),
            dtype=np.float32
        )

        if self.strand == '-':
            self.plot_data = np.fliplr(self.plot_data)

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


class BismarkFilesBase:
    def __init__(self, samples, labels: list[str] = None):
        self.samples = self.__check_metadata(
            samples if isinstance(samples, list) else [samples])
        if samples is None:
            raise Exception("Flank or gene windows number does not match!")
        self.labels = [str(v) for v in list(
            range(len(samples)))] if labels is None else labels
        if len(self.labels) != len(self.samples):
            raise Exception("Labels length doesn't match samples number")

    def save_rds(self, base_filename, compress: bool = False, merge: bool = False):
        if merge:
            merged = pl.concat(
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
            )
            write_rds(base_filename, merged.to_pandas(),
                      compress="gzip" if compress else None)
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_rds(
                    f"{remove_extension(base_filename)}_{label}.rds", compress="gzip" if compress else None)

    def save_tsv(self, base_filename, compress: bool = False, merge: bool = False):
        if merge:
            merged = pl.concat(
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
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
                    f"{remove_extension(base_filename)}_{label}.rds", compress=compress)

    @staticmethod
    def __check_metadata(samples: list[BismarkBase]):
        upstream_check = set([sample.metadata["upstream_windows"]
                             for sample in samples])
        downstream_check = set(
            [sample.metadata["downstream_windows"] for sample in samples])
        gene_check = set([sample.metadata["gene_windows"]
                         for sample in samples])

        if len(upstream_check) == len(gene_check) == len(downstream_check) == 1:
            return samples
        else:
            return None


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

    def line_plot(self, resolution: int = None):
        """
        :class:`LinePlot` for all files.
        """
        return LinePlotFiles([sample.line_plot(resolution) for sample in self.samples], self.labels)

    def heat_map(self, nrow: int = 100, ncol: int = None):
        """
        :class:`HeatMap` for all files.
        """
        return HeatMapFiles([sample.heat_map(nrow, ncol) for sample in self.samples], self.labels)

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


class LinePlotFiles(BismarkFilesBase):
    def draw(
        self,
        smooth: int = 50,
        linewidth: float = 1.0,
        linestyle: str = '-',
        confidence=0
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
