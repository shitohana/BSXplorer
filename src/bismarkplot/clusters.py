import gzip
from collections import Counter
from functools import cache

import numpy as np
import polars as pl
from dynamicTreeCut import cutreeHybrid
from matplotlib import pyplot as plt, colormaps
from matplotlib.figure import Figure
from pyreadr import write_rds
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from .base import BismarkBase
from .utils import prepare_labels, hm_flank_lines


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

    def __add_flank_lines(self, axes, major_labels: list, minor_labels: list, show_border=True):
        labels = prepare_labels(major_labels, minor_labels)

        if self.downstream_windows < 1:
            labels["down_mid"], labels["body_end"] = [""] * 2

        if self.upstream_windows < 1:
            labels["up_mid"], labels["body_start"] = [""] * 2

        x_ticks = self.tick_positions
        x_labels = [labels[key] for key in x_ticks.keys()]

        axes.set_xticks(x_ticks, labels=x_labels)

        if show_border:
            for tick in [x_ticks["body_start"], x_ticks["body_end"]]:
                axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)

        return axes

    def draw(
            self,
            fig_axes: tuple = None,
            title: str = None,
            color_scale="Viridis",
            major_labels=["TSS", "TES"],
            minor_labels=["Upstream", "Body", "Downstream"],
            show_border=True
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
            cmap=colormaps[color_scale.lower()],
            vmin=vmin, vmax=vmax
        )
        axes.set_title(title)
        axes.set_xlabel('Position')
        axes.set_ylabel('')

        self.__add_flank_lines(axes, major_labels, minor_labels, show_border)

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
