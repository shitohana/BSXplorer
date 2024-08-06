from __future__ import annotations

import multiprocessing
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from dynamicTreeCut import cutreeHybrid
from dynamicTreeCut.dynamicTreeCut import get_heights
from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from .Base import MetageneBase, MetageneFilesBase
from .Plots import flank_lines_plotly

default_n_threads = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
from sklearn.cluster import KMeans, MiniBatchKMeans


# noinspection PyMissingOrEmptyDocstring
class _ClusterBase(ABC):
    @abstractmethod
    def kmeans(self, n_clusters: int = 8, n_init: int = 10, **kwargs):
        ...

    @abstractmethod
    def cut_tree(self, dist_method="euclidean", clust_method="average", cutHeight_q=.99, **kwargs):
        ...

    @abstractmethod
    def all(self):
        ...

    def __merge_strands(self, df: pl.DataFrame):
        return df.filter(pl.col("strand") == "+").vstack(self.__strand_reverse(df.filter(pl.col("strand") == "-")))

    @staticmethod
    def __strand_reverse(df: pl.DataFrame):
        max_fragment = df["fragment"].max()
        return df.with_columns((max_fragment - pl.col("fragment")).alias("fragment"))

    def _process_metagene(
            self,
            metagene: MetageneBase,
            count_threshold=5,
            na_rm: float | None = None
    ) -> (np.ndarray, np.ndarray):
        # Merge strands
        df = self.__merge_strands(metagene.report_df)

        grouped = (
            df.lazy()
            .filter(pl.col("count") > count_threshold)
            .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
            .group_by(["chr", "strand", "gene", "context"])
            .agg([pl.first("id"),
                  pl.first("start"),
                  pl.col("density"),
                  pl.col("fragment"),
                  pl.sum("count").alias("gene_count"),
                  pl.count("fragment").alias("count")])
        ).collect()

        # by_count = grouped.filter(pl.col("gene_count") > (count_threshold * pl.col("count")))
        # print(f"Left after count theshold filtration:\t{len(by_count)}")

        by_count = grouped
        if na_rm is None:
            by_count = grouped.filter(pl.col("count") == metagene.total_windows)
            print(f"Left after empty windows filtration:\t{len(by_count)}")

        if len(by_count) == 0:
            raise ValueError("All genes have empty windows")

        by_count = by_count.explode(["density", "fragment"]).drop(["gene_count", "count"]).fill_nan(0)

        unpivot: pl.DataFrame = (
            by_count
            .sort(["chr", "start"])
            .with_columns(pl.when(pl.col("id").is_null()).then(pl.col("gene")).otherwise(pl.col("id")).alias("name"))
            .pivot(
                index=["chr", "strand", "name"],
                values="density",
                columns="fragment",
                aggregate_function="sum",
                maintain_order=True
            )
            .select(["chr", "strand", "name"] + list(map(str, range(int(metagene.total_windows)))))
            .cast({"name": pl.Utf8})
        )

        if na_rm is None:
            unpivot = unpivot.drop_nulls()
        else:
            unpivot = unpivot.fill_null(na_rm)

        # add id if present
        names = unpivot["name"].to_numpy()
        matrix = unpivot.select(pl.all().exclude(["strand", "chr", "name"])).to_numpy()

        return matrix, names


class ClusterSingle(_ClusterBase):
    """Class for operating with single sample regions clustering"""

    def __init__(self, metagene: MetageneBase, count_threshold=5, na_rm: float | None = None, empty=False):
        if not empty:
            self.matrix, self.names = self._process_metagene(metagene, count_threshold, na_rm)
            self._x_ticks = metagene._x_ticks
            self._borders = metagene._borders

    @classmethod
    def _from_raw(cls, matrix, names, x_ticks, _borders):
        c = cls(None, empty=True)
        c.matrix = matrix
        c.names = names
        c._x_ticks = x_ticks
        c._borders = _borders
        return c

    def kmeans(self, n_clusters: int = 8, n_init: int = 10, **kwargs):
        """
        KMeans clustering on sample regions. Clustering is being made with `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

        Parameters
        ----------
        n_clusters
            The number of clusters to generate.
        n_init
            Number of times the k-means algorithm is run with different centroid seeds.
        kwargs
            See `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

        Returns
        -------
        :class:`ClusterPlot`

        """

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, **kwargs).fit(self.matrix)

        print(f"Clustering done in {kmeans.n_iter_} iterations")

        return ClusterPlot(ClusterData.from_kmeans(kmeans, self.names))

    def cut_tree(self, dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs):
        """
        KMeans clustering on sample regions. Clustering is being made with `dynamicTreeCut.cutreeHybrid <https://github.com/kylessmith/dynamicTreeCut>`_.

        Parameters
        ----------
        dist_method
            Distances calculation metric
        clust_method
            Hierarchical clustering method
        cut_height_q
            Quantile of leaves height to be cut.
        kwargs
            See `dynamicTreeCut <https://github.com/kylessmith/dynamicTreeCut>`_.

        Returns
        -------
        :class:`ClusterPlot`
        """

        dist = pdist(self.matrix, metric=dist_method)
        link_matrix = linkage(dist, method=clust_method)

        cutHeight = np.quantile(get_heights(link_matrix), q=cut_height_q)
        tree = cutreeHybrid(link_matrix, dist, cutHeight=cutHeight, **kwargs)

        labels = tree["labels"]
        return ClusterPlot(ClusterData.from_matrix(self.matrix, labels, self.names))

    def all(self):
        """
        Returns all regions for downstream plotting.

        Returns
        -------
        :class:`ClusterPlot`
        """
        return ClusterPlot(ClusterData(self.matrix, np.arange(len(self.matrix), dtype=np.int64), self.names))


class ClusterMany(_ClusterBase):
    """Class for operating with multiple samples regions clustering"""

    def __init__(self, metagenes: MetageneFilesBase, count_threshold=5, na_rm: float | None = None):
        intersect_list = set.intersection(*[set(metagene.report_df["gene"].to_list()) for metagene in metagenes.samples])
        for i in range(len(metagenes.samples)):
            metagenes.samples[i].report_df = metagenes.samples[i].report_df.filter(pl.col("gene").is_in(intersect_list))

        self.clusters = [ClusterSingle(metagene, count_threshold, na_rm) for metagene in metagenes.samples]
        self.sample_names = metagenes.labels

    def compare(self):
        if len(self.clusters) > 2:
            raise ValueError("This method is available only for 2 samples")

        # Match region set
        a_sample = self.clusters[0]
        b_sample = self.clusters[1]

        intersection = list(set.intersection(*map(lambda cluster: set(cluster.names), self.clusters)))
        intersection.sort()

        a_order = np.argsort(a_sample.names)
        b_order = np.argsort(b_sample.names)

        a_matrix = a_sample.matrix[a_order[np.searchsorted(a_sample.names, intersection, sorter=a_order)], :]
        b_matrix = b_sample.matrix[b_order[np.searchsorted(b_sample.names, intersection, sorter=b_order)], :]

        diff_matrix = b_matrix - a_matrix

        names = a_sample.names[a_order[np.searchsorted(a_sample.names, intersection, sorter=a_order)]]
        cluster_single = ClusterSingle._from_raw(diff_matrix, names, a_sample._x_ticks, a_sample._borders)

        return cluster_single


    def kmeans(self, n_clusters: int = 8, n_init: int = 10, **kwargs):
        """
        KMeans clustering on sample regions. Clustering is being made with `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

        Parameters
        ----------
        n_clusters
            The number of clusters to generate.
        n_init
            Number of times the k-means algorithm is run with different centroid seeds.
        kwargs
            See `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

        Returns
        -------
        :class:`ClusterPlot`

        """
        return ClusterPlot([cluster.kmeans(n_clusters, n_init, **kwargs).data for cluster in self.clusters], self.sample_names)

    def cut_tree(self, dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs):
        """
        KMeans clustering on sample regions. Clustering is being made with `dynamicTreeCut.cutreeHybrid <https://github.com/kylessmith/dynamicTreeCut>`_.

        Parameters
        ----------
        dist_method
            Distances calculation metric
        clust_method
            Hierarchical clustering method
        cut_height_q
            Quantile of leaves height to be cut.
        kwargs
            See `dynamicTreeCut <https://github.com/kylessmith/dynamicTreeCut>`_.

        Returns
        -------
        :class:`ClusterPlot`
        """

        return ClusterPlot([
            cluster.cut_tree(dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs).data
            for cluster in self.clusters
        ], self.sample_names)

    def all(self):
        """
        Returns all regions for downstream plotting.

        Returns
        -------
        :class:`ClusterPlot`
        """

        return ClusterPlot([cluster.all().data for cluster in self.clusters], self.sample_names)


# noinspection PyMissingOrEmptyDocstring
class ClusterData:
    def __init__(self, centers: np.ndarray, labels: np.array, names: list[str] | np.array,
                 ticks: list[int] = None, borders: list[int] = None, matrix: np.ndarray = None):
        self.centers = centers
        self.labels = labels
        self.names = names

        self.ticks = ticks
        self.borders = borders
        self.matrix = matrix

    @classmethod
    def from_kmeans(cls, kmeans: KMeans, names: list[str] | np.array):
        return cls(kmeans.cluster_centers_, kmeans.labels_, names)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, labels: np.array, names: list[str] | np.array,
                    method=Literal["mean", "median", "log1p"]):
        if method == "mean":
            agg_fun = lambda matrix: np.mean(matrix, axis=0)
        elif method == "median":
            agg_fun = lambda matrix: np.median(matrix, axis=0)
        elif method == "log1p":
            agg_fun = lambda matrix: np.log1p(matrix, axis=0)
        else:
            agg_fun = lambda matrix: np.mean(matrix, axis=0)

        modules = np.array([agg_fun(matrix[labels == label, :]) for label in labels])

        return cls(modules, labels, names)


class ClusterPlot:
    """Class for plotting cluster data."""
    def __init__(self, data: ClusterData | list[ClusterData], sample_names=None):
        if isinstance(data, list) and len(data) == 1:
            self.data = data[0]
        else:
            self.data = data

        self.sample_names = sample_names

    def save_tsv(self, filename: str):
        """
        Save labels for regions in a TSV file.

        Parameters
        ----------
        filename
            File name for output file
        """

        filename = Path(filename)

        def save(data: ClusterData, path: Path):
            df = pl.DataFrame(dict(name=list(map(str, data.names)), label=data.labels), schema=dict(name=pl.Utf8, label=pl.Utf8))
            df.write_csv(path, include_header=False, separator="\t")

        if self.sample_names is not None and isinstance(self.data, list):
            for data, sample_name in zip(self.data, self.sample_names):

                new_name = filename.name + sample_name
                save(data, filename.with_name(new_name).with_suffix(".tsv"))

        if not isinstance(self.data, list):

            save(self.data, filename.with_suffix(".tsv"))

    def __intersect_genes(self):
        if isinstance(self.data, list):
            names = [d.names for d in self.data]
            intersection = set.intersection(*map(set, names))

            if len(intersection) < 1:
                raise ValueError("No same regions between samples")
            elif len(intersection) < max(map(len, names)):
                print(
                    f"Found {len(intersection)} intersections between samples with {max(map(len, names))} regions max")

    def draw_mpl(self, method='average', metric='euclidean', cmap: str = "cividis", **kwargs):
        """
        Draws clustermap with seaborn.clustermap.

        Parameters
        ----------
        method
            Method for hierarchical clustering.
        metric
            Metric for distance calculation
        cmap
            Colormap to use
        **kwargs
            ``seaborn.clustermap`` parameters

        See Also
        --------
        `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_ : For more information about possible parameters
        """

        if isinstance(self.data, list):
            warnings.warn("Matplotlib version of cluster plot is not available for multiple samples")
            return None
        else:
            df = pd.DataFrame(
                self.data.centers,
                index=[f"{name} ({count})" for name, count in zip(*np.unique(self.data.labels, return_counts=True))])

            args = dict(col_cluster=False) | kwargs
            args |= dict(cmap=cmap, method=method, metric=metric)

            fig = sns.clustermap(df, **args)
            return fig

    def draw_plotly(self, method='average', metric='euclidean', cmap: str = "cividis", tick_labels: list[str] = None, **kwargs):
        """
        Draws clustermap with plotly imshow.

        Parameters
        ----------
        method
            Method for hierarchical clustering.
        metric
            Metric for distance calculation
        cmap
            Colormap to use

        Returns
        --------
        ``plotly.graph_objects.Figure``
        """

        if isinstance(self.data, list):
            # order for first sample
            dist = pdist(self.data[0].centers, metric=metric)
            link = linkage(dist, method, metric)
            link = optimal_leaf_ordering(link, dist, metric=metric)

            order = leaves_list(link)

            im = np.dstack([d.centers[order, :] for d in self.data])

            figure = px.imshow(im, color_continuous_scale=cmap, animation_frame=2, aspect='auto', **kwargs)
            figure.update_layout(sliders=[{"currentvalue": {"prefix": "Sample = "}}])
            if self.sample_names is not None:
                for step, sample_name in zip(figure.layout.sliders[0].steps, self.sample_names):
                    step.label = sample_name
                    step.name = sample_name
            return figure
        else:
            dist = pdist(self.data.centers, metric=metric)
            link = linkage(dist, method, metric)
            link = optimal_leaf_ordering(link, dist, metric=metric)

            order = leaves_list(link)
            im = self.data.centers[order, :]

            ticktext = np.array([f"{label} ({count})" for label, count in zip(*np.unique(self.data.labels, return_counts=True))])

            figure = px.imshow(im, color_continuous_scale=cmap, aspect='auto', **kwargs)
            figure.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(order))),
                    ticktext=ticktext[order]
                )
            )

            if tick_labels is None:
                tick_labels = ["Upstream", "", "Body", "", "Downstream"]

            figure = flank_lines_plotly(figure, self.data.ticks, tick_labels, self.data.borders)
            return figure

    @property
    def labels(self):
        return self.data.labels

    @property
    def names(self):
        return self.data.names

    def module_corr(self, module: int = None, p_cutoff: float = None):
        if self.data.matrix is None:
            return None
        if module > self.data.centers.shape[0] - 1:
            raise ValueError(f"Max cluster index is {self.data.centers.shape[0] - 1}!")
        else:
            module_index = self.data.labels == module
            module_members = self.data.matrix[module_index, :]
            module_vector = self.data.centers[module, :]

            cor = np.apply_along_axis(lambda row: pearsonr(row, module_vector), axis=1, arr=module_members).astype(np.float64)

            res_df = pl.DataFrame(
                data=np.c_[self.data.names[self.data.labels == module], cor].T.tolist(),
                schema=dict(name=pl.String, cor=pl.Float64, pvalue=pl.Float64)
            )

            if p_cutoff is not None:
                res_df = res_df.filter(pl.col("pvalue") <= p_cutoff)

            return res_df


    def get_module_ids(self, module: int):
        if module > self.data.centers.shape[0] - 1:
            raise ValueError(f"Max cluster index is {self.data.centers.shape[0] - 1}!")

        return self.names[self.labels == module]
