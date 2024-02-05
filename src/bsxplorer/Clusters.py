from __future__ import annotations

import gzip
import warnings
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from dynamicTreeCut import cutreeHybrid
from dynamicTreeCut.dynamicTreeCut import get_heights
from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering
import seaborn as sns
import plotly.express as px

from .Base import MetageneBase, MetageneFilesBase
from abc import ABC, abstractmethod


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
        return df.filter(pl.col("strand") == "+").extend(self.__strand_reverse(df.filter(pl.col("strand") == "-")))

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
        df = self.__merge_strands(metagene.bismark)

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
            .select(["chr", "strand", "name"] + list(map(str, range(metagene.total_windows))))
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
    def __init__(self, metagene: MetageneBase, count_threshold=5, na_rm: float | None = None):
        self.matrix, self.names = self._process_metagene(metagene, count_threshold, na_rm)

    def kmeans(self, n_clusters: int = 8, n_init: int = 10, **kwargs):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, **kwargs).fit(self.matrix)

        print(f"Clustering done in {kmeans.n_iter_} iterations")

        return ClusterPlot(ClusterData.from_kmeans(kmeans, self.names))

    def cut_tree(self, dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs):
        dist = pdist(self.matrix, metric=dist_method)
        link_matrix = linkage(dist, method=clust_method)

        cutHeight = np.quantile(get_heights(link_matrix), q=cut_height_q)
        tree = cutreeHybrid(link_matrix, dist, cutHeight=cutHeight, **kwargs)

        labels = tree["labels"]
        return ClusterPlot(ClusterData.from_matrix(self.matrix, labels, self.names))

    def all(self):
        return ClusterPlot(ClusterData(self.matrix, np.arange(len(self.matrix), dtype=np.int64), self.names))


class ClusterMany(_ClusterBase):
    def __init__(self, metagenes: MetageneFilesBase, count_threshold=5, na_rm: float | None = None):
        intersect_list = set.intersection(*[set(metagene.bismark["gene"].to_list()) for metagene in metagenes.samples])
        for i in range(len(metagenes.samples)):
            metagenes.samples[i].bismark = metagenes.samples[i].bismark.filter(pl.col("gene").is_in(intersect_list))

        self.clusters = [ClusterSingle(metagene, count_threshold, na_rm) for metagene in metagenes.samples]
        self.sample_names = metagenes.labels

    def kmeans(self, n_clusters: int = 8, n_init: int = 10, **kwargs):
        return ClusterPlot([cluster.kmeans(n_clusters, n_init, **kwargs).data for cluster in self.clusters], self.sample_names)

    def cut_tree(self, dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs):
        return ClusterPlot([
            cluster.cut_tree(dist_method="euclidean", clust_method="average", cut_height_q=.99, **kwargs).data
            for cluster in self.clusters
        ], self.sample_names)

    def all(self):
        return ClusterPlot([cluster.all().data for cluster in self.clusters], self.sample_names)


class ClusterData:
    def __init__(self, centers: np.ndarray, labels: np.array, names: list[str] | np.array):
        self.centers = centers
        self.labels = labels
        self.names = names

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
    def __init__(self, data: ClusterData | list[ClusterData], sample_names=None):
        if isinstance(data, list) and len(data) == 1:
            self.data = data[0]
        else:
            self.data = data

        self.sample_names = sample_names

    def save_tsv(self, filename: str):
        filename = Path(filename)

        def save(data: ClusterData, path: Path):
            df = pl.DataFrame(dict(name=list(map(str, data.names)), label=data.labels), schema=dict(name=pl.Utf8, label=pl.Utf8))
            df.write_csv(path, has_header=False, separator="\t")

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

    def draw_mpl(self, method='average', metric='euclidean', cmap: str = "cividis"):
        if isinstance(self.data, list):
            warnings.warn("Matplotlib version of cluster plot is not available for multiple samples")
            return None
        else:
            df = pd.DataFrame(
                self.data.centers,
                index=[f"{name} ({count})" for name, count in zip(*np.unique(self.data.labels, return_counts=True))])

            fig = sns.clustermap(df, col_cluster=False, cmap=cmap, method=method, metric=metric)
            return fig

    def draw_plotly(self, method='average', metric='euclidean', cmap: str = "cividis"):
        if isinstance(self.data, list):
            # order for first sample
            dist = pdist(self.data[0].centers, metric=metric)
            link = linkage(dist, method, metric)
            link = optimal_leaf_ordering(link, dist, metric=metric)

            order = leaves_list(link)

            im = np.dstack([d.centers[order, :] for d in self.data])

            figure = px.imshow(im, color_continuous_scale=cmap, animation_frame=2, aspect='auto')
            figure.update_layout(sliders=[{"currentvalue": {"prefix": "Sample = "}}])
            if self.sample_names is not None:
                for step, sample_name in zip(figure.layout.sliders[0].steps, self.sample_names):
                    step.label = sample_name
                    step.name = sample_name
            return figure
        else:
            dist = pdist(self.centers, metric=metric)
            link = linkage(dist, method, metric)
            link = optimal_leaf_ordering(link, dist, metric=metric)

            order = leaves_list(link)
            im = self.data.centers[order, :]

            figure = px.imshow(im, color_continuous_scale=cmap, aspect='auto')
            return figure
