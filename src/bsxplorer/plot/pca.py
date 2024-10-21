from __future__ import annotations

import sys

import numpy as np
import polars as pl
from plotly import express as px


class PCA:
    """PCA for samples initialized with same annotation."""
    def __init__(self):
        self.region_density = []

        self.mapping = {}

    def append_metagene(self, metagene, label, group):
        """
        Add metagene to PCA object.

        Parameters
        ----------
        metagene
            Metagene to add.
        label
            Label for this appended metagene.
        group
            Sample group this metagene belongs to

        Examples
        --------

        >>> pca = bsxplorer.PCA()
        >>>
        >>> metagene = bsxplorer.Metagene.from_bismark(...)
        >>> pca.append_metagene(metagene, 'control-1', 'control')
        """
        self.region_density.append(
            metagene.report_df
            .group_by("gene")
            .agg((pl.sum("sum") / pl.sum("count")).alias("density"))
            .with_columns([
                pl.lit(label).alias("label")
            ])
        )

        self.mapping[label] = group

    def _get_pivoted(self):
        if self.region_density:
            concated = pl.concat(self.region_density)

            return concated.pivot(values="density",
                                  index="gene",
                                  columns="label",
                                  aggregate_function="mean",
                                  separator=";").drop_nulls()
        else:
            raise ValueError()

    def _get_pca_data(self):
        pivoted = self._get_pivoted()
        excluded = pivoted.select(pl.all().exclude("gene"))

        labels = excluded.columns
        groups = list(map(lambda key: self.mapping[key], labels))
        matrix = excluded.to_numpy()

        return self.PCA_data(matrix, labels, groups)

    class PCA_data:
        """
        PCA data is calculated with `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
        """
        def __init__(self, matrix: np.ndarray, labels: list[str], groups: list[str]):
            self.matrix = matrix
            self.labels = labels
            self.groups = groups
            if 'sklearn.decomposition' not in sys.modules:
                from sklearn.decomposition import PCA as PCA_sklearn
            pca = PCA_sklearn(n_components=2)
            fit: PCA_sklearn = pca.fit(matrix)

            self.eigenvectors = fit.components_
            self.explained_variance = fit.explained_variance_ratio_

    def draw_plotly(self):
        """
        Draw PCA plot.
        PCA data is calculated with `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """
        data = self._get_pca_data()

        x = data.eigenvectors[0, :]
        y = data.eigenvectors[1, :]

        df = pl.DataFrame({"x": x, "y": y, "group": data.groups, "label": data.labels}).to_pandas()
        figure = px.scatter(df, x="x", y="y", color="group", text="label")

        figure.update_layout(
            xaxis_title="PC1: %.2f" % (data.explained_variance[0]*100) + "%",
            yaxis_title="PC2: %.2f" % (data.explained_variance[1]*100) + "%"
        )

        return figure

    def draw_mpl(self):
        """
        Draw PCA plot.
        PCA data is calculated with `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_

        Returns
        -------
        ``matplotlib.pyplot.Figure``

        See Also
        --------
        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_
        """
        data = self._get_pca_data()
        if 'matplotlib' not in sys.modules:
            import pyplot as plt, colors as mcolors
        fig, axes = plt.subplots()

        x = data.eigenvectors[:, 0]
        y = data.eigenvectors[:, 1]

        color_mapping = {label: list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in zip(range(len(set(data.groups))), set(data.groups))}

        for group in set(data.groups):
            axes.scatter(x[data.groups == group], y[data.groups == group], c=color_mapping[group], label=group)

        axes.set_xlabel("PC1: %.2f" % data.explained_variance[0] + "%")
        axes.set_ylabel("PC2: %.2f" % data.explained_variance[1] + "%")

        for x, y, label in zip(x, y, data.labels):
            axes.text(x, y, label)

        axes.legend()

        return fig
