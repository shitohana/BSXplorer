from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Iterable

import matplotlib.cbook
import numpy as np
import polars as pl
from matplotlib import pyplot as plt, colormaps, colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly import graph_objects as go, express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA as PCA_sklearn

from .utils import prepare_labels


def savgol_line(data: np.ndarray | None, window, polyorder=3, mode="nearest"):
    if window and data is not None:
        if np.isnan(data).sum() == 0:
            window = window if window > polyorder else polyorder + 1
            return savgol_filter(data, window, polyorder, mode=mode)
    return data


def plot_stat_expr(stat):
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
    return stat_expr


def lp_ticks(ticks: dict, major_labels: list, minor_labels: list):
    labels = dict(
        up_mid="Upstream",
        body_start="TSS",
        body_mid="Body",
        body_end="TES",
        down_mid="Downstream"
    )

    if major_labels and len(major_labels) == 2:
        labels["body_start"], labels["body_end"] = major_labels
    elif major_labels:
        print("Length of major tick labels != 2. Using default.")
    else:
        labels["body_start"], labels["body_end"] = [""] * 2

    if minor_labels and len(minor_labels) == 3:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = minor_labels
    elif minor_labels:
        print("Length of minor tick labels != 3. Using default.")
    else:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = [""] * 3

    labels = prepare_labels(major_labels, minor_labels)

    if ticks["body_start"] < 1:
        labels["down_mid"], labels["body_end"] = [""] * 2

    if (ticks["down_mid"] - ticks["body_end"]) < 1:
        labels["up_mid"], labels["body_start"] = [""] * 2

    x_ticks = [ticks[key] for key in ticks.keys()]
    x_labels = [labels[key] for key in ticks.keys()]

    return x_ticks, x_labels


def flank_lines_mpl(axes: Axes, x_ticks, x_labels: list, borders: list = None):
    borders = list() if borders is None else borders
    if x_labels is None or not x_labels:
        x_labels = [""] * 5
    axes.set_xticks(x_ticks, labels=x_labels)

    for border in borders:
        axes.axvline(x=border, linestyle='--', color='k', alpha=.3)

    return axes


def flank_lines_plotly(figure: go.Figure, x_ticks, x_labels, borders: list = None, fig_rows=None, fig_cols=None):
    borders = list() if borders is None else borders
    if x_labels is None or not x_labels:
        x_labels = [""] * 5

    fig_rows = fig_rows if isinstance(fig_rows, list) else [fig_rows]
    fig_cols = fig_cols if isinstance(fig_cols, list) else [fig_cols]

    for row in fig_rows:
        for col in fig_cols:
            figure.for_each_xaxis(
                lambda xaxis: xaxis.update(dict(
                    tickmode='array',
                    tickvals=x_ticks,
                    ticktext=x_labels,
                    showticklabels=True,
                )),
                row=row, col=col
            )

            for border in borders:
                figure.add_vline(x=border, line_dash="dash", line_color="rgba(0,0,0,0.2)", row=row, col=col)

    return figure


@dataclass
class LinePlotData:
    x: np.ndarray
    y: np.ndarray
    x_ticks: Iterable
    borders: Iterable
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None,
    label: str = ""
    x_labels: list[str] | None = None


class LinePlot:
    """
    Class for generating line plot.

    Parameters
    ----------
    data
        Instance of :class:`LinePlotData` (e.g. generated from :func:`Metagene.line_plot_data`)

    """
    def __init__(self, data: list[LinePlotData] | LinePlotData):
        self.data = data if isinstance(data, list) else [data]

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            label: str = "",
            tick_labels: list[str] = None,
            show_border: bool = True,
            **kwargs
    ) -> Figure:
        """
        Draws line-plot on given matplotlib axes.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_). New are created if ``None``
        label
            Label of line on line-plot
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not
        kwargs
            Keyword arguments for matplotlib.plot

        Returns
        -------
        ``matplotlib.pyplot.Figure``

        See Also
        --------
        :doc:`LinePlot example<../../markdowns/lineplot>`

        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_ : To create fig, axes

        `Linestyles <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ : For possible linestyles.
        """
        fig, axes = plt.subplots() if fig_axes is None else fig_axes

        for line_data in self.data:
            axes.plot(
                line_data.x,
                line_data.y,
                label=line_data.label if line_data.label else label,
                **kwargs
            )

            if line_data.lower is not None:
                axes.fill_between(line_data.x, line_data.lower, line_data.upper, alpha=.2)

        flank_lines_mpl(
            axes=axes,
            x_ticks=self.data[0].x_ticks,
            x_labels=tick_labels if tick_labels is not None else self.data[0].x_labels,
            borders=self.data[0].borders if show_border else []
        )

        axes.legend()
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        return fig

    def draw_plotly(
            self,
            figure: go.Figure = None,
            label: str = "",
            tick_labels: list[str] = None,
            show_border: bool = True,
            fig_rows: int | list = None,
            fig_cols: int | list = None,
            **kwargs
    ):
        """
        Draws line-plot on given figure.

        Parameters
        ----------
        figure
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_. New is created if ``None``
        label
            Label of line on line-plot
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not
        kwargs
            Keyword arguments for plotly.graph_objects.Scatter
        fig_cols
            Cols of the subplot where to draw the line plot.
        fig_rows
            Rows of the subplot where to draw the line plot.

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        :doc:`LinePlot example<../../markdowns/lineplot>`

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        figure = make_subplots() if figure is None else figure

        for line_data in self.data:
            traces = []
            name = line_data.label if line_data.label else label
            args = dict(x=line_data.x, y=line_data.y, mode="lines")

            traces.append(go.Scatter(**args, name=name, **kwargs))

            if line_data.lower is not None:
                ci_args = dict(line_color='rgba(0,0,0,0)', name=name + "_CI", mode="lines")
                traces.append(go.Scatter(x=line_data.x, y=line_data.upper, **ci_args, showlegend=False))
                traces.append(go.Scatter(x=line_data.x, y=line_data.lower, **ci_args, showlegend=True, fill="tonexty", fillcolor='rgba(0, 0, 0, 0.2)'))

            figure.add_traces(traces, rows=fig_rows, cols=fig_cols)

        figure.for_each_xaxis(lambda axis: axis.update(title="Position"))
        figure.for_each_yaxis(lambda axis: axis.update(title="Methylation density, %"))

        figure = flank_lines_plotly(
            figure=figure,
            x_ticks=self.data[0].x_ticks,
            x_labels=tick_labels if tick_labels is not None else self.data[0].x_labels,
            borders=self.data[0].borders if show_border else [],
            fig_rows=fig_rows,
            fig_cols=fig_cols
        )

        return figure


@dataclass
class HeatMapData:
    matrix: np.ndarray
    x_ticks: list
    borders: list
    label: str = ""
    x_labels: list[str] | None = None


class HeatMap:
    """
    Class for generating heat map.

    Parameters
    ----------
    data
        Instance of :class:`HeatMapData` (e.g. generated from :func:`Metagene.heat_map_data`)

    """
    def __init__(self, data: list[HeatMapData] | HeatMapData):
        self.data = data if isinstance(data, list) else [data]

    def draw_mpl(
            self,
            label: str = "",
            tick_labels: list[str] = None,
            show_border: bool = True,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            facet_cols: int = 4,
            **kwargs
    ):
        """
        Draws heat-map plot on given matplotlib axes.

        Parameters
        ----------
        label
            Title for axis
        vmin
            Set minimum value for colorbar explicitly.
        vmax
            Set maximum value for colorbar explicitly.
        color_scale
            Name of color scale.
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not.
        facet_cols
            Maximum number of plots on the same row.

        Returns
        -------
        ``matplotlib.pyplot.Figure``

        See Also
        --------
        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        `Matplotlib color scales <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_: For possible colormap ``color_scale`` arguments.
        """
        plt.clf()
        subplots_y = len(self.data) // facet_cols + 1
        subplots_x = facet_cols if len(self.data) > facet_cols else len(self.data)

        fig, axes_matrix = plt.subplots(subplots_y, subplots_x)
        if not isinstance(axes_matrix, np.ndarray):
            axes_matrix = np.array(axes_matrix)
        axes_matrix = axes_matrix.reshape((subplots_x, subplots_y))

        for count, hm_data in enumerate(self.data):
            vmin = 0 if vmin is None else vmin
            vmax = np.max(hm_data.matrix) if vmax is None else vmax

            axes = axes_matrix[count % facet_cols, count // facet_cols]
            assert isinstance(axes, Axes)
            image = axes.imshow(
                hm_data.matrix,
                interpolation="nearest", aspect='auto',
                cmap=colormaps[color_scale.lower()],
                vmin=vmin, vmax=vmax
            )

            axes.set_title(label)
            axes.set_xlabel('Position')
            axes.set_ylabel('')

            plt.colorbar(image, ax=axes, label='Methylation density, %')

            flank_lines_mpl(axes, self.data[0].x_ticks, self.data[0].x_labels if tick_labels is None else tick_labels, self.data[0].borders if show_border else [])
            axes.set_yticks([])

        return fig

    def draw_plotly(
            self,
            figure: Figure = None,
            title: str = None,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            tick_labels: list[str] = None,
            show_border: bool = True,
            row: int | list = None,
            col: int | list = None,
            facet_cols: int = 4
    ):
        """
        Draws heat-map plot on given plotly figure.

        Parameters
        ----------
        figure
            Plotly Figure, where to plot HeatMap
        title
            Title for axis
        vmin
            Set minimum value for colorbar explicitly.
        vmax
            Set maximum value for colorbar explicitly.
        color_scale
            Name of color scale.
        tick_labels
            Labels for upstream, body region start and end, downstream (e.g. TSS, TES).
            **Exactly 5** need to be provided. Set ``None`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not.
        row
            Row, where heatmap will be plotted.
        col
            Column, where heatmap will be plotted.
        facet_cols
            Maximum number of plots on the same row.

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `Plotly color scales <https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales>`_: For possible colormap ``color_scale`` arguments.

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        labels = dict(
            x="Position",
            y="Rank",
            color="Methylation density, %"
        )

        if len(self.data) > 1 and (row is not None or col is not None):
            warnings.warn("Selecting row or col is not compatitable with multiple HeatMap instances.")
            row, col = None, None

        subplots_y = len(self.data) // facet_cols + 1 if len(self.data) != facet_cols else 1
        subplots_x = facet_cols if len(self.data) > facet_cols else len(self.data)
        figure = make_subplots(rows=subplots_y, cols=subplots_x, shared_yaxes=True) if figure is None else figure

        for count, hm_data in enumerate(self.data):
            fig_row = count // facet_cols + 1 if row is None else row
            fig_col = count % facet_cols + 1 if col is None else col

            new_fig = px.imshow(
                hm_data.matrix,
                labels=labels,
                title=title,
                zmin=vmin, zmax=vmax,
                aspect="auto",
                color_continuous_scale=color_scale,
            )

            figure.add_traces(new_fig.data, rows=fig_row, cols=fig_col)
            figure.for_each_yaxis(lambda axis: axis.update(showticklabels=False, title="Rank"), row=fig_row, col=fig_col)
            figure.for_each_xaxis(lambda axis: axis.update(title="Position"), row=fig_row, col=fig_col)
            flank_lines_plotly(figure, hm_data.x_ticks, hm_data.x_labels if tick_labels is None else tick_labels, hm_data.borders if show_border else [],
                               fig_row, fig_col)
            figure.add_annotation(
                text=hm_data.label if title is None else title + hm_data.label,
                row=fig_row, col=fig_col,
                xref="x domain", yref="y domain",
                x=0.5,  y=1.1,
                showarrow=False, font=dict(size=16)
            )

            figure.layout.coloraxis.update(colorbar=new_fig.layout["coloraxis"]["colorbar"].update(len=.8, lenmode="fraction", yref="paper"))

        return figure


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


@dataclass
class BoxPlotData:
    values: list
    label: str
    locus: list = None
    id: list = None

    @classmethod
    def empty(cls, label=None):
        return cls([], label if label is not None else "", [], [])


class BoxPlot:
    """
    Class for generating box plot.

    Parameters
    ----------
    data
        Instance of :class:`BoxPlotData` (e.g. generated from :func:`Metagene.box_plot_data`)

    """
    def __init__(self, data: list[BoxPlotData] | BoxPlotData):
        self.data = data if isinstance(data, list) else [data]
        self.values = [bp_data.values for bp_data in data]
        self.labels = [bp_data.label for bp_data in data]
        self.n_boxes = len(self.labels)

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            showfliers=False,
            title: str = None,
            violin: bool = False
    ):
        """
        Draw box plot with matplotlib.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_). New are created if ``None``
        showfliers
            Show outliers on the boxplot.
        title
            Title of the box plot.
        violin
            Should box plot be visualized as violin plot.

        Returns
        -------
        ``matplotlib.pyplot.Figure``
        """
        if fig_axes is None:
            plt.clf()
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        if violin:
            axes.violinplot(self.values, showmeans=False, showmedians=True)
        else:
            axes.boxplot(self.values, showfliers=showfliers)
        axes.set_xticks(np.arange(1, self.n_boxes + 1), labels=self.labels)
        axes.set_title(title)
        axes.set_ylabel('Methylation density')

        return fig

    def draw_plotly(
            self,
            figure: Figure = None,
            title="",
            violin: bool = False,
            points: bool | str = False,
            fig_rows: int | list = None,
            fig_cols: int | list = None
    ):
        """
        Draw box plot with plotly.

        Parameters
        ----------
        figure
            `plotly.graph_objects.Figure
            <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_.
            New is created if ``None``
        points
            Specify which points should be included
            (`see Plotly docs <https://plotly.github.io/plotly.py-docs/generated/plotly.express.box.html>`_)
        title
            Title of the box plot.
        violin
            Should box plot be visualized as violin plot.
        fig_cols
            Cols of the subplot where to draw the box plot.
        fig_rows
            Rows of the subplot where to draw the box plot.

        Returns
        -------
        ``matplotlib.pyplot.Figure``
        """
        figure = make_subplots() if figure is None else figure

        for data, label in zip(self.values, self.labels):
            args = dict(y=data, name=label)
            trace = go.Violin(points=points, **args) if violin else go.Box(**args, boxpoints=points)
            figure.add_traces([trace], rows=fig_rows, cols=fig_cols)

        for row in (fig_rows if isinstance(fig_rows, list) else [fig_rows]):
            for col in (fig_cols if isinstance(fig_cols, list) else [fig_cols]):
                figure.for_each_annotation(lambda layout: layout.update(dict(
                    title=title,
                    yaxis_title="Methylation density"
                )), row=row, col=col)

        return figure

    def mpl_box_data(self):
        return {hm_data.label: matplotlib.cbook.boxplot_stats(hm_data.values) for hm_data in self.data}


