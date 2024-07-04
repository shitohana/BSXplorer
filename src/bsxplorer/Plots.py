from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import matplotlib.cbook
import numpy as np
import packaging.version
import polars as pl
from matplotlib import pyplot as plt, colormaps, colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as pdDataFrame
from plotly import graph_objects as go, express as px
from pyreadr import write_rds
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA as PCA_sklearn

from .Base import PlotBase, MetageneFilesBase
from .utils import MetageneSchema, interval, remove_extension, prepare_labels


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

def flank_lines_plotly(figure: go.Figure, x_ticks, x_labels, borders: list = None):
    borders = list() if borders is None else borders
    if x_labels is None or not x_labels:
        x_labels = [""] * 5
    figure.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_labels)
    )

    for border in borders:
        figure.add_vline(x=border, line_dash="dash", line_color="rgba(0,0,0,0.2)")

    return figure


@dataclass
class LinePlotData:
    x: np.ndarray
    y: np.ndarray
    x_ticks: list
    borders: list
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None,
    label: str = ""
    x_labels: list[str] | None = None


class LinePlot:
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

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        :doc:`LinePlot example<../../markdowns/lineplot>`

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        figure = go.Figure() if figure is None else figure

        for line_data in self.data:
            traces = []
            name = line_data.label if line_data.label else label
            args = dict(x=line_data.x, y=line_data.y, mode="lines")

            traces.append(go.Scatter(**args, name=name, **kwargs))

            if line_data.lower is not None:
                ci_args = args | dict(line_color='rgba(0,0,0,0)', name=name + "_CI")
                traces.append(go.Scatter(**ci_args, showlegend=True))
                traces.append(go.Scatter(**ci_args, showlegend=False))

            figure.add_traces(traces)


        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %"
        )

        figure = flank_lines_plotly(
            figure=figure,
            x_ticks=self.data[0].x_ticks,
            x_labels=tick_labels if tick_labels is not None else self.data[0].x_labels,
            borders=self.data[0].borders if show_border else []
        )

        return figure


@dataclass
class HeatMapData:
    matrix: np.ndarray
    x_ticks: list
    borders: list
    label: str = ""


class HeatMapNew:
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

            flank_lines_mpl(axes, self.data[0].x_ticks, tick_labels, self.data[0].borders if show_border else [])
            axes.set_yticks([])

        return fig

    def draw_plotly(
            self,
            title: str = None,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            tick_labels: list[str] = None,
            show_border: bool = True,
            facet_cols: int = 4
    ):
        """
        Draws heat-map plot on given plotly figure.

        Parameters
        ----------
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

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `Plotly color scales <https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales>`_: For possible colormap ``color_scale`` arguments.

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        samples_matrix = np.stack([hm_data.matrix for hm_data in self.data])

        labels = dict(
            x="Position",
            y="Rank",
            color="Methylation density, %"
        )

        figure = px.imshow(
            samples_matrix,
            labels=labels,
            title=title,
            zmin=vmin, zmax=vmax,
            aspect="auto",
            color_continuous_scale=color_scale,
            facet_col=0,
            facet_col_wrap=facet_cols if len(self.data) > facet_cols else len(self.data)
        )

        # set facet annotations
        hm_labels = [hm_data.label for hm_data in self.data]
        figure.for_each_annotation(lambda l: l.update(text=hm_labels[int(l.text.split("=")[1])]))

        # disable y ticks
        figure.update_layout(
            yaxis=dict(
                showticklabels=False
            )
        )

        figure = flank_lines_plotly(figure, self.data[0].x_ticks, tick_labels, self.data[0].borders if show_border else [])

        return figure


class HeatMap(PlotBase):
    """Heat-map single metagene"""

    def __init__(self, report_df: pl.DataFrame, nrow, order=None, stat="wmean", merge_strands: bool = True, **kwargs):
        super().__init__(report_df, **kwargs)

        if merge_strands:
            report_df = self._merge_strands(report_df)

        plot_data = self.__calculcate_plot_data(report_df, nrow, order, stat)

        if not merge_strands:
            # switch to base strand reverse
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
            .group_by(['chr', 'strand', "gene"], maintain_order=True)
            .agg(
                stat_expr.alias("order")
            )
        ).collect()["order"] if order is None else order

        # sort by rows and add row numbers
        hm_data = (
            df.lazy()
            .group_by(['chr', 'strand', "gene"], maintain_order=True)
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

        template = pl.LazyFrame(data={"row": list(range(nrow))})

        # this is needed because polars changed .lit list behaviour in > 0.20:
        if packaging.version.parse(pl.__version__) < packaging.version.parse('0.20.0'):
            template = template.with_columns(pl.lit([list(range(0, self.total_windows))]).alias("fragment"))
        else:
            template = template.with_columns(pl.lit(list(range(0, self.total_windows))).alias("fragment"))

        # prepare full template
        template = (
            template
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

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            title: str = None,
            vmin: float = None, vmax: float = None,
            color_scale="Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ):
        """
        Draws heat-map plot on given matplotlib axes.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_). New are created if ``None``
        title
            Title for axis
        vmin
            Set minimum value for colorbar explicitly.
        vmax
            Set maximum value for colorbar explicitly.
        color_scale
            Name of color scale.
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not.

        Returns
        -------
        ``matplotlib.pyplot.Figure``

        See Also
        --------
        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        `Matplotlib color scales <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_: For possible colormap ``color_scale`` arguments.
        """

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

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
        """
        Draws heat-map plot on given plotly figure.

        Parameters
        ----------
        title
            Title for axis
        vmin
            Set minimum value for colorbar explicitly.
        vmax
            Set maximum value for colorbar explicitly.
        color_scale
            Name of color scale.
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not.

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `Plotly color scales <https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales>`_: For possible colormap ``color_scale`` arguments.

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

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

        Parameters
        ----------
        path
            Path to saved file
        compress
            Whether data needs to be compressed.
        """
        write_rds(path, pdDataFrame(self.plot_data),
                  compress="gzip" if compress else None)


class HeatMapFiles(MetageneFilesBase):
    """Heat-map multiple metagenes"""

    def __add_flank_lines_plotly(self, figure: go.Figure, major_labels: list, minor_labels: list, show_border=True):
        """
        Add flank lines to the given axis (for line plot)
        """
        labels = prepare_labels(major_labels, minor_labels)

        if self.samples[0].downstream_windows < 1:
            labels["down_mid"], labels["body_end"] = [""] * 2

        if self.samples[0].upstream_windows < 1:
            labels["up_mid"], labels["body_start"] = [""] * 2

        ticks = self.samples[0]._tick_positions

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

    def draw_mpl(
            self,
            title: str = None,
            color_scale: str = "Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True,
    ):
        """
        Draws heat-map plot for all samples.

        Parameters
        ----------
        title
            Title for axis
        color_scale
            Name of color scale.
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not.


        Returns
        -------
        ``matplotlib.pyplot.Figure``

        See Also
        --------
        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        `Matplotlib color scales <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_: For possible colormap ``color_scale`` arguments.
        """

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
                hm.draw_mpl((fig, ax), self.labels[number], vmin, vmax, color_scale, major_labels, minor_labels, show_border)

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
        """
        Draws heat-map plot for all samples.

        Parameters
        ----------
        title
            Title for axis
        color_scale
            Name of color scale.
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        facet_cols
            How many columns will be in output multiple heat-map grid.

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `Plotly color scales <https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales>`_: For possible colormap ``color_scale`` arguments.

        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels
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
        """
        Save heat-map data in a matrix (ncol:nrow)

        Parameters
        ----------
        base_filename
            Base name for output files. Final will be ``[base_filename]_[label].rds``
        compress
            Whether data needs to be compressed.

        Returns
        -------

        """
        for sample, label in zip(self.samples, self.labels):
            sample.save_plot_rds(f"{remove_extension(base_filename)}_{label}.rds",
                                 compress="gzip" if compress else None)


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


class BoxPlot:
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

    def draw_plotly(self, title="", violin: bool = False):
        figure = go.Figure()

        for data, label in zip(self.values, self.labels):
            args = dict(y=data, name=label)
            trace = go.Violin(**args) if violin else go.Box(**args)
            figure.add_trace(trace)

        figure.update_layout(
            title=title,
            yaxis_title="Methylation density"
        )

        return figure

    def mpl_box_data(self):
        return {hm_data.label: matplotlib.cbook.boxplot_stats(hm_data.values) for hm_data in self.data}


