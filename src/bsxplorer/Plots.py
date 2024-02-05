from __future__ import annotations

import itertools
import re

import numpy as np
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


class LinePlot(PlotBase):
    """
    Line-plot single metagene
    """

    def __init__(self, bismark_df: pl.DataFrame, stat="wmean", merge_strands: bool = True, **kwargs):
        """
        Calculates plot data for line-plot.
        """
        super().__init__(bismark_df, **kwargs)

        self.stat = stat

        if merge_strands:
            bismark_df = self._merge_strands(bismark_df)
        plot_data = self.__calculate_plot_data(bismark_df, stat, self.total_windows)

        if not merge_strands:
            if self.strand == '-':
                plot_data = self._strand_reverse(plot_data)
        self.plot_data = plot_data

    @staticmethod
    def __calculate_plot_data(df: pl.DataFrame, stat, total_windows):
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

        contexts = res["context"].unique().to_list()

        template = pl.DataFrame({
            "fragment": list(range(total_windows)) * len(contexts),
            "context": list(itertools.chain(*[[c] * (total_windows) for c in contexts])),
        }, schema={"fragment": res.schema["fragment"], "context": res.schema["context"]})

        joined = template.join(res, on=["fragment", "context"], how="left")

        return joined

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

        data = df["density"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth and np.isnan(data).sum() == 0:
            data = savgol_filter(data, window, 3, mode='nearest')

        lower, upper = None, None
        data = data * 100  # convert to percents

        if (0 < confidence < 1) and np.isnan(data).sum() == 0:
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

        Parameters
        ----------
        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        df = self.bismark.group_by("fragment").agg(
            (pl.sum("sum") / pl.sum("count")).alias("density")
        )
        write_rds(path, df.to_pandas(),
                  compress="gzip" if compress else None)

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            smooth: int = 50,
            label: str = "",
            confidence: float = 0,
            linewidth: float = 1.0,
            linestyle: str = '-',
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ) -> Figure:
        """
        Draws line-plot on given axes.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_). New are created if ``None``
        smooth
            Number of windows for `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter (set 0 for no smoothing)
        label
            Label of line on line-plot
        confidence
            Probability for confidence bands (e.g. 95%)
        linewidth
            Width of the line
        linestyle
            Style of the line
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not

        Returns
        -------
            `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        See Also
        --------
        :doc:`LinePlot example<../../markdowns/test>`

        `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_ : To create fig, axes

        `Linestyles <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ : For possible linestyles.
        """
        fig, axes = plt.subplots() if fig_axes is None else fig_axes
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        contexts = self.plot_data["context"].unique().to_list()

        for context in contexts:
            df = self.plot_data.filter(pl.col("context") == context)

            lower, data, upper = self.__get_x_y(df, smooth, confidence)

            x = np.arange(len(data))

            axes.plot(x, data,
                      label=f"{context}" if not label else f"{label}_{context}",
                      linestyle=linestyle, linewidth=linewidth)

            if (0 < confidence < 1) and np.isnan(data).sum() == 0:
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
            confidence: float = .0,
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True
    ):
        """
        Draws line-plot on given figure.

        Parameters
        ----------
        figure
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_. New is created if ``None``
        smooth
            Number of windows for `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter (set 0 for no smoothing)
        label
            Label of line on line-plot
        confidence
            Probability for confidence bands (e.g. 95%)
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not

        Returns
        -------
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_

        """

        figure = go.Figure() if figure is None else figure
        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

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

            figure.add_traces(data=traces)

        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %"
        )

        figure = self.flank_lines_plotly(figure, major_labels, minor_labels, show_border)

        return figure


class LinePlotFiles(MetageneFilesBase):
    """Line-plot multiple metagenes"""

    def draw_mpl(
        self,
        smooth: int = 50,
        linewidth: float = 1.0,
        linestyle: str = '-',
        confidence: int = 0,
        major_labels: list[str] = None,
        minor_labels: list[str] = None,
        show_border: bool = True
    ):
        """
        Draws line-plot for all Metagenes on given axes.

        Parameters
        ----------
        smooth
            Number of windows for `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter (set 0 for no smoothing)
        confidence
            Probability for confidence bands (e.g. 95%)
        linewidth
            Width of the line
        linestyle
            Style of the line
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not

        Returns
        -------
            `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        See Also
        --------
        `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_ : To create fig, axes

        `Linestyles <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ : For possible linestyles.
        """

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

        plt.clf()
        fig, axes = plt.subplots()
        for lp, label in zip(self.samples, self.labels):
            assert isinstance(lp, LinePlot)
            lp.draw_mpl((fig, axes), smooth, label, confidence, linewidth, linestyle, major_labels, minor_labels, show_border)

        return fig

    def draw_plotly(self,
                    smooth: int = 50,
                    confidence: int = 0,
                    major_labels: list[str] = None,
                    minor_labels: list[str] = None,
                    show_border: bool = True
                    ):
        """
        Draws line-plot for all Metagenes on given figure.

        Parameters
        ----------
        smooth
            Number of windows for `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter (set 0 for no smoothing)
        confidence
            Probability for confidence bands (e.g. 95%)
        major_labels
            Labels for body region start and end (e.g. TSS, TES). **Exactly 2** need to be provided. Set ``[]`` to disable.
        minor_labels
            Labels for upstream, body and downstream regions. **Exactly 3** need to be provided. Set ``[]`` to disable.
        show_border
            Whether to draw dotted vertical line on body region borders or not

        Returns
        -------
            `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_

        """

        major_labels = ["TSS", "TES"] if major_labels is None else major_labels
        minor_labels = ["Upstream", "Body", "Downstream"] if minor_labels is None else minor_labels

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
    """Heat-map single metagene"""

    def __init__(self, bismark_df: pl.DataFrame, nrow, order=None, stat="wmean", merge_strands: bool = True, **kwargs):
        super().__init__(bismark_df, **kwargs)

        if merge_strands:
            bismark_df = self._merge_strands(bismark_df)

        plot_data = self.__calculcate_plot_data(bismark_df, nrow, order, stat)

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

    def draw_mpl(
            self,
            title: str = None,
            color_scale: str = "Viridis",
            major_labels: list[str] = None,
            minor_labels: list[str] = None,
            show_border: bool = True,
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
        for sample, label in zip(self.samples, self.labels):
            sample.save_plot_rds(f"{remove_extension(base_filename)}_{label}.rds",
                                 compress="gzip" if compress else None)


class PCA:
    def __init__(self):
        self.region_density = []

        self.mapping = {}

    def append_metagene(self, metagene, label, group):
        self.region_density.append(
            metagene.bismark
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
        def __init__(self, matrix: np.ndarray, labels: list[str], groups: list[str]):
            self.matrix = matrix
            self.labels = labels
            self.groups = groups

            pca = PCA_sklearn(n_components=2)
            fit: PCA_sklearn = pca.fit(matrix)

            self.eigenvectors = fit.components_
            self.explained_variance = fit.explained_variance_ratio_

    def draw_plotly(self):
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
