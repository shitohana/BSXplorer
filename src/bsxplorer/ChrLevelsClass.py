from __future__ import annotations

import os
import re

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyreadr import write_rds
from scipy.signal import savgol_filter
import pyarrow as pa
import plotly.graph_objects as go
import plotly.express as px
from plotly.express.colors import qualitative as PALETTE
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

from .UniversalReader_classes import UniversalReader
from .utils import interval, decompress, ReportBar, dotdict
from .Base import interval_chr, read_chromosomes, validate_chromosome_args
from .ArrowReaders import CsvReader, BismarkOptions, ParquetReader, CGmapOptions


class ChrLevels:
    def __init__(self, df: pl.DataFrame) -> None:
        """
        Read report and visualize chromosome methylation levels

        Parameters
        ----------
        df
        """
        self.bismark = df

        # delete this in future and change to calculation of plot data
        # when plot is drawn
        self.plot_data = self.__calculate_plot_data(df)

    @staticmethod
    def __calculate_plot_data(df: pl.DataFrame):
        group_cols = [pl.sum("sum"), pl.sum("count"), pl.min("start").alias("chr_pos")]
        mut_cols = [(pl.col("sum") / pl.col("count")).alias("density")]

        if "upper" in df.columns:
            group_cols += [pl.mean("upper"), pl.mean("lower")]

        return (
            df
            .sort(["chr", "window"])
            .group_by(["chr", "window", "context"], maintain_order=True)
            .agg(group_cols)
            .group_by(["chr", "window"], maintain_order=True)
            .agg(pl.all().exclude(["chr", "window"]))
            .with_row_count("fragment")
            .explode(pl.all().exclude(["chr", "window", "fragment"]))
            .with_columns(mut_cols)
        )


    @classmethod
    def from_bismark(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = True,
            confidence: float = None
    ):
        """
        Initialize ChrLevels with CX_report file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        """

        reader = UniversalReader(**(locals() | dict(report_type="bismark")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_cgmap(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = True,
            confidence: float = None
    ):
        """
        Initialize ChrLevels with CGmap file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        """
        reader = UniversalReader(**(locals() | dict(report_type="cgmap")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_parquet(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            confidence: float = None
    ):
        """
        Initialize ChrLevels with parquet file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        """
        raise NotImplementedError()

    @classmethod
    def from_binom(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            confidence: float = None,
            p_value: float = .05
    ):
        reader = UniversalReader(**(locals() | dict(report_type="binom", methylation_pvalue=p_value)))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

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
        Filter chromosome methylation levels data.

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH) to filter (only one).
        strand
            Strand to filter (+ or -).
        chr
            Chromosome name to filter.

        Returns
        -------
            :class:`ChrLevels`
        """
        context_filter = self.bismark["context"] == context if context is not None else True
        strand_filter = self.bismark["strand"] == strand if strand is not None else True
        chr_filter = self.bismark["chr"] == chr if chr is not None else True

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.bismark.filter(context_filter & strand_filter & chr_filter))

    @property
    def __ticks_data(self):
        ticks_data = self.plot_data.group_by("chr", maintain_order=True).agg(pl.min("fragment"))

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()
        return x_ticks, x_labels, x_lines

    def __add_flank_lines(self, axes):
        x_ticks, x_labels, x_lines = self.__ticks_data

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)

        for tick in x_lines:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.1)

    def __add_flank_lines_plotly(self, figure: go.Figure):
        x_ticks, x_labels, x_lines = self.__ticks_data

        figure.update_layout(
            xaxis = dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=x_labels)
        )

        for tick in x_lines:
            figure.add_vline(x=tick, line_dash="dash", line_color="rgba(0,0,0,0.2)")


    def __get_points(self, df, smooth):
        data = df["density"].to_numpy()

        upper, lower = None, None
        if "upper" in self.plot_data.columns and np.isnan(data).sum() == 0:
            not_nan_data = df.filter(pl.col("upper").is_not_nan())
            upper = not_nan_data["upper"].to_numpy()
            lower = not_nan_data["lower"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            _, _, lines = self.__ticks_data
            data_ranges = [data[lines[i]: lines[i + 1]] for i in range(len(lines) - 1)]
            data_ranges = [savgol_filter(r, window, 3, mode='nearest') for r in data_ranges]

            data = np.concatenate(data_ranges)

            if upper is not None:
                upper = np.concatenate([savgol_filter(r, window, 3, mode="nearest") for r in
                                        [upper[lines[i]: lines[i + 1]] for i in range(len(lines) - 1)]])
                lower = np.concatenate([savgol_filter(r, window, 3, mode="nearest") for r in
                                        [lower[lines[i]: lines[i + 1]] for i in range(len(lines) - 1)]])

        x = np.arange(len(data))

        # Convert to percents
        data = data * 100
        if upper is not None:
            upper, lower = upper * 100, lower * 100

        return x, data, upper, lower


    def draw_mpl(
            self,
            fig_axes: tuple = None,
            smooth: int = 10,
            label: str = None,
            draw_ci: bool = True,
            **kwargs
    ) -> Figure:
        """
        Draws line-plot on given axis.

        Parameters
        ----------
        fig_axes
            Tuple of (`matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_, `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_). New are created if ``None``
        smooth
            Number of windows for `SavGol <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ filter (set 0 for no smoothing)
        label
            Label of line on line-plot
        draw_ci
            Draw confidence intervals
        kwargs
            kwargs to pass to `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_

        Returns
        -------
            ``matplotlib.pyplot.Figure``

        See Also
        --------
        `matplotlib.pyplot.Figure <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure>`_

        `matplotlib.pyplot.subplot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot>`_ : To create fig, axes
        """
        if fig_axes is None:
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        for context in self.plot_data["context"].unique():
            filtered = self.plot_data.filter(context=context)
            x, y, upper, lower = self.__get_points(filtered, smooth)

            current_label = label + f"_{context}" if label is not None else context

            axes.plot(x, y, label=current_label, **kwargs)
            if upper is not None and draw_ci:
                axes.fill_between(x, lower, upper, alpha=.2)

        axes.legend()
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        self.__add_flank_lines(axes)

        fig.set_size_inches(12, 5)

        return fig

    def draw_plotly(
            self,
            figure: Figure = None,
            smooth: int = 10,
            label: str = None,
            draw_ci: bool = True,
            **kwargs
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
        draw_ci
            Draw confidence intervals
        kwargs
            kwargs to pass to `plotly.graph_objects.Scatter <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html>`_

        Returns
        -------
        ``plotly.graph_objects.Figure``

        See Also
        --------
        `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure>`_
        """

        figure = go.Figure() if figure is None else figure

        for context in self.plot_data["context"].unique():
            filtered = self.plot_data.filter(context=context)
            x, y, upper, lower = self.__get_points(filtered, smooth)

            label_current = label + f"_{context}" if label is not None else context

            traces = [go.Scatter(x=x, y=y, name=label_current, mode="lines", **kwargs)]

            if upper is not None and draw_ci:
                ci_kwargs = dict(
                    mode="lines",
                    line_color='rgba(0,0,0,0)',
                    name=f"{label_current}_CI"
                )

                traces += [
                    go.Scatter(x=x, y=upper, showlegend=False, **ci_kwargs),
                    go.Scatter(x=x, y=lower, fill="tonexty", showlegend=True, **ci_kwargs)
                ]

            figure.add_traces(data=traces)

        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %"
        )

        self.__add_flank_lines_plotly(figure)

        return figure

    def bar_plot(self):
        bismark = self.bismark
        class ChrBarPlot:
            def draw_mpl(self):
                pd_df = (
                    bismark
                    .group_by(["chr", "context"])
                    .agg((pl.sum("sum") / pl.sum("count")).alias("density"))
                    .sort("chr", "context")
                    .pivot(values="density", index="chr", columns="context")
                    .to_pandas()
                    .set_index("chr")
                )

                fig, axes = plt.subplots()
                pd_df.plot.barh(ax=axes, figsize=(12, 9))
                fig.subplots_adjust(left=0.2)
                return fig

            def draw_plotly(self):
                pd_df = (
                    bismark
                    .group_by(["chr", "context"])
                    .agg((pl.sum("sum") / pl.sum("count")).alias("density"))
                    .sort("chr", "context")
                    .to_pandas()
                )

                figure = px.histogram(
                    pd_df,
                    x="chr", y="density",
                    color="context",
                    barmode="group",
                    histfunc="avg"
                )

                figure.update_layout(bargap=.05)

                return figure

        return ChrBarPlot()
