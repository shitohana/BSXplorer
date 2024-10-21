from __future__ import annotations

import re

import numpy as np
import polars as pl
from plotly import graph_objects as go
from scipy.signal import savgol_filter


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


def flank_lines_mpl(axes, x_ticks, x_labels: list, borders: list = None):
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


def prepare_labels(major_labels: list, minor_labels: list):
    labels = dict(
        up_mid="Upstream",
        body_start="TSS",
        body_mid="Body",
        body_end="c",
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

    return labels
