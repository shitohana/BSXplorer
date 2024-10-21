from __future__ import annotations

import sys

import matplotlib.cbook
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .data import BoxPlotData


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
        if 'matplotlib' not in sys.modules:
            import pyplot as plt
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
            figure = None,
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
