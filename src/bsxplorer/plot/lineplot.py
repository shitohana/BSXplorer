from __future__ import annotations

from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .data import LinePlotData
from .utils import flank_lines_mpl, flank_lines_plotly


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
    ):
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
