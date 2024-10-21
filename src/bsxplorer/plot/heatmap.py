from __future__ import annotations

import sys
import warnings

import numpy as np
from plotly import express as px
from plotly.subplots import make_subplots

from .data import HeatMapData
from .utils import flank_lines_mpl, flank_lines_plotly


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
        if 'matplotlib' not in sys.modules:
            import pyplot as plt, colormaps, colors as mcolors
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
            figure = None,
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
