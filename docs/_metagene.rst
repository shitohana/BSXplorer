Metagene
========

Metagene calculation and plotting related methods.

.. currentmodule:: bsxplorer

.. class:: Metagene

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
        :toctree: _Metagene/_method
        :nosignatures:
        :template: method.rst

            ~Metagene.from_bismark

            ~Metagene.from_cgmap

            ~Metagene.from_bedGraph

            ~Metagene.from_coverage

            ~Metagene.from_binom

            ~Metagene.filter

            ~Metagene.resize

            ~Metagene.trim_flank

            ~Metagene.line_plot

            ~Metagene.contexts_line_plot

            ~Metagene.heat_map

            ~Metagene.context_box_plot

            ~Metagene.cluster


.. class:: MetageneFiles

    .. rubric:: Methods

    .. automethod:: __init__

    .. autosummary::
        :toctree: _MetageneFiles/_method
        :nosignatures:
        :template: method.rst

            ~MetageneFiles.from_list

            ~MetageneFiles.filter

            ~MetageneFiles.resize

            ~MetageneFiles.trim_flank

            ~MetageneFiles.merge

            ~MetageneFiles.line_plot

            ~MetageneFiles.heat_map

            ~MetageneFiles.box_plot

            ~MetageneFiles.cluster

            ~MetageneFiles.dendrogram

