Genome
======

^^^^^^^^^
Reference
^^^^^^^^^

.. currentmodule:: bsxplorer

.. autosummary::
    :nosignatures:
    :toctree: _Genome
    :template: class.rst

    Genome
    RegAlignResult

.. autosummary::
    :nosignatures:
    :toctree: _Genome
    :template: func.rst

    align_regions

^^^^^
Usage
^^^^^

BSXplorer offers functionality to align one set of regions over another. Regions can
be read either with :class:`Genome` or initialized directly with
`polars functionality <https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html>`_
(DataFrame need to have `chr`, `start` and `end` columns).

To align regions (e.g. define DMR position relative to genes) use :func:`align_regions`.

.. code-block:: python

    import bsxplorer as bsx

    genes = bsx.Genome.from_gff("path/to/annot.gff").gene_body(min_length=0)
    dmr = bsx.Genome.from_custom(
        "path/to/dmr.txt",
        chr_col=0,
        start_col=1,
        end_col=2
    ).all()

    res = align_regions(dmr, along_regions=genes, flank_length=2000)

:func:`align_regions` returns :class:`RegAlignResult`, which stores regions which are
in upstream, body, downstream or intergenic space. Then metagene coverage with regions
can be plotted via :func:`RegAlignResult.plot_density_mpl` method.

.. code-block:: python

    fig = res.plot_density_mpl(
        flank_windows=100,
        body_windows=200,
        major_labels=["TSS", "TES"],
        minor_labels=["-2000bp", "Gene body", "+2000bp"]
    )

Example of resulting image:

.. image:: images/genome/CG_DMR_density.png
    :width: 600