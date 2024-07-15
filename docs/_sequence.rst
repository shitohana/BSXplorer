Cluster
========

.. currentmodule:: bsxplorer

^^^^^^^^^
Reference
^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _Sequence
    :template: class.rst

    SequenceFile

^^^^^
Usage
^^^^^

For some tasks, such as converting .bedGraph files to Bismark reports, BSXplorer
needs information about all cytosines in reference genome. For such tasks the user need
to preprocess reference genome with :class:`SequenceFile`.

.. code-block:: python

    import bsxplorer as bsx

    file = bsx.SequenceFile("path/to/genomeseq.fa")
    file.preprocess_cytosines("path/to/output.parquet")

