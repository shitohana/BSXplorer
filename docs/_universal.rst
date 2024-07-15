Universal I/O
=================

.. currentmodule:: bsxplorer

^^^^^^^^^
Reference
^^^^^^^^^

.. autosummary::
    :toctree: _Universal
    :nosignatures:
    :template: class_with_init.rst

        UniversalReader

        UniversalReplicatesReader

        UniversalWriter

        UniversalBatch

-------------------
Reading and writing
-------------------


For reading methylation reports of different types BSXplorer offers :class:`UniversalReader` and
:class:`UniversalReplicatesReader`. They allow user to iterate over methylation report data in
fast and convinient way. :class:`UniversalReplicatesReader` merges methylation data from several
methylation reports of biological replicates and returns merged data.

.. code-block:: python

    import bsxplorer as bsx

    # For single methylation report
    reader = bsx.UniversalReader("path/to/file.txt", report_type="bismark", use_threads=True)

    for batch in reader:
        # Note that the returned batch is instance of UniversalBatch
        do_something(batch)

    # For reading replicates, firstly initialize single readers
    reader1 = bsx.UniversalReader("path/to/file1.txt", report_type="bismark", use_threads=True)
    reader2 = bsx.UniversalReader("path/to/file2.txt", report_type="bismark", use_threads=True)

    # Than you can initialilize UniversalReplicatesReader class with them
    for batch in bsx.UniversalReplicatesReader([reader1, reader2]):
        do_something(batch)

BSXplorer inner methylation data format is :class:`UniversalBatch`, which stores maximum available
information about cytosine methylation status and context. :class:`UniversalBatch` `data` attribute
stores methylation information in `polars.DataFrame` with schema:

.. list-table:: UniversalBatch.data Schema
    :header-rows: 1

    * - Field name
      - Data type
      - Description

    * - strand
      - Utf8
      - DNA strand

    * - position
      - UInt64
      - Chromosome position

    * - context
      - Utf8
      - Methylation context

    * - trinuc
      - Utf8
      - Cytosine trinucleotide sequence

    * - count_m
      - UInt32
      - Count of methylated reads

    * - count_total
      - UInt32
      - Total reads

    * - density
      - Float64
      - Methylation density (NaN if no reads cover cytosine)



For converting one report type into another, BSXplorer offers :class:`UniversalWriter`,
which accepts :class:`UniversalBatch` as an input and writes it into the file with specified format.

.. code-block:: python

    import bsxplorer as bsx

    # For single methylation report
    reader = bsx.UniversalReader("path/to/file.txt", report_type="bismark", use_threads=True)

    with bsx.UniversalWriter("path/to/out.txt", report_type="cgmap") as writer:
        for batch in reader:
            writer.write(batch)

.. note::
    :class:`UniversalWriter` accepts only :class:`UniversalBatch` as input for :func:`UniversalWriter.write`.