Binomial
========

.. currentmodule:: bsxplorer

^^^^^^^^^
Reference
^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _Binom
    :template: class.rst

    BinomialData
    RegionStat

^^^^^
Usage
^^^^^

BSXplorer offers methods to define cytosine methylation status. Assuming the distribution of
count of the methylated cytosine reads is binomial, BSXplorer runs a binomial test to
calculate cytosine methylation p-value. :func:`BinomialData.from_report` reads the methylation
report and saves cytosine methylation p-value into .parquet file for further usage.

.. code-block:: python

    import bsxplorer as bsx

    binom = bsx.BinomialData.from_report(
        file="/path/to/report.txt",
        report_type="bismark",
        save="savename",
        dir="/path/to/work_dir"
    )

    # class BinomialData can also be initialized from preprocessed file directly
    # binom = bsx.BinomialData("path/to/preprocessed.parquet")

Preprocessed methylation p-value report can be used as input report for :class:`Metagene` or
:class:`ChrLevels` or can be utilized for Gene Body Methylation (gBM) [1]_ epigenetic mark
categorization with :class:`RegionStat`.

.. code-block:: python

    genome = bsx.Genome.from_gff("path/to/annot.gff").gene_body(min_length=0)
    region_stat = binom.region_pvalue(genome, methylation_pvalue=.05)

    bm, im, um = region_stat.categorise(context="CG", p_value=.05)

This way regions are categorised as BM (Body Methylation), IM (Intermediate Methylated)
and UM (Undermethylated).

.. [1] Takuno, S., & Gaut, B. S. (2013).
       Gene body methylation is conserved between plant orthologs and is of evolutionary consequence.
       Proceedings of the National Academy of Sciences of the United States of America, 110(5),
       1797–1802. https://doi.org/10.1073/pnas.1215380110