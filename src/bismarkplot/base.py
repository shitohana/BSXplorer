import gzip

import polars as pl
from pyreadr import write_rds

from src.bismarkplot.utils import remove_extension


class BismarkBase:
    """
    Base class for :class:`Metagene` and plots.
    """

    def __init__(self, bismark_df: pl.DataFrame, **kwargs):
        """
        Base class for Bismark data.

        DataFrame Structure:

        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+
        | chr             | strand      | context             | gene                 | fragment         | sum            | count                                   |
        +=================+=============+=====================+======================+==================+================+=========================================+
        | Categorical     | Categorical | Categorical         | Categorical          | Int32            | Int32          | Int32                                   |
        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+
        | chromosome name | strand      | methylation context | position of cytosine | fragment in gene | sum methylated | count of all cytosines in this position |
        +-----------------+-------------+---------------------+----------------------+------------------+----------------+-----------------------------------------+


        :param bismark_df: pl.DataFrame with cytosine methylation status.
        :param upstream_windows: Number of upstream windows. Required.
        :param gene_windows: Number of gene windows. Required.
        :param downstream_windows: Number of downstream windows. Required.
        :param strand: Strand if filtered.
        :param context: Methylation context if filtered.
        :param plot_data: Data for plotting.
        """
        self.bismark: pl.DataFrame = bismark_df

        self.upstream_windows: int = kwargs.get("upstream_windows")
        self.downstream_windows: int = kwargs.get("downstream_windows")
        self.gene_windows: int = kwargs.get("gene_windows")
        self.plot_data: pl.DataFrame = kwargs.get("plot_data")
        self.context: str = kwargs.get("context")
        self.strand: str = kwargs.get("strand")

    @property
    def metadata(self) -> dict:
        """
        :return: Bismark metadata in dict
        """
        return {
            "upstream_windows": self.upstream_windows,
            "downstream_windows": self.downstream_windows,
            "gene_windows": self.gene_windows,
            "plot_data": self.plot_data,
            "context": self.context,
            "strand": self.strand
        }

    def save_rds(self, filename, compress: bool = False):
        """
        Save Bismark DataFrame in Rds.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        write_rds(filename, self.bismark.to_pandas(),
                  compress="gzip" if compress else None)

    def save_tsv(self, filename, compress=False):
        """
        Save Bismark DataFrame in TSV.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.bismark.write_csv(file, separator="\t")
        else:
            self.bismark.write_csv(filename, separator="\t")

    @property
    def total_windows(self):
        return self.upstream_windows + self.downstream_windows + self.gene_windows

    @property
    def tick_positions(self):
        return dict(
            up_mid=self.upstream_windows / 2,
            body_start=self.upstream_windows,
            body_mid=self.total_windows / 2,
            body_end=self.gene_windows + self.upstream_windows,
            down_mid=self.total_windows - (self.downstream_windows / 2)
        )

    def __len__(self):
        return len(self.bismark)


class BismarkFilesBase:
    def __init__(self, samples, labels: list[str] = None):
        self.samples = self.__check_metadata(
            samples if isinstance(samples, list) else [samples])
        if samples is None:
            raise Exception("Flank or gene windows number does not match!")
        self.labels = [str(v) for v in list(
            range(len(samples)))] if labels is None else labels
        if len(self.labels) != len(self.samples):
            raise Exception("Labels length doesn't match samples number")

    def save_rds(self, base_filename, compress: bool = False, merge: bool = False):
        if merge:
            merged = pl.concat(
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
            )
            write_rds(base_filename, merged.to_pandas(),
                      compress="gzip" if compress else None)
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_rds(
                    f"{remove_extension(base_filename)}_{label}.rds", compress="gzip" if compress else None)

    def save_tsv(self, base_filename, compress: bool = False, merge: bool = False):
        if merge:
            merged = pl.concat(
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
            )
            if compress:
                with gzip.open(base_filename + ".gz", "wb") as file:
                    # noinspection PyTypeChecker
                    merged.write_csv(file, separator="\t")
            else:
                merged.write_csv(base_filename, separator="\t")
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_tsv(
                    f"{remove_extension(base_filename)}_{label}.rds", compress=compress)

    @staticmethod
    def __check_metadata(samples: list[BismarkBase]):
        upstream_check = set([sample.metadata["upstream_windows"]
                             for sample in samples])
        downstream_check = set(
            [sample.metadata["downstream_windows"] for sample in samples])
        gene_check = set([sample.metadata["gene_windows"]
                         for sample in samples])

        if len(upstream_check) == len(gene_check) == len(downstream_check) == 1:
            return samples
        else:
            return None
