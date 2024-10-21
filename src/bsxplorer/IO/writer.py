from __future__ import annotations

from pathlib import Path

import pyarrow as pa
from pyarrow import csv as pcsv

from ..schemas import ReportSchema
from .batches import UniversalBatch


class UniversalWriter:
    """
    Class for writing reports in specific methylation report format.

    Parameters
    ----------
    file
        Path where the file will be written.
    report_type
        Type of the methylation report. Possible types: "bismark", "cgmap", "bedgraph",
        "coverage", "binom"
    """

    def __init__(
        self,
        file: str | Path,
        report_type: ReportSchema,
    ):
        file = Path(file).expanduser().absolute()
        self.file = file

        if report_type not in ReportSchema.__members__:
            raise KeyError(report_type)
        self.report_type = report_type

        self.memory_pool = pa.system_memory_pool()
        self.writer = None

    def __enter__(self):
        self.open()
        return self

    def close(self):
        """
        This method should be called after writing all data, when writer is called
        without `with` context manager.
        """
        self.writer.close()
        self.memory_pool.release_unused()

    def open(self):
        """
        This method should be called before writing data, when writer is called without
        `with` context manager.
        """
        write_options = pcsv.WriteOptions(
            delimiter="\t", include_header=False, quoting_style="none"
        )

        self.writer = pcsv.CSVWriter(
            self.file,
            self.report_type.arrow,
            write_options=write_options,
            memory_pool=self.memory_pool,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, universal_batch: UniversalBatch):
        """
        Method for writing batch to the report file.
        """
        if universal_batch is None:
            return

        if self.writer is None:
            self.__enter__()

        if self.report_type == ReportSchema.BISMARK:
            fmt_df = universal_batch.cast(ReportSchema.BISMARK)
        elif self.report_type == ReportSchema.CGMAP:
            fmt_df = universal_batch.cast(ReportSchema.CGMAP)
        elif self.report_type == ReportSchema.BEDGRAPH:
            fmt_df = universal_batch.cast(ReportSchema.BEDGRAPH)
        elif self.report_type == ReportSchema.COVERAGE:
            fmt_df = universal_batch.cast(ReportSchema.COVERAGE)
        else:
            raise KeyError(f"{self.report_type} not supported")

        self.writer.write(fmt_df.to_arrow().cast(self.report_type.arrow))
