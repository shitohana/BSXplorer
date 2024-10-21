from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import pyarrow as pa
from pydantic import AliasChoices, BaseModel, Field, field_validator

from ..misc.schemas import ReportSchema
from ..misc.types import ExistentPath
from ..misc.utils import ReportBar
from .arrow_readers import ArrowParquetReader, ArrowReaderCSV
from .batches import UniversalBatch


class UniversalReader(BaseModel):
    """
    Class for batched reading methylation reports.

    Examples
    --------
    >>> reader = UniversalReader(
    ...     file="path/to/file.txt",
    ...     report_type="bismark",
    ...     use_threads=True,
    ... )
    >>> for batch in reader:
    >>>     do_something(batch)
    """

    file: ExistentPath = Field(
        title="Methylation file", description="Path to the methylation report file"
    )
    report_schema: ReportSchema = Field(
        validation_alias=AliasChoices("report_schema", "report_type"),
        title="Schema of methylation report",
        description="Either an instance of Enum :class:`ReportSchema` or one of the "
        'possible types: "bismark", "cgmap", "bedgraph", "coverage", '
        '"binom".',
    )
    use_threads: bool = Field(default=True, description="Will reading be multithreaded")
    bar_enabled: Optional[bool] = Field(
        default=False,
        validation_alias=AliasChoices("bar_enabled", "bar"),
        description="Indicate the progres bar while reading.",
    )
    allocator: Optional[Literal["system", "default", "mimalloc", "jemalloc"]] = Field(
        default="system"
    )
    cytosine_file: Optional[ExistentPath] = Field(
        default=None,
        title="Path to preprocessed cytosine file",
        description="Instance of :class:`Sequence` for reading bedGraph "
        "or .coverage reports.",
    )
    methylation_pvalue: Optional[Annotated[float, Field(gt=0, lt=1)]] = Field(
        default=None,
        title="Methylation PValue",
        description="Pvalue with which cytosine will be considered methylated.",
    )
    block_size_mb: Optional[Annotated[int, Field(gt=0)]] = Field(
        default=100,
        title="Size of batch in mb",
        description="Size of batch in bytes, which will be read from report file "
        '(for report typesother than "binom").',
    )

    @property
    def reader_kwargs(self):
        return dict(
            cytosine_file=self.cytosine_file,
            methylation_pvalue=self.methylation_pvalue,
            block_size_mb=self.block_size_mb,
        )

    @field_validator("report_schema", mode="before")
    @classmethod
    def _check_report_schema(cls, value):
        if isinstance(value, str):
            return ReportSchema(value.upper())
        elif isinstance(value, ReportSchema):
            return value
        else:
            return ValueError

    @property
    def memory_pool(self) -> Union[type[pa.MemoryPool], pa.MemoryPool]:
        if self.allocator == "system":
            return pa.system_memory_pool()
        if self.allocator == "default":
            return pa.default_memory_pool()
        if self.allocator == "mimalloc":
            return pa.mimalloc_memory_pool()
        if self.allocator == "jemalloc":
            return pa.jemalloc_memory_pool()

    @property
    def _bar(self) -> ReportBar | None:
        if self.bar_enabled:
            return ReportBar(max=self.file_size)
        else:
            return None

    @property
    def file_size(self):
        """

        Returns
        -------
        int
            File size in bytes.
        """
        return self.file.stat().st_size

    def init_reader(self, infile: Path):
        if self.report_schema in {
            ReportSchema.BISMARK,
            ReportSchema.COVERAGE,
            ReportSchema.CGMAP,
            ReportSchema.BEDGRAPH,
        }:
            return ArrowReaderCSV(
                file=infile,
                report_schema=self.report_schema,
                memory_pool=self.memory_pool,
                use_threads=self.use_threads,
                **self.reader_kwargs,
            )
        if self.report_schema == ReportSchema.BINOM:
            return ArrowParquetReader(
                file=infile,
                report_schema=self.report_schema,
                use_threads=self.use_threads,
                **self.reader_kwargs,
            )

    def __iter__(self) -> UniversalBatch:
        if ".gz" in self.file.suffixes:
            with (
                tempfile.NamedTemporaryFile(
                    dir=self.file.parent, prefix=self.file.stem, delete=False
                ) as infile,
                gzip.open(self.file, mode="rb") as zip_file,
            ):
                print(f"Temporarily unpack {self.file} to {infile.name}")
                shutil.copyfileobj(zip_file, infile)
                infile = Path(infile.name)
        else:
            infile = self.file

        if self._bar is not None:
            self._bar.finish()

        for batch in self.init_reader(infile):
            if self._bar is not None:
                self._bar.next()
            yield batch

        if self._bar is not None:
            self._bar.goto(self._bar.max)
            self._bar.finish()
        self.memory_pool.release_unused()
        if ".gz" in self.file.suffixes:
            infile.unlink()
