from __future__ import annotations

from collections.abc import Generator
from typing import Optional, Union

import func_timeout
import pyarrow as pa
from pyarrow import csv as pcsv, parquet as pq
from pydantic import BaseModel, Field

from ..schemas import ReportSchema
from ..types import ExistentPath, Mb2Bytes
from .batches import UniversalBatch


class ArrowParquetReader2(BaseModel):
    file: ExistentPath
    report_schema: ReportSchema
    use_cols: Optional[list[str]] = Field(default=None)
    use_threads: bool = Field(default=True)

    @property
    def reader(self):
        return pq.ParquetFile(self.file)

    def __len__(self):
        return self.reader.num_row_groups

    def __iter__(self):
        i = 0
        while i < self.__len__():
            raw = self.reader.read_row_group(i, self.use_cols, self.use_threads)
            yield UniversalBatch.from_arrow(raw, self.report_schema)
        self.reader.close()

    @property
    def batch_size(self):
        return int(self.file.stat().st_size / self.reader.num_row_groups)


class ArrowReaderCSV2(BaseModel):
    file: ExistentPath
    block_size_mb: Mb2Bytes
    report_schema: ReportSchema
    memory_pool: Union[type[pa.MemoryPool], pa.MemoryPool] = Field(default_factory=pa.system_memory_pool, exclude=True)
    kwargs: dict = Field(default_factory=dict)
    use_threads: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    @property
    def convert_options(self) -> pcsv.ConvertOptions:
        return pcsv.ConvertOptions(
            column_types=self.report_schema.value.arrow, strings_can_be_null=False
        )

    @property
    def parse_options(self) -> pcsv.ParseOptions:
        return pcsv.ParseOptions(
            delimiter="\t",
            quote_char=False,
            escape_char=False,
            ignore_empty_lines=True,
            invalid_row_handler=lambda _: "skip",  # TODO maybe add logging
        )

    @property
    def read_options(self) -> pcsv.ReadOptions:
        return pcsv.ReadOptions(
            use_threads=self.use_threads,
            block_size=self.block_size_mb,
            skip_rows=0,
            skip_rows_after_names=0,
            column_names=self.report_schema.value.arrow.names,
        )

    @func_timeout.func_set_timeout(20)
    def init_arrow(self) -> pcsv.CSVStreamingReader:
        try:
            reader = pcsv.open_csv(
                self.file,
                self.read_options,
                self.parse_options,
                self.convert_options,
                self.memory_pool,
            )
            return reader
        except pa.ArrowInvalid:
            print(f"Error openning file: {self.file}")

    def __iter__(self) -> Generator[UniversalBatch]:
        reader = self.init_arrow()
        while True:
            try:
                raw = reader.read_next_batch()
                yield UniversalBatch.from_arrow(raw, self.report_schema, **self.kwargs)

            except pa.ArrowInvalid:
                # Todo add logging
                continue
            except StopIteration:
                break
        reader.close()
