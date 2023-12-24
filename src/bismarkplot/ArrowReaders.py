from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq


class CsvReader(object):
    def __init__(self,
                 file: str | Path,
                 options: CsvOptions,
                 memory_pool: pa.MemoryPool = None):

        self.reader = pcsv.open_csv(file,
                                    options.read_options,
                                    options.parse_options,
                                    options.convert_options,
                                    memory_pool)

        self.__current_batch = self.reader.read_next_batch()

    def __iter__(self):
        return self

    def __next__(self) -> pa.RecordBatch:
        return self.next()

    def next(self) -> pa.RecordBatch:
        old_batch = self.__current_batch
        self.__current_batch = self.reader.read_next_batch()

        if self.__current_batch.num_rows != 0:
            return old_batch
        raise StopIteration()


class ParquetReader(object):
    def __init__(self,
                 file: str | Path,
                 use_cols: list = None,
                 use_threads: bool = True):
        self.pq_file = pq.ParquetFile(file)
        self.__current_group = 0

        self.__use_cols = use_cols
        self.__use_threads = use_threads

    def __iter__(self):
        return self

    def __next__(self) -> pa.Table:
        return self.next()

    def next(self) -> pa.Table:
        old_group = self.__current_group
        if old_group < self.pq_file.num_row_groups:
            self.__current_group += 1
            return self.pq_file.read_row_group(old_group, columns=self.__use_cols, use_threads=self.__use_threads)
        raise StopIteration()


class CsvOptions:
    def __init__(self):
        self.read_options = pcsv.ReadOptions()
        self.parse_options = pcsv.ParseOptions()
        self.convert_options = pcsv.ConvertOptions()


class BismarkOptions(CsvOptions):
    def __init__(self,
                 use_threads: bool = True,
                 block_size: int = None):
        super().__init__()

        schema = dict(
            chr=pa.utf8(),
            position=pa.uint64(),
            strand=pa.utf8(),
            count_m=pa.uint32(),
            count_um=pa.uint32(),
            context=pa.utf8(),
            trinuc=pa.utf8()
        )
        include_cols = ["chr", "position", "strand", "count_m", "count_um", "context"]

        self.read_options = pcsv.ReadOptions(
            use_threads=use_threads,
            block_size=block_size,
            skip_rows=0,
            skip_rows_after_names=0,
            column_names=["chr", "position", "strand", "count_m", "count_um", "context", "trinuc"]
        )

        self.parse_options = pcsv.ParseOptions(
            delimiter="\t",
            quote_char=False,
            escape_char=False
        )

        self.convert_options = pcsv.ConvertOptions(
            column_types=schema,
            strings_can_be_null=False,
            include_columns=include_cols
        )

