from __future__ import annotations

import gc
import gzip
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path
from typing import Literal

import func_timeout
import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import csv as pcsv, parquet as pq

from .SeqMapper import Sequence
from .UniversalReader_batches import FullSchemaBatch, ARROW_SCHEMAS
from .utils import ReportBar


def invalid_row_handler(row):
    print(f"Got invalid row: {row}")
    return("skip")


@func_timeout.func_set_timeout(20)
def open_csv(
        file : str | Path,
        read_options: pcsv.ReadOptions = None,
        parse_options: pcsv.ParseOptions = None,
        convert_options: pcsv.ConvertOptions = None,
        memory_pool=None
):
    return pcsv.open_csv(
            file,
            read_options,
            parse_options,
            convert_options,
            memory_pool
    )


class ArrowReaderBase(ABC, object):
    def __enter__(self):
        return self

    def __iter__(self):
        return self


    def _mutate_next(self, batch: pa.RecordBatch):
        return batch

    def __next__(self) -> pa.RecordBatch:
        raw = self.next_raw()
        if raw is not None:
            batch = self._mutate_next(raw)
            return batch
        return raw

    def __init__(self, file: str | Path):
        self.file = Path(file).expanduser().absolute()
        if not self.file.exists():
            raise FileNotFoundError()
        self._current_batch = None

    @abstractmethod
    def next_raw(self) -> pa.RecordBatch:
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


class ArrowReaderCSV(ArrowReaderBase):
    @property
    @abstractmethod
    def pa_schema(self):
        ...

    @abstractmethod
    def convert_options(self, **kwargs):
        ...

    @abstractmethod
    def parse_options(self, **kwargs):
        ...

    @abstractmethod
    def read_options(self, **kwargs):
        ...

    def __init__(self, file, block_size_mb):
        super().__init__(file)
        self.batch_size = block_size_mb  * 1024**2

    def get_reader(
            self,
            read_options: pcsv.ReadOptions = None,
            parse_options: pcsv.ParseOptions = None,
            convert_options: pcsv.ConvertOptions = None,
            memory_pool=None
    ) -> pcsv.CSVStreamingReader:

        try:
            reader = open_csv(
                self.file,
                read_options,
                parse_options,
                convert_options,
                memory_pool
            )
        except pa.ArrowInvalid as e:
            print(f"Error opening file: {self.file}")
            raise e
        except func_timeout.exceptions.FunctionTimedOut:
            print("Time for oppening file exceeded. Check if input type is correct or try making your batch size smaller.")
            os._exit(0)

        return reader

    def next_raw(self) -> pa.RecordBatch:
        old_batch = self._current_batch
        try:
            self._current_batch = self.reader.read_next_batch()
        except pa.ArrowInvalid as e:
            print(e)
            return self.next_raw()
        if self._current_batch.num_rows != 0:
            return old_batch
        raise StopIteration()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()


class BismarkReader(ArrowReaderCSV):
    pa_schema = ARROW_SCHEMAS["bismark"]

    def convert_options(self):
        return pcsv.ConvertOptions(
            column_types=self.pa_schema,
            strings_can_be_null=False
        )

    def parse_options(self):
        return pcsv.ParseOptions(
            delimiter="\t",
            quote_char=False,
            escape_char=False,
            ignore_empty_lines=True,
            invalid_row_handler=lambda _: "skip"
        )

    def read_options(self, use_threads=True, block_size=None):
        return pcsv.ReadOptions(
            use_threads=use_threads,
            block_size=block_size,
            skip_rows=0,
            skip_rows_after_names=0,
            column_names=self.pa_schema.names
        )

    def __init__(
            self,
            file: str | Path,
            use_threads: bool = True,
            block_size_mb: int = 50,
            memory_pool: pa.MemoryPool = None
    ):
        super().__init__(file, block_size_mb)
        self.reader = self.get_reader(
            self.read_options(use_threads=use_threads, block_size=self.batch_size),
            self.parse_options(),
            self.convert_options(),
            memory_pool=memory_pool
        )


class CGMapReader(BismarkReader):
    pa_schema = ARROW_SCHEMAS["cgmap"]


class CoverageReader(ArrowReaderCSV):
    pa_schema = ARROW_SCHEMAS["coverage"]

    read_options = BismarkReader.read_options
    parse_options = BismarkReader.parse_options
    convert_options = BismarkReader.convert_options

    def __init__(
            self,
            file: str | Path,
            sequence: Sequence,
            use_threads: bool = True,
            block_size_mb: int = 50,
            memory_pool: pa.MemoryPool = None
    ):
        super().__init__(file, block_size_mb)
        self.reader = self.get_reader(
            self.read_options(use_threads=use_threads, block_size=self.batch_size),
            self.parse_options(),
            self.convert_options(),
            memory_pool=memory_pool
        )

        self.sequence = sequence
        self._sequence_rows_read = 0
        self._sequence_metadata = sequence.get_metadata()

    def _align(self, pa_batch: pa.RecordBatch):
        batch = pl.from_arrow(pa_batch)

        # get batch stats
        batch_stats = batch.group_by("chr").agg([
            pl.col("position").max().alias("max"),
            pl.col("position").min().alias("min")
        ])

        output = None

        for chrom in batch_stats["chr"]:
            chrom_min, chrom_max = batch_stats.filter(chr=chrom).select(["min", "max"]).row(0)

            # Read and filter sequence
            filters = [
                ("chr", "=", chrom),
                ("position", ">=", chrom_min),
                ("position", "<=", chrom_max)
            ]
            pa_filtered_sequence = pq.read_table(self.sequence.cytosine_file, filters=filters)

            modified_schema = pa_filtered_sequence.schema
            modified_schema = modified_schema.set(1, pa.field("context", pa.utf8()))
            modified_schema = modified_schema.set(2, pa.field("chr", pa.utf8()))

            pa_filtered_sequence = pa_filtered_sequence.cast(modified_schema)

            pl_sequence = (
                pl.from_arrow(pa_filtered_sequence)
                .with_columns([
                    pl.when(pl.col("strand") == True).then(pl.lit("+"))
                    .otherwise(pl.lit("-"))
                    .alias("strand")
                ])
                .cast(dict(position=batch.schema["position"]))
            )

            # Align batch with sequence
            batch_filtered = batch.filter(pl.col("chr") == chrom)
            aligned = (
                batch_filtered.lazy()
                .cast({"position": pl.UInt64})
                .set_sorted("position")
                .join(pl_sequence.lazy(), on="position", how="left")
            ).collect()

            self._sequence_rows_read += len(pl_sequence)

            if output is None:
                output = aligned
            else:
                output.extend(aligned)

            gc.collect()

        return output

    def _mutate_next(self, batch: pa.RecordBatch):
        aligned = self._align(batch)
        return aligned.to_arrow()


class BedGraphReader(CoverageReader):
    pa_schema = ARROW_SCHEMAS["bedgraph"]

    read_options = BismarkReader.read_options
    parse_options = BismarkReader.parse_options
    convert_options = BismarkReader.convert_options

    def __init__(self, file: str | Path, sequence: Sequence, max_count=100, **kwargs):
        super().__init__(file, sequence, **kwargs)
        self.max_count = max_count

    def _get_fraction(self, df: pl.DataFrame):
        def fraction(n, limit):
            f = Fraction(n).limit_denominator(limit)
            return (f.numerator, f.denominator)

        fraction_v = np.vectorize(fraction)
        converted = fraction_v((df["density"] / 100).to_list(), self.max_count)

        final = (
            df.with_columns([
                pl.lit(converted[0]).alias("count_m"),
                pl.lit(converted[1]).alias("count_total")
            ])
        )

        return final

    def _mutate_next(self, batch: pa.RecordBatch):
        aligned = self._align(batch)
        fractioned = self._get_fraction(aligned)
        full = fractioned.with_columns(pl.col("context").alias("trinuc"))

        return full.to_arrow()


def cast2full_batch(
        batch: pa.RecordBatch | pa.Table | pl.DataFrame,
        from_type: str
) -> FullSchemaBatch:
    if not isinstance(batch, pl.DataFrame):
        pl_df = pl.from_arrow(batch)
    else:
        pl_df = batch
    mutated = pl_df

    if from_type == "bismark":
        mutated = (
            pl_df
            .with_columns((pl.col("count_m") + pl.col("count_um")).alias("count_total"))
            .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
        )

    elif from_type == "cgmap":
        mutated = (
            pl_df
            .with_columns([
                pl.when(pl.col("nuc") == "C").then(pl.lit("+")).otherwise(pl.lit("-")).alias("strand"),
                pl.when(pl.col("dinuc") == "CG")
                # if dinuc == "CG" => trinuc = "CG"
                .then(pl.col("dinuc"))
                # otherwise trinuc = dinuc + context_last
                .otherwise(pl.concat_str([pl.col("dinuc"), pl.col("context").str.slice(-1)]))
                .alias("trinuc")
            ])
        )
    elif from_type == "bedgraph":
        mutated = (
            pl_df
            .with_columns(pl.col("context").alias("trinuc"))
        )
    elif from_type == "coverage":
        mutated = (
            pl_df
            .with_columns([
                (pl.col("count_m") + pl.col("count_um")).alias("count_total"),
                pl.col("context").alias("trinuc")
            ])
            .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
        )
    elif from_type == "parquet":
        # TODO finish
        pass
    else:
        raise KeyError(from_type)

    return FullSchemaBatch(mutated)


class UniversalReader(object):
    def __init__(
            self,
            file: str | Path,
            report_type: Literal["bismark", "cgmap", "bedgraph", "coverage", "parquet"],
            use_threads: bool = True,
            **kwargs
    ):
        super().__init__()
        self.__validate(file, report_type)
        self.__decompressed = self.__decompress()
        if self.__decompressed is not None:
            self.file = self.__decompressed.name

        if report_type == "bismark":
            self.reader = BismarkReader(file, use_threads, **kwargs)
        elif report_type == "cgmap":
            self.reader = CGMapReader(file, use_threads, **kwargs)
        elif report_type == "bedgraph":
            self.reader = BedGraphReader(file, use_threads=use_threads, **kwargs)
        elif report_type == "coverage":
            self.reader = CoverageReader(file, use_threads=use_threads, **kwargs)
        elif report_type == "parquet":
            pass

        self.bar = None

    def __validate(self, file, report_type):
        file = Path(file).expanduser().absolute()
        if not file.exists():
            raise FileNotFoundError()
        if not file.is_file():
            raise IsADirectoryError()
        self.file = file

        allowed_types = ["bismark", "cgmap", "bedgraph", "coverage", "parquet"]
        if report_type not in allowed_types:
            raise KeyError(report_type)
        self.report_type: str = report_type

    def __decompress(self):
        if ".gz" in self.file.suffixes:
            temp_file = tempfile.NamedTemporaryFile(dir=self.file.parent, prefix=self.file.stem)
            print(f"Temporarily unpack {self.file} to {temp_file.name}")

            with gzip.open(self.file, mode="rb") as file:
                shutil.copyfileobj(file, temp_file)

            return temp_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__decompressed is not None:
            self.__decompressed.close()
        self.reader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        self.reader.__next__()
        self.bar = ReportBar(max=Path(self.file).stat().st_size)
        return self

    def __next__(self) -> FullSchemaBatch:
        next_batch = self.reader.__next__()
        converted = cast2full_batch(next_batch, self.report_type)

        if self.bar is not None:
            self.bar.next(self.batch_size)

        return converted

    @property
    def batch_size(self):
        return self.reader.batch_size
