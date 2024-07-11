from __future__ import annotations

import gc
import gzip
import os
import shutil
import tempfile
import warnings
from abc import abstractmethod
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import func_timeout
import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import csv as pcsv, parquet as pq

from .UniversalReader_batches import UniversalBatch, ARROW_SCHEMAS, ReportTypes, REPORT_TYPES_LIST
from .utils import ReportBar


# noinspection PyMissingOrEmptyDocstring
def invalid_row_handler(row):
    print(f"Got invalid row: {row}")
    return "skip"


# noinspection PyMissingOrEmptyDocstring
@func_timeout.func_set_timeout(20)
def open_csv(
        file: str | Path,
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


# noinspection PyMissingOrEmptyDocstring
class ArrowParquetReader:
    def __init__(self,
                 file: str | Path,
                 use_cols: list = None,
                 use_threads: bool = True,
                 **kwargs):
        self.file = Path(file).expanduser().absolute()
        if not self.file.exists():
            raise FileNotFoundError()

        self.reader = pq.ParquetFile(file)
        self.__current_group = 0

        self.__use_cols = use_cols
        self.__use_threads = use_threads

    def __len__(self):
        return self.reader.num_row_groups

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()

    def __iter__(self):
        return self

    def _mutate_next(self, batch: pa.RecordBatch):
        return batch

    def __next__(self) -> UniversalBatch:
        old_group = self.__current_group
        # Check if it is the last row group
        if old_group < self.reader.num_row_groups:
            self.__current_group += 1

            batch = self.reader.read_row_group(
                old_group,
                columns=self.__use_cols,
                use_threads=self.__use_threads
            )

            mutated = self._mutate_next(batch)
            return mutated
        raise StopIteration()

    @property
    def batch_size(self):
        return int(self.file.stat().st_size / self.reader.num_row_groups)


# noinspection PyMissingOrEmptyDocstring
class BinomReader(ArrowParquetReader):
    """
    Class for reading methylation data from .parquet files generated by :class:`BinomialData`.

    Parameters
    ----------
    file
        Path to the .parquet file.
    methylation_pvalue
        Pvalue with which cytosine will be considered methylated.
    """

    def __init__(self, file: str | Path, methylation_pvalue=.05, **kwargs):
        super().__init__(file, **kwargs)

        if 0 < methylation_pvalue <= 1:
            self.methylation_pvalue = methylation_pvalue
        else:
            self.methylation_pvalue = 0.05
            warnings.warn(
                f"P-value needs to be in (0;1] interval, not {methylation_pvalue}. Setting to default ({self.methylation_pvalue})")

    def _mutate_next(self, batch: pa.RecordBatch):
        df = pl.from_arrow(batch)

        mutated = (
            df
            .with_columns(
                (pl.col("p_value") <= self.methylation_pvalue).cast(pl.UInt8).alias("count_m")
            )
            .with_columns([
                pl.col("context").alias("trinuc"),
                pl.lit(1).alias("count_total"),
                pl.col("count_m").cast(pl.Float64).alias("density")
            ])
        )

        return UniversalBatch(mutated, batch)


# noinspection PyMissingOrEmptyDocstring
class ArrowReaderCSV:
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
        self.file = file
        self.batch_size = block_size_mb * 1024 ** 2
        self._current_batch = None

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def get_reader(
            self,
            read_options: pcsv.ReadOptions = None,
            parse_options: pcsv.ParseOptions = None,
            convert_options: pcsv.ConvertOptions = None,
            memory_pool=None
    ) -> pcsv.CSVStreamingReader:
        """
        This function is needed, because if Arrow tries to open CSV with wrong schema, it gets stuck on it forever.
        This function makes a timout on initializing reader
        """
        try:
            reader = open_csv(
                self.file,
                read_options,
                parse_options,
                convert_options,
                memory_pool
            )

            return reader
        except pa.ArrowInvalid as e:
            print(f"Error opening file: {self.file}")
            raise e
        except func_timeout.exceptions.FunctionTimedOut:
            print(
                "Time for oppening file exceeded. Check if input type is correct or try making your batch size smaller.")
            os._exit(0)

    def _mutate_next(self, batch: pa.RecordBatch) -> UniversalBatch:
        return UniversalBatch(pl.from_arrow(batch), batch)

    def __next__(self):
        try:
            raw = self.reader.read_next_batch()
            batch = self._mutate_next(raw)
            return batch
        except pa.ArrowInvalid as e:
            print(e)
            return self.__next__()
        except StopIteration:
            raise StopIteration

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()


# noinspection PyMissingOrEmptyDocstring
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
            memory_pool: pa.MemoryPool = None,
            **kwargs
    ):
        super().__init__(file, block_size_mb)
        self.reader = self.get_reader(
            self.read_options(use_threads=use_threads, block_size=self.batch_size),
            self.parse_options(),
            self.convert_options(),
            memory_pool=memory_pool
        )

    def _mutate_next(self, batch: pa.RecordBatch):
        mutated = (
            pl.from_arrow(batch)
            .with_columns((pl.col("count_m") + pl.col("count_um")).alias("count_total"))
            .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
        )
        return UniversalBatch(mutated, batch)


# noinspection PyMissingOrEmptyDocstring
class CGMapReader(BismarkReader):
    pa_schema = ARROW_SCHEMAS["cgmap"]

    def _mutate_next(self, batch: pa.RecordBatch):
        mutated = (
            pl.from_arrow(batch)
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
        return UniversalBatch(mutated, batch)


# noinspection PyMissingOrEmptyDocstring
class CoverageReader(ArrowReaderCSV):
    pa_schema = ARROW_SCHEMAS["coverage"]

    read_options = BismarkReader.read_options
    parse_options = BismarkReader.parse_options
    convert_options = BismarkReader.convert_options

    def __init__(
            self,
            file: str | Path,
            cytosine_file: str | Path,
            use_threads: bool = True,
            block_size_mb: int = 50,
            memory_pool: pa.MemoryPool = None,
            **kwargs
    ):
        super().__init__(file, block_size_mb)
        self.reader = self.get_reader(
            self.read_options(use_threads=use_threads, block_size=self.batch_size),
            self.parse_options(),
            self.convert_options(),
            memory_pool=memory_pool
        )

        self.cytosine_file = cytosine_file
        self._sequence_rows_read = 0
        self._sequence_metadata = pq.read_metadata(cytosine_file)

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
            pa_filtered_sequence = pq.read_table(self.cytosine_file, filters=filters)

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
        mutated = (
            self._align(batch)
            .with_columns([
                (pl.col("count_m") + pl.col("count_um")).alias("count_total"),
                pl.col("context").alias("trinuc")
            ])
            .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
        )
        return UniversalBatch(mutated, batch)


# noinspection PyMissingOrEmptyDocstring
class BedGraphReader(CoverageReader):
    pa_schema = ARROW_SCHEMAS["bedgraph"]

    def __init__(self, file: str | Path, cytosine_file: str | Path, max_count=100, **kwargs):
        super().__init__(file, cytosine_file, **kwargs)
        self.max_count = max_count

    def _get_fraction(self, df: pl.DataFrame):
        def fraction(n, limit):
            f = Fraction(n).limit_denominator(limit)
            return f.numerator, f.denominator

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

        return UniversalBatch(full, batch)


class UniversalReader(object):
    """
    Class for batched reading methylation reports.

    Parameters
    ----------
    file
        Path to the methylation report file.
    report_type
        Type of the methylation report. Possible types: "bismark", "cgmap", "bedgraph", "coverage", "binom"
    use_threads
        Will reading be multithreaded.
    bar
        Indicate the progres bar while reading.
    kwargs
        Specific keyword arguments to the reader.

    Other Parameters
    ----------------
    cytosine_file: str | Path
        Instance of :class:`Sequence` for reading bedGraph or .coverage reports.
    methylation_pvalue: float
        Pvalue with which cytosine will be considered methylated.
    block_size_mb: int
        Size of batch in bytes, which will be read from report file (for report types other than "binom").

    Examples
    --------
    >>> reader = UniversalReader("path/to/file.txt", report_type="bismark", use_threads=True)
    >>> for batch in reader:
    >>>     do_something(batch)
    """

    def __init__(
            self,
            file: str | Path,
            report_type: ReportTypes,
            use_threads: bool = False,
            bar: bool = True,
            **kwargs
    ):

        super().__init__()
        self.__validate(file, report_type)
        self.__decompressed = self.__decompress()
        if self.__decompressed is not None:
            self.file = Path(self.__decompressed.name)

        self.memory_pool = pa.system_memory_pool()
        self.reader = self.__readers[report_type](file, use_threads=use_threads, memory_pool=self.memory_pool, **kwargs)
        self.report_type = report_type
        self._bar_on = bar
        self.bar = None

    __readers = {
        "bismark": BismarkReader,
        "cgmap": CGMapReader,
        "bedgraph": BedGraphReader,
        "coverage": CoverageReader,
        "binom": BinomReader
    }

    def __validate(self, file, report_type):
        file = Path(file).expanduser().absolute()
        if not file.exists():
            raise FileNotFoundError(file)
        if not file.is_file():
            raise IsADirectoryError(file)
        self.file = file

        if report_type not in self.__readers.keys():
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
        if self.bar is not None:
            self.bar.goto(self.bar.max)
            self.bar.finish()
        self.reader.__exit__(exc_type, exc_val, exc_tb)
        self.memory_pool.release_unused()

    def __iter__(self):
        if self._bar_on:
            self.bar = ReportBar(max=self.file_size)
            self.bar.start()
        return self

    def __next__(self) -> UniversalBatch:
        full_batch = self.reader.__next__()

        if self.bar is not None:
            self.bar.next(self.batch_size)

        return full_batch

    @property
    def batch_size(self):
        """

        Returns
        -------
        int
            Size of batch in bytes.
        """
        return self.reader.batch_size

    @property
    def file_size(self):
        """

        Returns
        -------
        int
            File size in bytes.
        """
        return self.file.stat().st_size


class UniversalReplicatesReader(object):
    """
    Class for reading from replicates methylation reports. The reader sums up the methylation counts.

    Parameters
    ----------
    readers
        List of initialized instances of :class:`UniversalReader`.
    bar
        Indicate the progres bar while reading.

    Examples
    --------
    >>> reader1 = UniversalReader("path/to/file1.txt", report_type="bismark", use_threads=True)
    >>> reader2 = UniversalReader("path/to/file2.txt", report_type="bismark", use_threads=True)
    >>>
    >>> for batch in UniversalReplicatesReader([reader1, reader2]):
    >>>     do_something(batch)
    """

    def __init__(
            self,
            readers: list[UniversalReader]
    ):
        self.readers = readers
        self.haste_limit = 1e9
        self.bar = None

        if any(map(lambda reader: reader.report_type in ["bedgraph"], self.readers)):
            warnings.warn("Merging bedGraph may lead to incorrect results. Please, use other report types.")

    def __iter__(self):
        self.bar = ReportBar(max=self.full_size)
        self.bar.start()

        self._seen_chroms = []
        self._unfinished = None

        self._readers_data = {
            idx: dict(iterator=iter(reader), read_rows=0, haste=0, finished=False, name=reader.file, chr=None, pos=None)
            for reader, idx in zip(self.readers, range(len(self.readers)))}

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self.readers:
            reader.__exit__(exc_type, exc_val, exc_tb)
        if self.bar is not None:
            self.bar.goto(self.bar.max)
            self.bar.finish()

    def __next__(self):
        # Key is readers' index
        reading_from = [key for key, value in self._readers_data.items() if
                        (value["haste"] < self.haste_limit and not value["finished"])]

        # Read batches from not hasting readers
        batches_data = {}
        for key in reading_from:
            # If batch is present, drop "density" col and set "chr" and "position" sorted for further sorted merge
            try:
                batch = next(self._readers_data[key]["iterator"])
                batches_data[key] = (
                    batch.data
                    .set_sorted(["chr", "position"])
                    .drop("density")
                    .with_columns(pl.lit([np.uint8(key)]).alias("group_idx"))
                )
                # Upd metadata
                self._readers_data[key]["read_rows"] += len(batch)
                self._readers_data[key] |= dict(chr=batch.data[-1]["chr"].to_list()[0],
                                                pos=batch.data[-1]["position"].to_list()[0])
            # Else mark reader as finished
            except StopIteration:
                self._readers_data[key]["finished"] = True

        if not batches_data and len(self._unfinished) == 0:
            raise StopIteration
        elif batches_data:
            self.haste_limit = sum(len(batch) for batch in batches_data.values()) // 3

        # We assume that input file is sorted in some way
        # So we gather chromosomes order in the input files to understand which is last
        if self._unfinished is not None:
            batches_data[-1] = self._unfinished

        pack_seen_chroms = []
        for key in batches_data.keys():
            batch_chrs = batches_data[key]["chr"].unique(maintain_order=True).to_list()
            [pack_seen_chroms.append(chrom) for chrom in batch_chrs if chrom not in pack_seen_chroms]
        [self._seen_chroms.append(chrom) for chrom in pack_seen_chroms if chrom not in self._seen_chroms]

        # Merge read batches and unfinished data with each other and then group
        merged = []
        for chrom in pack_seen_chroms:
            # Retrieve first batch (order doesn't matter) with which we will merge
            chr_merged = batches_data[list(batches_data.keys())[0]].filter(chr=chrom)
            # Get keys of other batches
            other = [key for key in batches_data.keys() if key != list(batches_data.keys())[0]]
            for key in other:
                chr_merged = chr_merged.merge_sorted(batches_data[key].filter(chr=chrom), key="position")
            merged.append(chr_merged)
        merged = pl.concat(merged).set_sorted(["chr", "position"])

        # Group merged rows by chromosome and position and check indexes that have merged
        grouped = (
            merged.lazy()
            .group_by(["chr", "position"], maintain_order=True)
            .agg([
                pl.first("strand", "context", "trinuc"),
                pl.sum("count_m", "count_total"),
                pl.col("group_idx").explode()
            ])
            .with_columns(pl.col("group_idx").list.len().alias("group_count"))
        )

        # Finished rows are those which have grouped among all readers
        min_chr_idx = min(self._seen_chroms.index(reader_data["chr"]) for reader_data in self._readers_data.values())
        min_position = min(reader_data["pos"] for reader_data in self._readers_data.values() if
                           reader_data["chr"] == self._seen_chroms[min_chr_idx])

        # Finished if all readers have grouped or there is no chance to group because position is already skipped
        marked = (
            grouped
            .with_columns([
                pl.col("chr").replace(self._seen_chroms, list(range(len(self._seen_chroms)))).cast(pl.Int8).alias(
                    "chr_idx"),
                pl.lit(min_position).alias("min_pos")
            ])
            .with_columns(
                pl.when(
                    (pl.col("group_count") == len(self._readers_data)) | (
                                (pl.col("chr_idx") < min_chr_idx) | (pl.col("chr_idx") == min_chr_idx) & (
                                    pl.col("position") < pl.col("min_pos")))
                ).then(pl.lit(True)).otherwise(pl.lit(False)).alias("finished")
            ).collect()
        )

        self._unfinished = marked.filter(finished=False).select(merged.columns)

        hasting_stats = defaultdict(int)
        if len(self._unfinished) > 0:
            group_idx_stats = self._unfinished.select(pl.col("group_idx").list.to_struct()).unnest("group_idx")
            for col in group_idx_stats.columns:
                hasting_stats[group_idx_stats[col].drop_nulls().item(0)] += group_idx_stats[col].count()

        for key in self._readers_data.keys():
            self._readers_data[key]["haste"] = hasting_stats[key]

        # Update bar
        if reading_from:
            self.bar.next(sum(self.readers[idx].batch_size for idx in reading_from))

        out = marked.filter(finished=True)
        return UniversalBatch(self._convert_to_full(out), out)

    @staticmethod
    def _convert_to_full(df: pl.DataFrame):
        return (
            df.with_columns((pl.col("count_m") / pl.col("count_total")).alias("density")).select(
                UniversalBatch.colnames())
        )

    @property
    def full_size(self):
        """

        Returns
        -------
        int
            Total size of readers' files in bytes.
        """
        return sum(map(lambda reader: reader.file_size, self.readers))


class UniversalWriter:
    """
    Class for writing reports in specific methylation report format.

    Parameters
    ----------
    file
        Path where the file will be written.
    report_type
        Type of the methylation report. Possible types: "bismark", "cgmap", "bedgraph", "coverage", "binom"
    """

    def __init__(
            self,
            file: str | Path,
            report_type: ReportTypes,
    ):
        file = Path(file).expanduser().absolute()
        self.file = file

        if report_type not in REPORT_TYPES_LIST:
            raise KeyError(report_type)
        self.report_type: str = report_type

        self.memory_pool = pa.system_memory_pool()
        self.writer = None

    def __enter__(self):
        self.open()
        return self

    def close(self):
        """
        This method should be called after writing all data, when writer is called without `with` context manager.
        """
        self.writer.close()
        self.memory_pool.release_unused()

    def open(self):
        """
        This method should be called before writing data, when writer is called without `with` context manager.
        """
        write_options = pcsv.WriteOptions(
            delimiter="\t",
            include_header=False,
            quoting_style="none"
        )

        self.writer = pcsv.CSVWriter(
            self.file,
            ARROW_SCHEMAS[self.report_type],
            write_options=write_options,
            memory_pool=self.memory_pool
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

        if self.report_type == "bismark":
            fmt_df = universal_batch.to_bismark()
        elif self.report_type == "cgmap":
            fmt_df = universal_batch.to_cgmap()
        elif self.report_type == "bedgraph":
            fmt_df = universal_batch.to_bedGraph()
        elif self.report_type == "coverage":
            fmt_df = universal_batch.to_coverage()
        else:
            raise KeyError(f"{self.report_type} not supported")

        self.writer.write(fmt_df.to_arrow().cast(ARROW_SCHEMAS[self.report_type]))
