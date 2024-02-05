from __future__ import annotations

import gc
import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO as seqio
from numba import njit

import polars as pl

from .ArrowReaders import CsvReader, ParquetReader, BismarkOptions, CoverageOptions, BedGraphOptions

@njit
def convert_trinuc(trinuc, reverse=False):
    """
    Get trinucleotide context from raw trinucleotide
    :param trinuc: trinucleotide sequence
    :param reverse: is trinucleotide from reversed sequence
    :return: trinucleotide context
    """
    if reverse:
        if   trinuc[1] == "C": return "CG"
        elif trinuc[0] == "C": return "CHG"
        else:                  return "CHH"
    else:
        if   trinuc[1] == "G": return "CG"
        elif trinuc[2] == "G": return "CHG"
        else:                  return "CHH"


@njit
def get_trinuc(record_seq: str, reverse=False):
    """
    Parse sequence and extract trinucleotide contexts and positions
    :param record_seq: sequence
    :param reverse: does sequence need to be reversed
    :return: tuple(positions, contexts)
    """
    positions = []
    trinucs = []

    record_seq = record_seq.upper()

    nuc = "G" if reverse else "C"
    up_shift = 1 if reverse else 3
    down_shift = -2 if reverse else 0

    for position in range(2 if reverse else 0, len(record_seq) if reverse else len(record_seq) - 2):
        if record_seq[position] == nuc:
            positions.append(position + 1)
            trinuc = record_seq[position + down_shift:position + up_shift]
            trinucs.append(convert_trinuc(trinuc, reverse))

    return positions, trinucs


def init_tempfile(temp_dir, name, delete, suffix=".bedGraph.parquet") -> Path:
    """
    Init temporary cytosine file
    :param temp_dir: directory where file will be created
    :param name: filename
    :param delete: does file need to be deleted after script completion
    :return: temporary file
    """
    # temp cytosine file
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, delete=delete)

    # change name if not None
    if name is not None:
        new_path = Path(temp_dir) / (Path(name).stem + suffix)

        os.rename(temp_file.name, new_path)
        temp_file.name = new_path

    return Path(temp_file.name)


class Sequence:
    def __init__(self, cytosine_file: str | Path):
        """
        Class for extracting cytosine contexts and positions
        :param path: path to fasta sequence
        :param temp_dir: directory, where temporary file will be created
        :param name: filename of temporary file
        :param delete: does temporary file need to be deleted after script completion
        """
        self.cytosine_file = Path(cytosine_file)

    @classmethod
    def from_fasta(cls, path: str | Path, temp_dir: str = "./", name: str = None, delete: bool = True):
        """
        :param path: path to fasta sequence
        :param temp_dir: directory, where temporary file will be created
        :param name: filename of temporary file
        :param delete: does temporary file need to be deleted after script completion
        """
        path = Path(path).expanduser().absolute()

        cytosine_file = init_tempfile(temp_dir, name, delete, suffix=".parquet")
        sequence = cls(cytosine_file)

        # read sequence into cytosine file
        cls.__read_fasta_wrapper(sequence, fasta_path=path)

        return sequence

    @classmethod
    def from_preprocessed(cls, path: str | Path):
        """
        :param path: path to parquet preprocessed sequence
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Parquet file not found")
        try:
            pq.read_metadata(path)

            return cls(path)

        except Exception as e:
            raise Exception("Failed reading parquet with exception:\n", e)

    @property
    def cytosine_file_schema(self):
        return pa.schema([
            ("position", pa.int32()),
            ("context", pa.dictionary(pa.int8(), pa.utf8())),
            ("chr", pa.dictionary(pa.int8(), pa.utf8())),
            ("strand", pa.bool_())])

    def __infer_schema_table(self, handle: io.TextIOBase):
        """
        Initialize dummy table with dictionary columns encoded
        :param handle: fasta input stream
        :return: dummy table
        """
        handle.seek(0)
        schema = self.cytosine_file_schema

        # get all chromosome names to categorise them
        chrom_ids = [record.id for record in seqio.parse(handle, format="fasta")]
        # longer list for length be allways greater than chroms or contexts
        chrom_ids += [str(chrom_ids[0])] * 3

        # init other rows of table
        contexts = ["CG", "CHG", "CHH"]
        contexts += [str(contexts[0])] * (len(chrom_ids) - len(contexts))
        positions = [-1] * len(chrom_ids)
        strand = [True] * len(chrom_ids)

        schema_table = pa.Table.from_arrays(
            arrays=[positions, contexts, chrom_ids, strand],
            schema=schema
        )

        return schema_table

    def __read_fasta(self, handle):
        # init arrow parquet writer
        arrow_writer = pq.ParquetWriter(self.cytosine_file, self.cytosine_file_schema)
        # prepare dummy table with all dictionary columns already mapped
        print("Scanning file to get chromosome ids.")
        schema_table = self.__infer_schema_table(handle)

        print("Extracting cytosine contexts.")
        print("Writing into", self.cytosine_file)
        # return to start byte
        handle.seek(0)
        for record in seqio.parse(handle, "fasta"):
            # POSITIVE
            # parse sequence and get all trinucleotide positions and contexts
            positions, trinuc = get_trinuc(str(record.seq))

            # convert into arrow  table
            arrow_table = pa.Table.from_arrays(
                arrays=[positions, trinuc, [record.id for _ in positions], [True for _ in positions]],
                schema=self.cytosine_file_schema
            )
            # unify dictionary keys with dummy table
            # and deselect dummy rows
            arrow_table = pa.concat_tables([schema_table, arrow_table]).unify_dictionaries()[len(schema_table):]
            # write to file
            arrow_writer.write(arrow_table)

            print(f"Read chromosome: {record.id}\t+", end="\r")

            # NEGATIVE
            positions, trinuc = get_trinuc(str(record.seq), reverse=True)

            arrow_table = pa.Table.from_arrays(
                arrays=[positions, trinuc, [record.id] * len(positions), [False for _ in positions]],
                schema=schema_table.schema
            )

            arrow_table = pa.concat_tables([schema_table, arrow_table]).unify_dictionaries()[len(schema_table):]
            arrow_writer.write(arrow_table)

            print(f"Read chromosome: {record.id}\t+-")

        print("Done reading fasta sequence.\n")
        arrow_writer.close()

    def __read_fasta_wrapper(self, fasta_path: str | Path) -> tempfile.TemporaryFile:
        fasta_path = Path(fasta_path)

        print("Reading sequence from:", fasta_path)
        if fasta_path.suffix == ".gz":
            with gzip.open(fasta_path.absolute(), 'rt') as handle:
                return self.__read_fasta(handle)
        else:
            with open(fasta_path.absolute()) as handle:
                return self.__read_fasta(handle)

    def get_metadata(self):
        return pq.read_metadata(self.cytosine_file)


class Mapper:
    def __init__(self, path):
        self.report_file = path

    @staticmethod
    def __map_with_sequence(df_lazy, sequence_df) -> pl.DataFrame:
        file_types = [
            pl.col("chr").cast(pl.Categorical),
            pl.col("position").cast(pl.Int32)
        ]

        # arrow table aligned to genome
        chrom_aligned = (
            df_lazy
            .with_columns(file_types)
            .set_sorted("position")
            .join(sequence_df.lazy(), on=["chr", "position"])
            .collect()
        )

        return chrom_aligned

    @staticmethod
    def __read_filter_sequence(sequence: Sequence, filter: list) -> pa.Table:
        table = pq.read_table(sequence.cytosine_file, filters=filter)

        modified_schema = table.schema
        modified_schema = modified_schema.set(1, pa.field("context", pa.utf8()))
        modified_schema = modified_schema.set(2, pa.field("chr", pa.utf8()))

        return table.cast(modified_schema)

    @staticmethod
    def __bedGraph_reader(path, batch_size, cpu, skip_rows):
        return pl.read_csv_batched(
            path,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'count_m'],
            columns=[0, 2, 3],
            batch_size=batch_size,
            n_threads=cpu,
            skip_rows=skip_rows,
            dtypes=[pl.Utf8, pl.Int64, pl.Float32]
        )

    @staticmethod
    def __coverage_reader(path, batch_size, cpu, skip_rows):
        return pl.read_csv_batched(
            path,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'count_m', 'count_um'],
            columns=[0, 2, 4, 5],
            batch_size=batch_size,
            n_threads=cpu,
            skip_rows=skip_rows,
            dtypes=[pl.Utf8, pl.Int64, pl.Int32, pl.Int32]
        )

    @classmethod
    def __map(cls, where, sequence, batched_reader, mutations: list[pl.Expr] = None):
        pl.enable_string_cache()
        genome_metadata = sequence.get_metadata()
        genome_rows_read = 0

        pq_writer = None

        for df in batched_reader:
            batch = pl.from_arrow(df)

            # get batch stats
            batch_stats = batch.group_by("chr").agg([
                pl.col("position").max().alias("max"),
                pl.col("position").min().alias("min")
            ])

            for chrom in batch_stats["chr"]:
                chrom_min, chrom_max = [batch_stats.filter(pl.col("chr") == chrom)[stat][0] for stat in
                                        ["min", "max"]]

                filters = [
                    ("chr", "=", chrom),
                    ("position", ">=", chrom_min),
                    ("position", "<=", chrom_max)
                ]

                chrom_genome = (
                    pl.from_arrow(cls.__read_filter_sequence(sequence, filters))
                    .with_columns([
                        pl.col("context").cast(pl.Categorical),
                        pl.col("chr").cast(pl.Categorical),
                        pl.when(pl.col("strand") == True).then(pl.lit("+"))
                        .otherwise(pl.lit("-"))
                        .cast(pl.Categorical)
                        .alias("strand")
                    ])
                )

                # arrow table aligned to genome
                filtered_lazy = batch.lazy().filter(pl.col("chr") == chrom)
                filtered_aligned = cls.__map_with_sequence(filtered_lazy, chrom_genome)
                print(len(chrom_genome))
                if mutations is not None:
                    filtered_aligned = filtered_aligned.with_columns(mutations)

                missing_cols = set(["chr", "position", "strand", "context", "count_m", "count_total"]) - set(filtered_aligned.columns)

                for column in missing_cols:
                    filtered_aligned.with_columns(pl.lit(None).alias(column))

                filtered_aligned = filtered_aligned.select(["chr", "position", "strand", "context", "count_m", "count_total"])

                filtered_aligned = filtered_aligned.to_arrow()
                if pq_writer is None:
                    pq_writer = pa.parquet.ParquetWriter(
                        where,
                        schema=filtered_aligned.schema
                    )

                pq_writer.write(filtered_aligned)
                genome_rows_read += len(chrom_genome)

                print("Mapped {rows_read}/{rows_total} ({percent}%) cytosines".format(
                    rows_read=genome_rows_read,
                    rows_total=genome_metadata.num_rows,
                    percent=round(genome_rows_read / genome_metadata.num_rows * 100, 2)
                ), end="\r")

        gc.collect()

        pq_writer.close()

    @staticmethod
    def __check_compressed(path: str | Path, temp_dir=None):
        path = Path(path)

        if path.suffix == ".gz":
            temp_file = tempfile.NamedTemporaryFile(dir=temp_dir)
            print(f"Temporarily unpack {path} to {temp_file.name}")

            with gzip.open(path, mode="rb") as file:
                shutil.copyfileobj(file, temp_file)

            return temp_file

        else:
            return path

    @classmethod
    def bedGraph(
            cls,
            path,
            sequence: Sequence,
            temp_dir: str = "./",
            name: str = None,
            delete: bool = True,
            block_size_mb: int = 30,
            use_threads: bool = True
    ):
        """
        :param path: path to .bedGraph file
        :param sequence: initialized Sequence object
        :param temp_dir: directory for temporary files
        :param name: temporary file basename
        :param delete: save or delete temporary file
        :param block_size_mb: Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        :param use_threads: Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        """
        path = Path(path).expanduser().absolute()
        if not path.exists():
            raise FileNotFoundError()

        file = cls.__check_compressed(path, temp_dir)
        path = file.name

        report_file = init_tempfile(temp_dir, name, delete, suffix=".bedGraph.parquet")
        mapper = Mapper(report_file)

        path = Path(path)
        print(f"Started reading bedGraph file from {path}")

        bedGraph_reader = CsvReader(file.name, BedGraphOptions(use_threads, block_size_mb * 1024**2))

        mutations = [
            pl.col("count_m") / 100,
            pl.lit(1).alias("count_total")
        ]

        cls.__map(mapper.report_file, sequence, bedGraph_reader, mutations)

        print(f"\nDone reading bedGraph sequence\nTable saved to {mapper.report_file}")

        return mapper

    @classmethod
    def coverage(
            cls,
            path,
            sequence: Sequence,
            temp_dir: str = "./",
            name: str = None,
            delete: bool = True,
            block_size_mb: int = 30,
            use_threads: bool = True
    ):
        """
        :param path: path to .cov file
        :param sequence: initialized Sequence object
        :param temp_dir: directory for temporary files
        :param name: temporary file basename
        :param delete: save or delete temporary file
        :param block_size_mb: Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        :param use_threads: Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        """
        path = Path(path).expanduser().absolute()
        if not path.exists():
            raise FileNotFoundError()

        file = cls.__check_compressed(path, temp_dir)
        path = file.name

        report_file = init_tempfile(temp_dir, name, delete, suffix=".cov.parquet")
        mapper = Mapper(report_file)

        path = Path(path)
        print(f"Started reading coverage file from {path}")

        coverage_reader = CsvReader(report_file, CoverageOptions(use_threads, block_size_mb * 1024**2))

        mutations = [
            (pl.col("count_m") + pl.col("count_um")).alias("count_total")
        ]

        cls.__map(mapper.report_file, sequence, coverage_reader, mutations)

        print(f"\nDone reading coverage file\nTable saved to {mapper.report_file}")

        return mapper
