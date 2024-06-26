import dataclasses
import datetime
import itertools
import queue
import time
from collections import Counter, UserDict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from threading import Thread
from typing import Any, Literal

import numba as nb
import numpy as np

import polars as pl
import pyarrow as pa
import pysam
from matplotlib import pyplot as plt
from numba import jit, prange
from progress.bar import Bar
from pyarrow import dataset as pads, compute as pc

from src.bsxplorer.SeqMapper import Sequence
from src.bsxplorer.UniversalReader_batches import FullSchemaBatch


def check_path(path: str | Path):
    path = Path(path).expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError
    return path


class QualsCounter(UserDict):
    def __init__(self, data=None):
        UserDict.__init__(self, data if data is not None else dict())

    def __add__(self, other: dict):
        for key in other.keys():
            if key in self.data.keys():
                self.data[key] += other[key]
            else:
                self.data[key] = other[key]
        return self

    def total(self):
        return sum(self.values())

    def weighted_sum(self):
        return sum(key * value for key, value in self.data.items())

    def mean_qual(self):
        return self.weighted_sum() / self.total()


@dataclass
class ReadTask:
    chrom: str
    start: int
    end: int
    filters: list[dict]
    context: str


@dataclass
class AlignmentResult:
    data: Any
    chrom: str
    start: int
    end: int
    alignments: int

    @property
    def is_empty(self):
        return self.data is None


@dataclass
class QCResult:
    quals_count: QualsCounter
    pos_count: list[QualsCounter]
    chrom: str
    start: int
    end: int

    @property
    def is_empty(self):
        return self.quals_count is None


class BAMOptions:
    def __init__(self, bamtype: Literal["bismark"]):
        self._bamtype = bamtype

    @property
    def strand_conv(self):
        if self._bamtype == "bismark":
            return [3, 4]

    @property
    def strand_conv_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ["CTCT", "CTGA", "GACT", "GAGA"],
                [True, False, True, False],
                [True, True, False, False]
            ],
            schema=dict(strand_conv=pl.String, strand=pl.Boolean, converted=pl.Boolean)
        )

    @property
    def calls_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ['z', 'Z', 'x', 'X', 'h', 'H', 'u', 'U'],
                ['CG', 'CG', 'CHG', 'CHG', 'CHH', 'CHH', 'U', 'U'],
                [False, True, False, True, False, True, False, True]
            ],
            schema=dict(call=pl.String, context=pl.String, m=pl.Boolean)
        )


class BAMBar(Bar):
    def __init__(self, reads_total: int = 0, **kwargs):
        self.reads = 0
        self.reads_total = reads_total
        super().__init__(**kwargs)


    suffix = "%(index_kb)d/%(max_kb)d kb (File reading time %(elapsed_fmt)s, ETA: %(eta_fmt)s)"
    fill = "@"

    @property
    def index_kb(self):
        return round(self.index // 1000)

    @property
    def max_kb(self):
        return round(self.max // 1000)

    @property
    def elapsed_fmt(self):
        return str(datetime.timedelta(seconds=int(self.elapsed)))

    @property
    def eta_fmt(self):
        return str(datetime.timedelta(seconds=self.eta))

    @property
    def eta(self):
        if self.reads:
            return int(ceil((self.elapsed / self.reads) * (self.reads_total - self.reads)))
        else:
            return 0

    @property
    def elapsed(self):
        return time.monotonic() - self.start_ts


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 2, "C"), nb.types.Array(nb.float64, 1, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_methylation_entropy(matrix, columns, window_length, min_depth):
    positions_matrix = np.zeros((matrix.shape[1] - window_length, window_length), dtype=np.int64)
    ME_array = np.zeros((matrix.shape[1] - window_length), dtype=np.float64)

    # Sliding window loop
    for i in prange(matrix.shape[1] - window_length):
        window = matrix[:, i:(i + window_length)]

        none_idx = np.not_equal(window, -1).sum(axis=1)

        # Filter empty values
        window = window[none_idx == window.shape[1], :]
        # Check minimal depth
        if window.shape[1] < min_depth:
            continue

        # Initialize signaures array
        signatures = np.zeros(window.shape[0], dtype=np.int64)
        # Fill signatures array
        for col in prange(window.shape[1]):
            signatures = np.multiply(signatures, 2)
            signatures += window[:, col]

        # Sort and find unique values
        sort_signatures = np.sort(signatures)
        nonzero_indicies = np.nonzero(sort_signatures - np.roll(sort_signatures, -1))[0]
        counts = nonzero_indicies + 1
        counts_shifted = np.roll(counts, 1)
        counts_shifted[0] = 0

        # Unique counts
        counts = counts - counts_shifted
        total_counts = counts.sum()

        ME_value = abs((1 / window_length) * sum([(count / total_counts) * np.log2(count / total_counts) for count in counts]))

        positions_matrix[i, :] = columns[i:(i + window_length)]
        ME_array[i] = ME_value

    return positions_matrix, ME_array


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 2, "C"), nb.types.Array(nb.float64, 1, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_epipolymorphism(matrix, columns, window_length, min_depth):
    positions_matrix = np.zeros((matrix.shape[1] - window_length, window_length), dtype=np.int64)
    PM_array = np.zeros((matrix.shape[1] - window_length), dtype=np.float64)

    # Sliding window loop
    for i in prange(matrix.shape[1] - window_length):
        window = matrix[:, i:(i + window_length)]

        none_idx = np.not_equal(window, -1).sum(axis=1)

        # Filter empty values
        window = window[none_idx == window.shape[1], :]
        # Check minimal depth
        if window.shape[1] < min_depth:
            continue

        # Initialize signaures array
        signatures = np.zeros(window.shape[0], dtype=np.int64)
        # Fill signatures array
        for col in prange(window.shape[1]):
            signatures = np.multiply(signatures, 2)
            signatures += window[:, col]

        # Sort and find unique values
        sort_signatures = np.sort(signatures)
        nonzero_indicies = np.nonzero(sort_signatures - np.roll(sort_signatures, -1))[0]
        counts = nonzero_indicies + 1
        counts_shifted = np.roll(counts, 1)
        counts_shifted[0] = 0

        # Unique counts
        counts = counts - counts_shifted
        total_counts = counts.sum()

        PM_value = 1 - sum([(count / total_counts) for count in counts])

        positions_matrix[i, :] = columns[i:(i + window_length)]
        PM_array[i] = PM_value

    return positions_matrix, PM_array


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 1, "C"), nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.int64, 2, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_PDR(matrix, columns, min_cyt: int, min_depth: int):
    # Filter low cytosines reads
    low_cyt = np.sum(matrix != -1, axis=1) >= min_cyt
    matrix = matrix[low_cyt, :]

    n_cols = matrix.shape[1]
    position_array = np.empty(n_cols, dtype=np.int64)
    pdr_array = np.empty(n_cols, dtype=np.float64)
    count_matrix = np.empty((n_cols, 2), dtype=np.int64)

    for i in prange(n_cols):
        covering_reads = matrix[:, i] != -1

        # Skip if no reads cover this cytosine
        if covering_reads.sum() < min_depth:
            continue

        window = matrix[covering_reads, :]

        concordant_reads = np.logical_xor((window == 1).sum(axis=1), (window == 0).sum(axis=1))

        total = len(concordant_reads)
        concordant_count = concordant_reads.sum()
        discordant_count = total - concordant_count
        PDR_value = discordant_count / total

        pdr_array[i] = PDR_value
        count_matrix[i] = concordant_count, discordant_count
        position_array[i] = columns[i]

    return position_array, pdr_array, count_matrix


class PivotRegion:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    @property
    def matrix_df(self):
        return self.df.select(pl.all().exclude(["qname", "converted", "strand"]))

    @property
    def _jit_compatitable(self):
        df = self.matrix_df
        matrix = df.to_numpy().astype(np.int8)
        columns = np.array(list(map(int, df.columns)), dtype=np.int64)

        return matrix, columns

    def methylation_entropy(self, window_length: int = 4, min_depth: int = 4):
        """
        Wrapper for _jit_methylation_entropy – JIT compiled ME calculating function

        Parameters
        ----------
        window_length
            Length of the sliding window
        min_depth
            Minimal depth of reads to consider this window for calculation

        Returns
        -------
            Matrix with position of cytosines from window and array with their ME values
        """
        matrix, columns = self._jit_compatitable
        return _jit_methylation_entropy(matrix, columns, window_length, min_depth)

    def epipolymorphism(self, window_length: int = 4, min_depth: int = 4):
        """
        Wrapper for _jit_epipolymorphism – JIT compiled PM calculating function

        Parameters
        ----------
        window_length
            Length of the sliding window
        min_depth
            Minimal depth of reads to consider this window for calculation

        Returns
        -------
            Matrix with position of cytosines from window and array with their ME values
        """
        matrix, columns = self._jit_compatitable
        return _jit_epipolymorphism(matrix, columns, window_length, min_depth)

    def PDR(self, min_cyt: int = 5, min_depth: int = 4):
        matrix, columns = self._jit_compatitable
        return _jit_PDR(matrix, columns, min_cyt, min_depth)


class BAMThread(Thread):
    def __init__(
            self,
            queue: queue.Queue,
            results_queue: queue.Queue,
            options: BAMOptions,
            sequence_ds: pads.Dataset,
            min_qual,
            keep_converted=False,
            mode="report",
            daemon=True,
    ):
        Thread.__init__(self, daemon=daemon)
        self.alignments_queue = queue
        self.results_queue = results_queue
        self.sequence_ds = sequence_ds
        self.options = options
        self._min_qual = min_qual
        self._keep_converted = keep_converted
        self._mode = mode

        self.finished = False

    def finish(self):
        self.finished = True

    def run(self):
        while not self.finished:
            try:
                task, alignments = self.alignments_queue.get()
                chrom, start, end, filters, context = dataclasses.astuple(task)

                if alignments:
                    # Parse alignments
                    # start_time = time.time()
                    parsed = list(filter(lambda parsed: parsed is not None, iter(self.parse_alignment(alignment) for alignment in alignments)))
                    # parsed = [
                    #     tuple(zip(*[(i, calls[i], quals[i]) for i in range(len(calls)) if calls[i] != "."])) +
                    #     (ref_start, qname, ''.join(tags[tag_idx][1] for tag_idx in self.options.strand_conv))
                    #     for calls, quals, alen, ref_start, qname, tags in
                    #     map(lambda alignment: (
                    #         alignment.tags[2][1],
                    #         alignment.query_alignment_qualities,
                    #         alignment.alen,
                    #         alignment.reference_start,
                    #         alignment.query_name,
                    #         alignment.tags), alignments)
                    #     if alen != 0]
                    # print(time.time() - start_time)
                    data = self.calls_to_df(parsed, start, end)
                    if context != "all":
                        data = data.filter(context=context)
                    data.collect()

                    filtered_regions = [data.filter(self.filters_to_expr(f)) for f in filters] if filters else data

                    if self._mode == "report":
                        merged = pl.concat(filtered_regions).lazy() if filters else data
                        if self.sequence_ds is not None:
                            merged = self.group_counts(chrom, start, end, context, merged.lazy())
                        final = self.mutate(merged)
                    else:
                        final = [self.to_pivot(region) for region in filtered_regions]
                else:
                    final = None

                self.results_queue.put(AlignmentResult(final, chrom, start, end, len(alignments)))
                self.alignments_queue.task_done()

            except Exception as e:
                print("Got exception in Report Thread")
                print(e)

    parsed_schema_pl = dict(
        ref_position=pl.List(pl.UInt32),
        call=pl.List(pl.String),
        qual=pl.List(pl.UInt8),
        ref_start=pl.UInt32,
        qname=pl.String,
        strand_conv=pl.String
    )

    @staticmethod
    def to_pivot(df):
        if len(df["strand"].unique()) > 1:
            indexes = ["qname", "converted", "strand"]
        else:
            indexes = ["qname", "converted"]

        pivoted = (
            df
            .sort("position")
            .cast(dict(m=pl.Int8))
            .pivot(index=indexes, columns="position", values="m")
            .fill_null(-1)
        )

        return PivotRegion(pivoted)

    @staticmethod
    def filters_to_expr(f):
        # Not filtering by chrome because assert all regions are on the same chrom
        expr = (pl.col("position") >= f["start"]) & (pl.col("position") <= f["end"])
        if f["strand"] in ["+", "-"]:
            expr = expr & (pl.col("strand") == (True if f["strand"] == "+" else False))
        return expr

    def calls_to_df(self, parsed: list[tuple], start, end) -> pl.LazyFrame:
        # Convert to polars
        # Explode ref_positions, calls and quals
        # Add absolute position column
        # Filter by read regions bounds
        # Map strand_conv values and drop this column
        pass
        parsed_df = (
            pl.LazyFrame(parsed, schema=self.parsed_schema_pl)
            .explode(["ref_position", "call", "qual"])
            .with_columns((pl.col("ref_position") + pl.col("ref_start") + 1).alias("position"))
            .filter((pl.col("position") >= start) & (pl.col("position") <= end))
            .join(self.options.strand_conv_df.lazy(), on="strand_conv", how="left")
            .drop("strand_conv")
        )

        # If specified, filter by converted
        if not self._keep_converted:
            parsed_df = parsed_df.filter(converted=False)
        # If specified, filter by converted
        if self._min_qual is not None:
            parsed_df = parsed_df.filter(pl.col("qual") > self._min_qual)

        # Convert calls values to context and methylation status
        parsed_df = (
            parsed_df
            .join(self.options.calls_df.lazy(), on="call", how="left")
            .drop("call")
            .filter(pl.col('context') != "U")
        )

        parsed_df = parsed_df

        # Count methylation
        data = (
            parsed_df
            .sort("qual", descending=True)
            .unique(["qname", "position"], keep="first")
        )

        return data

    def parse_alignment(self, alignment):
        if alignment.alen == 0:
            return None

        calls_string = alignment.tags[2][1]
        strand_conv = ""
        for tag_idx in self.options.strand_conv:
            strand_conv += alignment.tags[tag_idx][1]

        # parsed: tuple (      0     ,      1   ,     2   ,       3        ,     4     ,      5     )
        # parsed: tuple (read_pos_arr, calls_arr, qual_arr, reference_start, query_name, strand_conv)
        parsed = self.parse_calls(calls_string, alignment.query_alignment_qualities, alignment.qlen) + (alignment.reference_start, alignment.query_name, strand_conv)

        return parsed

    @staticmethod
    def parse_calls(calls_string: str, qual_arr: str, qlen: int):
        positions = []
        calls = []
        quals = []
        for read_pos in range(qlen):
            if calls_string[read_pos] == ".":
                continue
            positions.append(read_pos)
            calls.append(calls_string[read_pos])
            quals.append(qual_arr[read_pos])
        return positions, calls, quals

    def mutate(self, batch_lazy: pl.LazyFrame):
        if self._mode == "report":
            if "trinuc" not in batch_lazy.columns:
                batch_lazy = batch_lazy.with_columns(pl.col("context").alias("trinuc"))

            batch = (
                batch_lazy
                .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
                .fill_nan(0)
                .collect()
            )

            return FullSchemaBatch(batch, raw=None)

    def group_counts(self, chrom, start, end, context, data_lazy: pl.LazyFrame):
        if self.sequence_ds is not None:
            sequence_df = pl.from_arrow(self.sequence_ds.filter((pc.field("chr") == chrom) & (pc.field("position") > start) & (pc.field("position") < end)).to_table())
        else:
            sequence_df = None

        if context != "all":
            sequence_df = sequence_df.filter(context=context)

        new_batch = (
            data_lazy
            .group_by("position")
            .agg([
                pl.sum("m").alias("count_m"),
                pl.count("m").alias("count_total"),
                pl.first('context'),
                pl.first('strand')
            ])
        )
        if sequence_df is not None:
            new_batch = (
                sequence_df.lazy()
                .cast(dict(position=new_batch.schema["position"]))
                .sort("position")
                .join(
                    new_batch.select(["position", "count_m", "count_total"]),
                    on="position", how="left")
                .fill_null(0)
            )
        else:
            new_batch = (
                new_batch
                .with_columns(pl.lit(chrom, pl.String).alias("chr"))
                .sort("position")
            )

        return new_batch


class QCThread(Thread):
    def __init__(self, quals_queue: queue.Queue, result_queue: queue.Queue, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.finished = False
        self.quals_queue = quals_queue
        self.result_queue = result_queue

    def run(self):
        while not self.finished:
            try:
                task, alignments = self.quals_queue.get()
                chrom, start, end, filters, context = dataclasses.astuple(task)
                if alignments:
                    quals = [alignment.query_alignment_qualities for alignment in alignments if alignment.alen != 0]
                    quals_count, pos_count = self._qc_stats(quals)
                    self.result_queue.put(
                        QCResult(QualsCounter(dict(quals_count)), [QualsCounter(dict(counter)) for counter in pos_count],
                                 chrom, start, end))
            except Exception as e:
                print("Got exception in QC Thread")
                print(e)


    def finish(self):
        self.finished = True

    @staticmethod
    def _qc_stats(quals):
        # Gather batch_stats
        quals_count = Counter(itertools.chain(*quals))

        pos_count = [Counter() for _ in range(max(map(lambda seq: len(seq), quals)))]
        for qual_seq, counter in zip(itertools.zip_longest(*quals, fillvalue=-1), pos_count):
            counter += Counter(list(qual_seq))
            if -1 in counter.keys():
                del counter[-1]

        return quals_count, pos_count


class BAMReader:
    def __init__(
            self,
            bam_filename: str | Path,
            index_filename: str | Path,
            sequence: Sequence = None,
            bamtype: str = "bismark",
            regions: pl.DataFrame = None,
            threads: int = 1,
            batch_num: int = 1e4,
            min_qual: int = None,
            context: Literal["CG", "CHG", "CHH", "all"] = "all",
            keep_converted: bool = True,
            qc: bool = True,
            readahead=5,
            **pysam_kwargs
    ):
        # Init BAM
        bam_filename = check_path(bam_filename)
        index_filename = check_path(index_filename)
        self.bamfile = pysam.AlignmentFile(filename=str(bam_filename), index_filename=str(index_filename), threads=threads, **pysam_kwargs)

        # Init reference path
        self.sequence_ds = pads.dataset(sequence.cytosine_file) if sequence is not None else None

        # Init inner attributes
        self.regions = regions

        self._batch_num = int(batch_num) * threads
        self._min_qual = min_qual
        self._options = BAMOptions(bamtype)
        self._keep_converted = keep_converted
        self._readahead = readahead

        self._mode = "report"
        self._context = context

        self._memory_pool = pa.system_memory_pool()

        self.qc = qc
        if qc:
            self.quals_count = QualsCounter()
            self.pos_count = []
            self.reg_qual = []

    def report_iter(self):
        self._mode = "report"
        return self._iter()

    def stats_iter(self):
        self._mode = "stats"
        return self._iter()

    def _setup_threads(self):
        self.alignments_queue = queue.Queue(maxsize=self._readahead)
        self.results_queue = queue.Queue()
        self._result_iterator = iter([])

        self.thread = BAMThread(
            self.alignments_queue,
            self.results_queue,
            self._options,
            self.sequence_ds,
            self._min_qual,
            self._keep_converted,
            self._mode,
            daemon=True
        )
        self.thread.start()

        if self.qc:
            self.qc_queue = queue.Queue(maxsize=self._readahead)
            self.qc_results_queue = queue.Queue()

            self.qc_thread = QCThread(self.qc_queue, self.qc_results_queue, daemon=True)
            self.qc_thread.start()

    def _setup_tasks(self):
        # Init tasks for threads
        # Task: tuple (chrom, start, end)
        chr_step = sum(self.bamfile.lengths) // (self.bamfile.mapped // self._batch_num) + 1
        tasks = []

        for chrom, length in zip(self.bamfile.references, self.bamfile.lengths):
            for start in range(0, length, chr_step):
                end = start + chr_step if start + chr_step < length else length

                if self.regions is None:
                    filters = []
                else:
                    filters = self.regions.filter(
                        (pl.col("chr") == chrom) &
                        (pl.col("upstream") >= start) &
                        (pl.col("downstream") <= end)
                    ).select([
                        "chr",
                        pl.col("upstream").alias("start"),
                        pl.col("downstream").alias("end"),
                        "strand"
                    ])
                    if filters.is_empty():
                        continue
                    else:
                        filters = filters.to_dicts()

                tasks.append(ReadTask(chrom, start, end, filters, self._context))

        self._tasks_iterator = iter(tasks)

    def _iter(self):
        # Init threads
        self._setup_threads()
        self._setup_tasks()

        # Bar
        self._bar = BAMBar(
            message=f"{self.bamfile.references[0]} ({1}/{len(self.bamfile.references)})",
            max=self.bamfile.lengths[0],
            reads_total=self.bamfile.mapped)
        self._bar.start()

        tasks_ended = False
        while True:
            if tasks_ended and self.results_queue.qsize() == 0:
                self._finish()
                break

            try:
                while not self.alignments_queue.full():
                    task: ReadTask = next(self._tasks_iterator)
                    alignments = list(self.bamfile.fetch(contig=task.chrom, start=task.start, end=task.end))
                    self.alignments_queue.put((task, alignments))

                    if self.qc:
                        self.qc_queue.put((task, alignments))

            except StopIteration:
                tasks_ended = True

            alignment_result: AlignmentResult = self.results_queue.get()
            self._upd_bar(alignment_result.chrom, alignment_result.end, alignment_result.alignments)

            if isinstance(alignment_result.data, list):
                for result in alignment_result.data:
                    yield result
            else:
                yield alignment_result.data

    def qc_data(self) -> tuple[QualsCounter, list[QualsCounter], list[tuple]]:
        quals_stat = QualsCounter()
        pos_stat = []
        reg_stat = []

        while not self.qc_results_queue.all_tasks_done:
            time.sleep(1e-5)

        for qc_result in list(self.qc_results_queue.queue).copy():
            assert isinstance(qc_result, QCResult)
            quals_stat += qc_result.quals_count

            for index, pos_count in enumerate(qc_result.pos_count):
                if index > len(pos_stat) - 1:
                    pos_stat.append(QualsCounter())
                pos_stat[index] += pos_count

            reg_avg = qc_result.quals_count.mean_qual()
            reg_stat.append((qc_result.chrom, qc_result.end, reg_avg))

        return quals_stat, pos_stat, reg_stat


    def plot_qc(self):
        quals_stat, pos_stat, reg_stat = self.qc_data()

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        # Chr
        chr_ax = fig.add_subplot(gs[0, :])
        chr_ax.set_title("Avg. reads quality")
        chr_ax.set_xlabel("Position")
        chr_ax.set_ylabel("Quality")

        chr_ticks = list(itertools.accumulate(self.bamfile.lengths))

        x_data, y_data = list(zip(*[(chr_ticks[self.bamfile.references.index(chrom)] + end, qual) for chrom, end, qual in reg_stat]))
        chr_ax.plot(x_data, y_data, 'b')

        chr_ax.set_xticks(chr_ticks, labels=self.bamfile.references, rotation=-90, fontsize=8)

        # Quals
        qual_ax = fig.add_subplot(gs[1, 0])
        qual_ax.set_title("Hist. of map qualities")
        qual_ax.set_xlabel("Quality")
        qual_ax.set_ylabel("Density")

        qual_ax.hist(
            x=list(quals_stat.keys()),
            weights=list(quals_stat.values()),
            color='b',
            linewidth=1,
            edgecolor="white",
        )

        # Pos
        pos_ax = fig.add_subplot(gs[1, 1])
        pos_ax.set_title("Read quality by position")
        pos_ax.set_xlabel("Read position")
        pos_ax.set_ylabel("Quality")

        y_data = [s.mean_qual() for s in pos_stat]
        pos_ax.plot(y_data, 'b')

        return fig




    def _upd_gui(self, qual_stat, pos_stat):
        # Hist

        # Quals
        quals = sorted(iter((ord(qual) - 33, count) for qual, count in qual_stat.items() if qual != -1), key=lambda item: item[0])
        x_data, y_data = list(zip(*quals))
        self._qual_ax.hist(x_data, weights=y_data, color='b', linewidth=0.5, edgecolor="white", density=True)

        # Pos
        y_data = [sum((ord(qual) - 33) * count for qual, count in pos_count.items() if qual != -1) / pos_count.total() for pos_count in pos_stat]
        self._pos_ax.clear()
        self._pos_ax.plot(y_data, 'b')
        self._pos_ax.set_title("Read quality by position")

        x_ticks = list(itertools.accumulate(self.bamfile.lengths))
        plot_positions = [(x_ticks[self.bamfile.references.index(chrom)] + end, qual) for chrom, end, qual in self.reg_qual]
        x_data, y_data = list(zip(*plot_positions))
        self._chr_ax.clear()
        self._chr_ax.plot(x_data, y_data, 'b')
        self._chr_ax.set_xticks(x_ticks, labels=self.bamfile.references, rotation=-90, fontsize=8)
        self._chr_ax.set_title("Avg. reads quality")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _upd_qc(self, qc_result: QCResult):
        self.quals_count += qc_result.quals_count

        new_pos_count = []
        for new, total in itertools.zip_longest(qc_result.pos_count, self.pos_count, fillvalue=QualsCounter()):
            new_pos_count.append(total + new)
        self.pos_count = new_pos_count

        avg_region_qual = sum(qual * count for qual, count in qc_result.quals_count.items()) / qc_result.quals_count.total()
        self.reg_qual.append((qc_result.chrom, qc_result.end, avg_region_qual))

    def _upd_bar(self, chrom, end, reads):
        self._bar.message = f"{chrom} ({self.bamfile.references.index(chrom) + 1}/{len(self.bamfile.references)})"
        self._bar.max = self.bamfile.lengths[self.bamfile.references.index(chrom)]
        self._bar.index = end
        self._bar.reads += reads
        self._bar.update()

    def _finish(self):
        self.thread.finish()
        if self.qc:
            self.qc_thread.finish()

        self._bar.goto(self._bar.max)
        self._bar.update()
        self._bar.finish()

    def __iter__(self):
        raise NotImplementedError("BAMReader can't be iterated itself, use BAMReader.report_iter() or BAMReader.stats_iter() instead.")