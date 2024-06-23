import datetime
import itertools
import queue
import uuid
from collections import namedtuple, defaultdict, Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import ceil
from multiprocessing import Pool
from pathlib import Path
import time
from threading import Thread
from typing import Any

import matplotlib.pyplot as plt
import numba
import numpy as np
import pysam
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.ipc as paipc
import polars as pl
from numba import njit
from progress.bar import Bar

from src.bsxplorer import Genome, Config
from src.bsxplorer.SeqMapper import Sequence
from src.bsxplorer.UniversalReader_batches import FullSchemaBatch

nuc_row = namedtuple(typename="nuc_row", field_names=["position", "context", "m", "qual", "id", "strand", "converted", "chr"])


class BAMOptions:
    def __init__(self, bamtype: "bismark"):
        self._bamtype = bamtype

    @property
    def strand_conv(self):
        if self._bamtype == "bismark":
            return [3, 4]

    @property
    def strand_dict(self):
        if self._bamtype == "bismark":
            return {
                "CTCT": True,
                "CTGA": False,
                "GACT": True,
                "GAGA": False
            }

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
    def conv_dict(self):
        if self._bamtype == "bismark":
            return {
                "CTCT": False,
                "CTGA": False,
                "GACT": True,
                "GAGA": True
            }

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

    @property
    def calls_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "z": dict(context="CG",  strand=False),
                "Z": dict(context="CG",  strand=True),
                "x": dict(context="CHG", strand=False),
                "X": dict(context="CHG", strand=True),
                "h": dict(context="CHH", strand=False),
                "H": dict(context="CHH", strand=True),
                "u": dict(context="U",   strand=False),
                "U": dict(context="U",   strand=True)
            }

    @property
    def context_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "z": "CG",
                "Z": "CG",
                "x": "CHG",
                "X": "CHG",
                "h": "CHH",
                "H": "CHH",
                "u": "U",
                "U": "U"
            }

    @property
    def m_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "z": False,
                "Z": True,
                "x": False,
                "X": True,
                "h": False,
                "H": True,
                "u": False,
                "U": True
            }

    @property
    def orientation_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "CT": True,
                "GA": False
            }

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


@dataclass(order=True)
class AlignmentResult:
    data: pl.DataFrame = field(compare=False)

    @property
    def is_empty(self):
        return self.data is None


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



class BAMThread(Thread):
    def __init__(self, queue: queue.Queue, results_arr, options: BAMOptions, min_qual, keep_converted=False):
        Thread.__init__(self)
        self.alignments_queue = queue
        self.results_arr = results_arr
        self.options = options
        self.finished = False
        self._min_qual = min_qual
        self._keep_converted = keep_converted

    def finish(self):
        self.finished = True

    def run(self):
        while not self.finished:
            chrom, start, end, alignments = self.alignments_queue.get()

            if alignments:
                # Parse alignments
                parsed = [self.parse_alignment(alignment) for alignment in alignments if alignment.alen != 0]
                # Convert to polars
                # Explode ref_positions, calls and quals
                # Add absolute position column
                # Filter by read regions bounds
                # Map strand_conv values and drop this column
                parsed_df = (
                    pl.DataFrame(parsed, schema=self.parsed_schema_pl).lazy()
                    .explode(["ref_position", "call", "qual"])
                    .with_columns((pl.col("ref_position") + pl.col("ref_start") + 1).alias("position"))
                    .filter((pl.col("position") > start) & (pl.col("position") < end))
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

                parsed_df = parsed_df.collect()

                # Count methylation
                groupped = (
                    parsed_df
                    .sort("qual", descending=True)
                    .unique(["qname", "position"], keep="first")
                    # .group_by("position")
                    # .agg([
                    #     pl.sum("m").alias("count_m"),
                    #     pl.count("m").alias("count_total"),
                    #     pl.first('context'),
                    #     pl.first('strand')
                    # ])
                )
            else:
                groupped = None

            self.results_arr.append(AlignmentResult(data=groupped))
            self.alignments_queue.task_done()

    parsed_schema_pl = dict(
        ref_position=pl.List(pl.UInt32),
        call=pl.List(pl.String),
        qual=pl.List(pl.UInt8),
        ref_start=pl.UInt32,
        qname=pl.String,
        strand_conv=pl.String
    )

    def parse_alignment(self, alignment):
        if alignment.alen == 0:
            return None

        calls_string = alignment.tags[2][1]
        strand_conv = ""
        for tag_idx in self.options.strand_conv:
            strand_conv += alignment.tags[tag_idx][1]

        # parsed: tuple (      0     ,      1   ,     2   ,       3        ,     4     ,      5     )
        # parsed: tuple (read_pos_arr, calls_arr, qual_arr, reference_start, query_name, strand_conv)
        parsed = parse_calls(calls_string, alignment.query_qualities, alignment.qlen) + (alignment.reference_start, alignment.query_name, strand_conv)

        return parsed


def check_path(path: str | Path):
    path = Path(path).expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError
    return path



class BAMReportReaderThreaded:
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
            keep_converted: bool = True,
            qc: bool = True,
            gui: bool = True,
            temp_dir: Path = Path.cwd(),
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
        self._threads_num = threads
        self._keep_converted = keep_converted

        self._mode = "report"

        self._memory_pool = pa.system_memory_pool()

        self.qc = qc
        self.gui = gui
        if qc:
            self.quals_count = Counter()
            self.pos_count = []
            self.reg_qual = []

    def report_iter(self):
        self._mode = "report"
        return self.__iter__()

    def __iter__(self):
        # Init threads
        self.alignments_queue = queue.Queue(maxsize=self._threads_num)
        self._results_arr = []

        self.threads = []
        for _ in range(self._threads_num):
            new_thread = BAMThread(
                self.alignments_queue,
                self._results_arr,
                self._options,
                self._min_qual,
                self._keep_converted,
            )
            new_thread.daemon = True
            new_thread.start()
            self.threads.append(new_thread)

        # Init tasks for threads
        chr_step = sum(self.bamfile.lengths) // (self.bamfile.mapped // self._batch_num) + 1
        self._tasks = [
            (chrom, start + 1, start + chr_step if start + chr_step < length else length, batch_idx)
            for chrom, length in zip(self.bamfile.references, self.bamfile.lengths)
            for start, batch_idx in zip(range(0, length, chr_step), itertools.count())
        ]
        self._tasks_iterator = iter(self._tasks)

        # Bar
        self._bar = BAMBar(
            message=f"{self.bamfile.references[0]} ({1}/{len(self.bamfile.references)})",
            max=self.bamfile.lengths[0],
            reads_total=self.bamfile.mapped)
        self._bar.start()

        # GUI
        if self.gui:
            self._setup_gui()
        return self

    def _setup_gui(self):
        self._fig = plt.figure(constrained_layout=True)
        gs = self._fig.add_gridspec(2, 2)

        self._chr_ax = self._fig.add_subplot(gs[0, :])
        self._chr_ax.set_title("Avg. reads quality")
        self._chr_ax.set_xlabel("Position")
        self._chr_ax.set_ylabel("Quality")

        x_ticks = list(itertools.accumulate(self.bamfile.lengths))
        self._chr_ax.set_xticks(x_ticks, labels=self.bamfile.references, rotation=-90, fontsize=8)

        self._qual_ax = self._fig.add_subplot(gs[1, 0])
        self._qual_ax.set_title("Hist. of map qualities")
        self._qual_ax.set_xlabel("Quality")
        self._qual_ax.set_ylabel("Density")

        self._pos_ax = self._fig.add_subplot(gs[1, 1])
        self._pos_ax.set_title("Read quality by position")
        self._pos_ax.set_xlabel("Read position")
        self._pos_ax.set_ylabel("Quality")
        # plt.show()

        plt.ion()

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


    def _split_alignments(self, start, end, alignments):
        partitions = []

        partition_step = (end - start) // self._threads_num + 1
        for i in range(self.alignments_queue.maxsize):
            batch_start = start + partition_step * i if i != 0 else 0
            batch_end = start + partition_step * (i + 1) if i != self.alignments_queue.maxsize else max(
                map(lambda alignment: alignment.reference_end, alignments)) + 1

            # Filter alignments
            partitions.append(
                list(
                    filter(
                        lambda alignment: (batch_start <= alignment.reference_start) and (alignment.reference_end <= batch_end)
                                          or (alignment.reference_start < batch_end < alignment.reference_end),
                        alignments
            )))
        return partitions

    def _group_counts(self, chrom, start, end, data):
        if self.sequence_ds is not None:
            sequence_df = pl.from_arrow(self.sequence_ds.filter((pc.field("chr") == chrom) & (pc.field("position") > start) & (pc.field("position") < end)).to_table())
        else:
            sequence_df = None

        new_batch = (
            pl.concat(data)
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
                sequence_df
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

    def _qc_stats(self, quals, chr, end):
        # Gather batch_stats
        quals_count = Counter(itertools.chain(*quals))

        pos_count = [Counter() for _ in range(max(map(lambda seq: len(seq), quals)))]
        for qual_seq, counter in zip(itertools.zip_longest(*quals, fillvalue=-1), pos_count):
            counter += Counter(list(qual_seq))

        self.quals_count += quals_count

        new_pos_count = []
        for new, total in itertools.zip_longest(pos_count, self.pos_count, fillvalue=Counter()):
            new_pos_count.append(total + new)

        quals_ord = sorted(iter((ord(qual) - 33, count) for qual, count in quals_count.items() if qual != -1), key=lambda item: item[0])
        avg_region_qual = sum(qual * count for qual, count in quals_ord) / quals_count.total()
        self.reg_qual.append((chr, end, avg_region_qual))

        self.pos_count = new_pos_count
        return quals_count, pos_count


    def _upd_bar(self, chrom, start, reads):
        self._bar.message = f"{chrom} ({self.bamfile.references.index(chrom) + 1}/{len(self.bamfile.references)})"
        self._bar.index = start
        self._bar.reads += reads
        self._bar.update()

    def _mutate(self, batch: pl.DataFrame):
        if self._mode == "report":
            if "trinuc" not in batch.columns:
                batch = batch.with_columns(pl.col("context").alias("trinuc"))

            batch = batch.with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))

            return FullSchemaBatch(batch, raw=None)

    def _finish(self):
        for thread in self.threads:
            thread.finish()
        self._bar.goto(self._bar.max)
        self._bar.finish()
        self._bar.update()


    def __next__(self):
        try:
            chrom, start, end, batch_idx = next(self._tasks_iterator)
            alignments = list(self.bamfile.fetch(contig=chrom, start=start, end=end))

            for partition in self._split_alignments(start, end, alignments):
                self.alignments_queue.put((chrom, start, end, partition))

            if self.qc:
                quals = list(map(lambda alignment: alignment.qqual, alignments))
                if quals:
                    qual_stat, pos_stat = self._qc_stats(quals, chrom, end)

            if self.gui:
                self._upd_gui(qual_stat, pos_stat)

            self.alignments_queue.join()

            data = [result.data for result in self._results_arr if not result.is_empty]
            if data:
                batch = self._group_counts(chrom, start, end, data)

                self._results_arr.clear()
                final = self._mutate(batch)

                self._upd_bar(chrom, start, len(alignments))

                return final
            else:
                return self.__next__()

        except StopIteration:
            self._finish()
            raise StopIteration

# genome = Genome.from_gff("~/Documents/CX_reports/flax/Linum/genomic.gff").gene_body(0)
sequence = Sequence.from_preprocessed("/Users/shitohana/Desktop/PycharmProjects/BSXplorer/tests/Linum_sequence.parquet")
reader = BAMReportReaderThreaded(
    "/Users/shitohana/Documents/CX_reports/flax/Linum/MAtF3-1.sorted.bam",
    index_filename="/Users/shitohana/Documents/CX_reports/flax/Linum/MAtF3-1.sorted.bam.bai",
    sequence=sequence,
    threads=8,
    batch_num=10000,
    qc=False,
    gui=False
)


start_time = time.time()
for _ in reader.report_iter():
    pass
print("\n\n", time.time() - start_time)