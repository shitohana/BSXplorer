from __future__ import annotations

import gzip
import re
import shutil
import tempfile
from os.path import getsize
from pathlib import Path
from typing import Literal
from gc import collect

import numpy as np
import pyarrow as pa
from matplotlib.axes import Axes
from pyarrow import parquet as pq
from scipy import stats
import polars as pl
from progress.bar import Bar
import datetime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


def prepare_labels(major_labels: list, minor_labels: list):
    labels = dict(
        up_mid="Upstream",
        body_start="TSS",
        body_mid="Body",
        body_end="c",
        down_mid="Downstream"
    )

    if major_labels and len(major_labels) == 2:
        labels["body_start"], labels["body_end"] = major_labels
    elif major_labels:
        print("Length of major tick labels != 2. Using default.")
    else:
        labels["body_start"], labels["body_end"] = [""] * 2

    if minor_labels and len(minor_labels) == 3:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = minor_labels
    elif minor_labels:
        print("Length of minor tick labels != 3. Using default.")
    else:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = [""] * 3

    return labels


def approx_batch_num(path, batch_size, check_lines=1000):
    size = getsize(path)

    length = 0
    with open(path, "rb") as file:
        for _ in range(check_lines):
            length += len(file.readline())

    return round(np.ceil(size / (length / check_lines * batch_size)))


def hm_flank_lines(axes: Axes, upstream_windows: int, gene_windows: int, downstream_windows: int):
    """
    Add flank lines to the given axis (for line plot)
    """
    x_ticks = []
    x_labels = []
    if upstream_windows > 0:
        x_ticks.append(upstream_windows - .5)
        x_labels.append('TSS')
    if downstream_windows > 0:
        x_ticks.append(gene_windows + downstream_windows - .5)
        x_labels.append('TES')

    if x_ticks and x_labels:
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)
        for tick in x_ticks:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)


def interval(sum_density: list[int], sum_counts: list[int], alpha=.95):
    """
    Evaluate confidence interval for point

    :param sum_density: Sums of methylated counts in fragment
    :param sum_counts: Sums of all read cytosines in fragment
    :param alpha: Probability for confidence band
    """
    sum_density, sum_counts = np.array(sum_density), np.array(sum_counts)
    average = sum_density.sum() / sum_counts.sum()

    normalized = np.divide(sum_density, sum_counts)

    variance = np.average((normalized - average) ** 2, weights=sum_counts)

    n = sum(sum_counts) - 1

    i = stats.t.interval(alpha, df=n, loc=average, scale=np.sqrt(variance / n))

    return {"lower": i[0], "upper": i[1]}


MetageneSchema = dotdict(dict(
    chr=pl.Categorical,
    strand=pl.Categorical,
    position=pl.UInt64,
    gene=pl.Categorical,
    context=pl.Categorical,
    id=pl.Categorical,
    fragment=pl.UInt32,
    sum=pl.Float32,
    count=pl.UInt32,
))


class ReportBar(Bar):
    suffix = "%(progress2mb).2f/%(max2mb)d Mb [Elapsed: %(elapsed_fmt)s | ETA: %(eta_fmt)s]"
    fill = "@"

    @property
    def progress2mb(self):
        return int(self.index) / (1024**2)

    @property
    def max2mb(self):
        return int(self.max) / (1024**2)

    @property
    def elapsed_fmt(self):
        return str(datetime.timedelta(seconds=self.elapsed))

    @property
    def eta_fmt(self):
        return str(datetime.timedelta(seconds=self.eta))


def decompress(path: str | Path):
    if path.suffix == ".gz":
        temp_file = tempfile.NamedTemporaryFile()
        print(f"Temporarily unpack {path} to {temp_file.name}")

        with gzip.open(path, mode="rb") as file:
            shutil.copyfileobj(file, temp_file)

        return temp_file
    else:
        return path


def merge_replicates(
        paths: list[str | Path],
        report_type: Literal["bismark"] = "bismark",
        batch_size: int = 10**6
):
    paths = [Path(path).expanduser().absolute() for path in paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    files = [decompress(path) for path in paths]

    if report_type == "bismark":
        options = dict(
            separator="\t",
            columns=[0,1,2,3,4,5],
            new_columns=["chr", "position", "strand", "count_m", "count_um", "context"],
            dtypes=[pl.Utf8, pl.UInt64, pl.Utf8, pl.UInt32, pl.UInt32, pl.Utf8]
        )

    options |= dict(batch_size=batch_size)

    temp_parquet = tempfile.NamedTemporaryFile()

    pq_schema = pa.schema([
        ("chr", pa.dictionary(pa.int16(), pa.string())),
        ("position", pa.uint64()),
        ("strand", pa.dictionary(pa.int16(), pa.string())),
        ("count_m", pa.uint32()),
        ("count_um", pa.uint32()),
        ("context", pa.dictionary(pa.int16(), pa.string())),
    ])

    pq_writer = pq.ParquetWriter(temp_parquet.name, pq_schema)

    readers = [
        pl.read_csv_batched(
            file if isinstance(file, Path) else file.name,
            **options
        )
        for file in files
    ]

    print("Reading reports")

    batches = [reader.next_batches(1)[0] for reader in readers]
    while sum(map(lambda batch: batch is not None, batches)) > 0:

        batches = list(map(lambda group: group[0], filter(lambda group: group is not None, batches)))

        concat = pl.concat(batches)

        grouped = (
            concat
            .group_by(["chr", "strand", "position", "context"], maintain_order=True)
            .agg([pl.sum("count_m"), pl.sum("count_um")])
            .cast(dict(
                chr=pl.Categorical,
                strand=pl.Categorical,
                position=pl.UInt64,
                count_m=pl.UInt32,
                count_um=pl.UInt32,
                context=pl.Categorical
            ))
            .select(pq_schema.names)
        )

        print(f"Current position: {batches[0].row(0)[0]} {batches[0].row(0)[1]}", end="\r")

        pq_writer.write(
            grouped.to_arrow().cast(target_schema=pq_schema)
        )
        batches = [reader.next_batches(1) for reader in readers]
        collect()

    print("\nDONE")
    return temp_parquet
