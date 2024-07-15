from __future__ import annotations

import datetime
import re
from collections import OrderedDict
from typing import Literal

import polars as pl
import pyarrow as pa
from progress.bar import Bar


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


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


MetageneJoinedSchema = dotdict(dict(
    chr=pl.Categorical,
    strand=pl.Categorical,
    gene=pl.Categorical,
    start=pl.UInt64,
    id=pl.Categorical,
    context=pl.Categorical,
    fragment=pl.UInt32,
    sum=pl.Float32,
    count=pl.UInt32,
))


class ReportBar(Bar):
    suffix = "%(progress2mb)d/%(max2mb)d Mb [%(elapsed_fmt)s | ETA: %(eta_fmt)s]"
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


polars2arrow = {
    pl.Int8: pa.int8(),
    pl.Int16: pa.int16(),
    pl.Int32: pa.int32(),
    pl.Int64: pa.int64(),
    pl.UInt8: pa.uint8(),
    pl.UInt16: pa.uint16(),
    pl.UInt32: pa.uint32(),
    pl.UInt64: pa.uint64(),
    pl.Float32: pa.float32(),
    pl.Float64: pa.float64(),
    pl.Boolean: pa.bool_(),
    pl.Binary: pa.binary(),
    pl.Utf8: pa.utf8(),
}

arrow2polars = {value: key for key, value in polars2arrow.items()}


def polars2arrow_convert(pl_schema: OrderedDict):
    if pl.Categorical in pl_schema.values():
        raise ValueError("You should write schema for Categorical manually")
    pa_schema = pa.schema([
        (key, polars2arrow[value]) for key, value in pl_schema.items()
    ])
    return pa_schema


def arrow2polars_convert(pa_schema: pa.Schema):
    pl_schema = OrderedDict()

    if not all(pa_type in arrow2polars.keys() for pa_type in pa_schema.types):
        raise KeyError("Not all field types have match in polars.")

    for name, pa_type in zip(pa_schema.names, pa_schema.types):
        pl_schema[name] = arrow2polars[pa_type]

    return pl_schema


CONTEXTS = ["CG", "CHG", "CHH"]
CYTOSINE_SUMFUNC = ["wmean", "mean", "min", "max", "median", "1pgeom"]
AvailableSumfunc = Literal["wmean", "mean", "min", "max", "median", "1pgeom"]
AvailableBAM = Literal["bismark"]
AvailablePlotStats = Literal["mean", "wmean"]


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
