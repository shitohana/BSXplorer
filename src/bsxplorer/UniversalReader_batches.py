from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Literal

import polars as pl
import pyarrow as pa

from .utils import polars2arrow_convert, arrow2polars_convert

ARROW_SCHEMAS = {
    "bedgraph": pa.schema([
        ("chr", pa.utf8()),
        ("start", pa.uint64()),
        ("position", pa.uint64()),
        ("density", pa.float32()),
    ]),
    "coverage": pa.schema([
        ("chr", pa.utf8()),
        ("position", pa.uint64()),
        ("end", pa.uint64()),
        ("density", pa.float32()),
        ("count_m", pa.uint32()),
        ("count_um", pa.uint32())
    ]),
    "cgmap": pa.schema([
        ("chr", pa.utf8()),
        ("nuc", pa.utf8()),  # G/C
        ("position", pa.uint64()),
        ("context", pa.utf8()),
        ("dinuc", pa.utf8()),
        ("density", pa.float32()),
        ("count_m", pa.uint32()),
        ("count_total", pa.uint32()),
    ]),
    "bismark": pa.schema([
        ("chr", pa.utf8()),
        ("position", pa.uint64()),
        ("strand", pa.utf8()),
        ("count_m", pa.uint32()),
        ("count_um", pa.uint32()),
        ("context", pa.utf8()),
        ("trinuc", pa.utf8())
    ]),
    "binom": pa.schema([
        ("chr", pa.utf8()),
        ("strand", pa.utf8()),
        ("position", pa.uint64()),
        ("context", pa.utf8()),
        ("p_value", pa.float64()),
    ])
}

ReportTypes = Literal["bismark", "cgmap", "bedgraph", "coverage", "binom"]
REPORT_TYPES_LIST = ["bismark", "cgmap", "bedgraph", "coverage", "binom"]


class BaseBatch(ABC):
    def __init__(self, df: pl.DataFrame):
        self.data = self.get_validated(df)

    def get_validated(self, df: pl.DataFrame):
        if all(c in df.columns for c in self.colnames()):
            try:
                return df.select(self.colnames()).cast(self.pl_schema())
            except Exception as e:
                raise pl.SchemaError(e)
        else:
            raise KeyError("Not all columns from schema in batch")

    @classmethod
    @abstractmethod
    def pl_schema(cls) -> OrderedDict:
        ...

    @classmethod
    def pa_schema(cls) -> pa.Schema:
        return polars2arrow_convert(cls.pl_schema())

    @classmethod
    def colnames(cls):
        return list(FullSchemaBatch.pl_schema().keys())

    def to_arrow(self):
        return self.data.to_arrow().cast(self.pa_schema())


class ConvertedBatch(BaseBatch):
    @classmethod
    @abstractmethod
    def from_full(cls, full_batch: FullSchemaBatch):
        ...


class FullSchemaBatch(BaseBatch):
    @classmethod
    def pl_schema(cls) -> OrderedDict:
        return OrderedDict(
            chr=pl.Utf8,
            strand=pl.Utf8,
            position=pl.UInt64,
            context=pl.Utf8,
            trinuc=pl.Utf8,
            count_m=pl.UInt32,
            count_total=pl.UInt32,
            density=pl.Float64
        )

    def __init__(self, data: pl.DataFrame, raw: pa.Table | pa.RecordBatch):
        super().__init__(data)

        self.raw = raw

    def __len__(self):
        return len(self.data)

    def filter_not_none(self):
        self.data = self.data.filter(pl.col("density").is_not_nan())
        return self

    def to_bismark(self):
        converted = (
            self.data
            .select([
                "chr", "position", "strand", "count_m",
                (pl.col("count_total") - pl.col("count_m")).alias("count_um"),
                "context", "trinuc"
            ])
        )
        return converted

    def to_cgmap(self):
        converted = (
            self.data
            .select([
                "chr",
                pl.when(strand="+").then(pl.lit("C")).otherwise(pl.lit("G")).alias("nuc"),
                "position", "context",
                pl.col("trinuc").str.slice(0, 2).alias("dinuc"),
                "density", "count_m", "count_total"
            ])
        )
        return converted

    def to_coverage(self):
        converted = (
            self.data
            .filter(pl.col("count_total") != 0)
            .select([
                "chr",
                (pl.col("position")).alias("start"),
                (pl.col("position") + 1).alias("end"),
                "density",
                "count_m",
                (pl.col("count_total") - pl.col("count_m")).alias("count_um")
            ])
        )
        return converted

    def to_bedGraph(self):
        converted = (
            self.data
            .filter(pl.col("count_total") != 0)
            .select([
                "chr",
                (pl.col("position") - 1).alias("start"),
                (pl.col("position")).alias("end"),
                (pl.col("count_m") / pl.col("count_total")).alias("density")
            ])
        )
        return converted


