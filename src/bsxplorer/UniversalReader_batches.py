from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import polars as pl
import pyarrow as pa

from .utils import polars2arrow_convert

class BaseBatch(ABC):
    def __init__(self, df: pl.DataFrame):
        self.validate_schema(df)
        self.data = df

    def validate_schema(self, df: pl.DataFrame):
        for i, j in zip(df.schema.items(), self.pl_schema().items()):
            if i != j:
                raise pl.SchemaError()
    @classmethod
    @abstractmethod
    def pl_schema(cls) -> OrderedDict:
        ...

    @classmethod
    def pa_schema(cls) -> pa.Schema:
        return polars2arrow_convert(cls.pl_schema())

    def to_arrow(self):
        return self.data.to_arrow().cast(self.pa_schema())


class ConvertedBatch(BaseBatch):
    @classmethod
    @abstractmethod
    def from_full(cls, full_batch: FullSchemaBatch):
        ...


class BedGraphBatch(ConvertedBatch):
    @classmethod
    def pl_schema(cls):
        return OrderedDict(
            chr=pl.Utf8,
            start=pl.UInt64,
            end=pl.UInt64,
            density=pl.Float64
        )

    @classmethod
    def from_full(cls, full_batch: FullSchemaBatch):
        converted = (
            full_batch.data
            .filter(pl.col("count_total") != 0)
            .select([
                "chr",
                (pl.col("position") - 1).alias("start"),
                (pl.col("position")).alias("end"),
                (pl.col("count_m") / pl.col("count_um")).alias("density")
            ])
            .cast(cls.pl_schema())
        )
        return converted


# todo write other


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
            count_total=pl.UInt32
        )

    def __init__(self, data: pl.DataFrame):
        super().__init__(data)

    # def to_coverage(self):
    #     converted = (
    #         self.data
    #         .filter(pl.col("count_total") != 0)
    #         .select([
    #             "chr",
    #             (pl.col("position")).alias("start"),
    #             (pl.col("position")).alias("end"),
    #             (pl.col(count_m))
    #         ])
    #     )

