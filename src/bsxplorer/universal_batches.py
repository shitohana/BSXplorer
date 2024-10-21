from __future__ import annotations

import functools
from copy import deepcopy
from typing import Literal

import polars as pl
import pyarrow as pa

from .schemas import ReportSchema, validate

# Fixme
ReportTypes = Literal["bismark", "cgmap", "binom", "bedgraph", "coverage"]


class BaseBatch:
    def __init__(self, df: pl.DataFrame, schema: ReportSchema):
        self.schema = schema
        self.data = validate(df, schema)

    def to_arrow(self):
        return self.data.to_arrow().cast(self.schema.arrow)


class UniversalBatch(BaseBatch):
    """
    Class for storing and converting methylation report data.
    """

    def __init__(self, data: pl.DataFrame, raw: pa.Table | pa.RecordBatch | None):
        self.schema = ReportSchema.UNIVERSAL
        self.raw = raw
        super().__init__(data, self.schema)

    def __len__(self):
        return len(self.data)

    def filter_not_none(self):
        """
        Filter cytosines which have 0 reads.
        """
        self.data = self.data.filter(pl.col("density").is_not_nan())
        return self

    def cast(self, report_schema: ReportSchema) -> BaseBatch:
        """
        Cast UniversalBatch to another schema.

        Parameters
        ----------
        report_schema
            Report schema to cast to

        Returns
        -------
        BaseBatch
            Converted batch
        """
        if report_schema == ReportSchema.BISMARK:
            converted = self.data.select(
                [
                    "chr",
                    "position",
                    "strand",
                    "count_m",
                    (pl.col("count_total") - pl.col("count_m")).alias("count_um"),
                    "context",
                    "trinuc",
                ]
            )
        elif report_schema == report_schema.CGMAP:
            converted = self.data.select(
                [
                    "chr",
                    pl.when(pl.col("strand") == "+")
                    .then(pl.lit("C"))
                    .otherwise(pl.lit("G"))
                    .alias("nuc"),
                    "position",
                    "context",
                    pl.col("trinuc").str.slice(0, 2).alias("dinuc"),
                    "density",
                    "count_m",
                    "count_total",
                ]
            )
        elif report_schema == report_schema.COVERAGE:
            converted = self.data.filter(pl.col("count_total") != 0).select(
                [
                    "chr",
                    (pl.col("position")).alias("start"),
                    (pl.col("position") + 1).alias("end"),
                    "density",
                    "count_m",
                    (pl.col("count_total") - pl.col("count_m")).alias("count_um"),
                ]
            )
        elif report_schema == report_schema.BEDGRAPH:
            converted = self.data.filter(pl.col("count_total") != 0).select(
                [
                    "chr",
                    (pl.col("position") - 1).alias("start"),
                    (pl.col("position")).alias("end"),
                    (pl.col("count_m") / pl.col("count_total")).alias("density"),
                ]
            )
        else:
            raise ValueError(report_schema)

        return BaseBatch(converted, ReportSchema.COVERAGE)

    def filter_data(self, **kwargs):
        """
        Filter data by expression or keyword arguments

        Parameters
        ----------
        kwargs
            keywords arguements to pass to `polars.filter() <https://docs.pola.rs/api/python/version/0.20/reference/dataframe/api/polars.DataFrame.filter.html#polars.DataFrame.filter>`_

        Returns
        -------
        UniversalBatch
        """
        new = deepcopy(self)
        new.data = new.data.filter(**kwargs)
        return new

    @functools.cached_property
    def first_position(self) -> tuple[str, int]:
        """
        Returns first position in batch.

        Returns
        -------
        tuple[str, int]
            Tuple of chromosome name and position
        """
        return self.data["chr"][0], self.data["position"][0]

    @functools.cached_property
    def last_position(self) -> tuple[str, int]:
        """
        Returns last position in batch.

        Returns
        -------
        tuple[str, int]
            Tuple of chromosome name and position
        """
        return self.data["chr"][-1], self.data["position"][-1]
