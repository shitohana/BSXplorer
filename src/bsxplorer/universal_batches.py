from __future__ import annotations

import functools
import gc
from copy import deepcopy
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import ReportSchema, validate
from .utils import fraction_v

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

    @classmethod
    def from_arrow(cls, raw: pa.Table, report_schema: ReportSchema, **kwargs):
        if report_schema == ReportSchema.BISMARK:
            mutated = (
                pl.from_arrow(raw)
                .with_columns(
                    (pl.col("count_m") + pl.col("count_um")).alias("count_total")
                )
                .with_columns(
                    (pl.col("count_m") / pl.col("count_total")).alias("density")
                )
            )
        elif report_schema == ReportSchema.CGMAP:
            mutated = pl.from_arrow(raw).with_columns(
                [
                    pl.when(pl.col("nuc") == "C")
                    .then(pl.lit("+"))
                    .otherwise(pl.lit("-"))
                    .alias("strand"),
                    pl.when(pl.col("dinuc") == "CG")
                    # if dinuc == "CG" => trinuc = "CG"
                    .then(pl.col("dinuc"))
                    # otherwise trinuc = dinuc + context_last
                    .otherwise(
                        pl.concat_str(
                            [pl.col("dinuc"), pl.col("context").str.slice(-1)]
                        )
                    )
                    .alias("trinuc"),
                ]
            )
        elif (
                report_schema == ReportSchema.COVERAGE or
                report_schema == ReportSchema.BEDGRAPH
        ):
            batch = pl.from_arrow(raw)

            # get batch stats
            batch_stats = batch.group_by("chr").agg(
                [
                    pl.col("position").max().alias("max"),
                    pl.col("position").min().alias("min"),
                ]
            )

            output = None

            for chrom in batch_stats["chr"]:
                chrom_min, chrom_max = (
                    batch_stats.filter(chr=chrom).select(["min", "max"]).row(0)
                )

                # Read and filter sequence
                filters = [
                    ("chr", "=", chrom),
                    ("position", ">=", chrom_min),
                    ("position", "<=", chrom_max),
                ]
                pa_filtered_sequence = pq.read_table(kwargs.get("cytosine_file"), filters=filters)

                modified_schema = pa_filtered_sequence.schema
                modified_schema = modified_schema.set(1, pa.field("context", pa.utf8()))
                modified_schema = modified_schema.set(2, pa.field("chr", pa.utf8()))

                pa_filtered_sequence = pa_filtered_sequence.cast(modified_schema)

                pl_sequence = (
                    pl.from_arrow(pa_filtered_sequence)
                    .with_columns(
                        [
                            pl.when(pl.col("strand").eq(True))
                            .then(pl.lit("+"))
                            .otherwise(pl.lit("-"))
                            .alias("strand")
                        ]
                    )
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

                if output is None:
                    output = aligned
                else:
                    output.extend(aligned)

                gc.collect()

                if report_schema == ReportSchema.COVERAGE:
                    mutated = (
                        output
                        .with_columns(
                            [
                                (pl.col("count_m") + pl.col("count_um")).alias("count_total"),
                                pl.col("context").alias("trinuc"),
                            ]
                        )
                        .with_columns((pl.col("count_m") / pl.col("count_total")).alias("density"))
                    )
                else:
                    converted = fraction_v((output["density"] / 100).to_list(), kwargs.get("max_out"))

                    fractioned = output.with_columns(
                        [
                            pl.lit(converted[0]).alias("count_m"),
                            pl.lit(converted[1]).alias("count_total"),
                        ]
                    )
                    mutated = fractioned.with_columns(pl.col("context").alias("trinuc"))

        elif report_schema == ReportSchema.BINOM:
            assert "p_value" in kwargs, ValueError(
                "Methylation P-value (param p_value) needs to be specified"
            )
            mutated = (
                pl.from_arrow(raw)
                .with_columns(
                    (pl.col("p_value") <= kwargs["p_value"])
                    .cast(pl.UInt8)
                    .alias("count_m")
                )
                .with_columns(
                    [
                        pl.col("context").alias("trinuc"),
                        pl.lit(1).alias("count_total"),
                        pl.col("count_m").cast(pl.Float64).alias("density"),
                    ]
                )
            )
        else:
            raise ValueError(report_schema)

        return cls(mutated, report_schema)

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
