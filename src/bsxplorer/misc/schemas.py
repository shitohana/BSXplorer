import gc
from enum import Enum
from functools import cached_property

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..misc.utils import fraction_v


class BaseSchema:
    def __init__(self, pl_schema: pl.Schema) -> None:
        self.schema = pl_schema

    @cached_property
    def polars(self):
        return self.schema

    @cached_property
    def arrow(self):
        return pa.schema(
            [
                pa.field(name, self.pl2pa_mapping(dtype))
                for name, dtype in zip(self.polars.names(), self.polars.dtypes())
            ]
        )

    @staticmethod
    def pl2pa_mapping(dtype: pl.DataType) -> pa.DataType:
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
        return polars2arrow[dtype]  # type: ignore


class ReportSchema(Enum):
    __slots__ = ["polars", "arrow"]

    BEDGRAPH = BaseSchema(
        pl.Schema(
            dict(
                [
                    ("chr", pl.Utf8()),
                    ("start", pl.UInt64()),
                    ("end", pl.UInt64()),
                    ("density", pl.Float32()),
                ]
            )
        )
    )
    COVERAGE = BaseSchema(
        pl.Schema(
            dict(
                [
                    ("chr", pl.Utf8()),
                    ("start", pl.UInt64()),
                    ("end", pl.UInt64()),
                    ("density", pl.Float32()),
                    ("count_m", pl.UInt32()),
                    ("count_um", pl.UInt32()),
                ]
            )
        )
    )
    CGMAP = BaseSchema(
        pl.Schema(
            dict(
                [
                    ("chr", pl.Utf8()),
                    ("nuc", pl.Utf8()),  # G/C
                    ("position", pl.UInt64()),
                    ("context", pl.Utf8()),
                    ("dinuc", pl.Utf8()),
                    ("density", pl.Float32()),
                    ("count_m", pl.UInt32()),
                    ("count_total", pl.UInt32()),
                ]
            )
        )
    )
    BISMARK = BaseSchema(
        pl.Schema(
            dict(
                [
                    ("chr", pl.Utf8()),
                    ("position", pl.UInt64()),
                    ("strand", pl.Utf8()),  # G/C
                    ("count_m", pl.UInt32()),
                    ("count_um", pl.UInt32()),
                    ("context", pl.Utf8()),
                    ("trinuc", pl.Utf8()),
                ]
            )
        )
    )
    BINOM = BaseSchema(
        pl.Schema(
            dict(
                [
                    ("chr", pl.Utf8()),
                    ("strand", pl.Utf8()),
                    ("position", pl.UInt64()),
                    ("context", pl.Utf8()),
                    ("p_value", pl.Float64()),
                ]
            )
        )
    )
    UNIVERSAL = BaseSchema(
        pl.Schema(
            dict(
                chr=pl.Utf8,
                strand=pl.Utf8,
                position=pl.UInt64,
                context=pl.Utf8,
                trinuc=pl.Utf8,
                count_m=pl.UInt32,
                count_total=pl.UInt32,
                density=pl.Float64,
            )
        )
    )

    def get_universal(self, raw: pa.Table, **kwargs):
        if self == ReportSchema.BISMARK:
            mutated = (
                pl.from_arrow(raw)
                .with_columns(
                    (pl.col("count_m") + pl.col("count_um")).alias("count_total")
                )
                .with_columns(
                    (pl.col("count_m") / pl.col("count_total")).alias("density")
                )
            )
        elif self == ReportSchema.CGMAP:
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
            self == ReportSchema.COVERAGE
            or self == ReportSchema.BEDGRAPH
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
                assert "cytosine_file" in kwargs, ValueError(
                    "Cytosine file (param cytosine_file) needs to be specified"
                )
                chrom_min, chrom_max = (
                    batch_stats.filter(chr=chrom).select(["min", "max"]).row(0)
                )

                # Read and filter sequence
                filters = [
                    ("chr", "=", chrom),
                    ("position", ">=", chrom_min),
                    ("position", "<=", chrom_max),
                ]
                pa_filtered_sequence = pq.read_table(
                    kwargs.get("cytosine_file"), filters=filters
                )

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

                if self == ReportSchema.COVERAGE:
                    mutated = output.with_columns(
                        [
                            (pl.col("count_m") + pl.col("count_um")).alias(
                                "count_total"
                            ),
                            pl.col("context").alias("trinuc"),
                        ]
                    ).with_columns(
                        (pl.col("count_m") / pl.col("count_total")).alias("density")
                    )
                else:
                    assert "max_out" in kwargs, ValueError(
                        "Max coverage for bedGraph conversion (param max_out) needs to "
                        "be specified"
                    )
                    converted = fraction_v(
                        (output["density"] / 100).to_list(), kwargs.get("max_out")
                    )

                    fractioned = output.with_columns(
                        [
                            pl.lit(converted[0]).alias("count_m"),
                            pl.lit(converted[1]).alias("count_total"),
                        ]
                    )
                    mutated = fractioned.with_columns(pl.col("context").alias("trinuc"))

        elif self == ReportSchema.BINOM:
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
            raise ValueError(self.value)

        return mutated

    def __getattribute__(self, item):
        if item == "arrow":
            return self.__getattribute__("value").arrow
        elif item == "polars":
            return self.__getattribute__("value").polars
        else:
            return super().__getattribute__(item)


def validate(df: pl.DataFrame, schema: ReportSchema):
    assert all(c in df.columns for c in schema.polars.names()), KeyError(
        f"Not all columns from schema in batch "
        f"(missing {list(set(schema.polars.names()) - set(df.columns))})"
    )
    try:
        return df.select(schema.polars.names()).cast(schema.polars)  # type: ignore
    except Exception as e:
        raise pl.SchemaError(e) from None
