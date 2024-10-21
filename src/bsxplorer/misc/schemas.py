from enum import Enum
from functools import cached_property

import polars as pl
import pyarrow as pa


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
