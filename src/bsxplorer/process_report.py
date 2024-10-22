from typing import Annotated

import polars as pl
from pydantic import Field

from .IO import UniversalReader
from .misc.schemas import MetageneSchema
from .misc.types import GenomeDf
from .misc.utils import AvailableSumfunc


def get_fragment_expr(
    upstream_windows: Annotated[int, Field(ge=0)] = 100,
    body_windows: Annotated[int, Field(gt=0)] = 200,
    downstream_windows: Annotated[int, Field(ge=0)] = 100,
) -> pl.Expr:
    return (
        # Upstream
        pl.when(pl.col("position") < pl.col("start"))
        .then(
            (
                (pl.col("position") - pl.col("upstream"))
                / (pl.col("start") - pl.col("upstream"))
                * upstream_windows
            ).floor()
        )
        # Downstream
        .when(pl.col("position") > pl.col("end"))
        .then(
            (
                (pl.col("position") - pl.col("end"))
                / (pl.col("downstream") - pl.col("end") + 1e-10)
                * downstream_windows
            ).floor()
            + upstream_windows
            + body_windows
        )
        .otherwise(
            (
                (pl.col("position") - pl.col("start"))
                / (pl.col("end") - pl.col("start") + 1e-10)
                * body_windows
            ).floor()
            + upstream_windows
        )
        .alias("fragment")
    )


def get_locus_expr():
    return pl.concat_str(
        [
            pl.col("chr"),
            pl.concat_str(pl.col("start"), pl.col("end"), separator="-"),
        ],
        separator=":",
    )


def get_agg_expr(sumfunc: str) -> dict[str, pl.Expr]:
    if sumfunc == "median":
        return dict(sum=pl.median("density"), count=pl.lit(1))
    if sumfunc == "min":
        return dict(sum=pl.min("density"), count=pl.lit(1))
    if sumfunc == "max":
        return dict(sum=pl.max("density"), count=pl.lit(1))
    if sumfunc == "mean":
        return dict(sum=pl.mean("density"), count=pl.lit(1))
    if sumfunc == "wmean":
        return dict(sum=pl.sum("density"), count=pl.len())
    if sumfunc == "1pgeom":
        return dict(sum=(pl.col("density").log1p().mean().exp() - 1), count=pl.lit(1))
    raise NotImplementedError(sumfunc)


def process_batch_lazy(
    batch: pl.LazyFrame,
    genome: pl.LazyFrame,
    fragment_expr: pl.Expr,
    agg_expr: dict[str, pl.Expr],
    locus_expr: pl.Expr,
):
    result = (
        batch.filter(pl.col("count_total").gt(0))
        .sort(["chr", "position"])
        .rename(dict(strand="c_strand"))
        .join_asof(
            genome,
            left_on="position",
            right_on="upstream",
            by="chr",
            strategy="backward",
        )
        .filter(
            *[
                pl.col("position").le(pl.col("downstream")),
                pl.col("strand").eq(".").xor(pl.col("strand").eq(pl.col("c_strand"))),
            ]
        )
        .with_columns(fragment_expr)
        .group_by("chr", "start", "context", "fragment")
        .agg(
            strand=pl.first("strand"),
            id=pl.first("id"),
            end=pl.first("end"),
            # sum, count
            **agg_expr,
        )
        .with_columns(gene=locus_expr)
        .drop_nulls(subset=["sum"])
        .cast(MetageneSchema.schema)
        .select(MetageneSchema.schema)
    )
    return result


def read_report(
    reader: UniversalReader,
    genome: GenomeDf,
    upstream_windows: Annotated[int, Field(ge=0)] = 100,
    body_windows: Annotated[int, Field(gt=0)] = 200,
    downstream_windows: Annotated[int, Field(ge=0)] = 100,
    sumfunc: AvailableSumfunc = "wmean",
):
    agg_expr = get_agg_expr(sumfunc)
    fragment_expr = get_fragment_expr(
        upstream_windows, body_windows, downstream_windows
    )
    locus_expr = get_locus_expr()

    report_df = pl.DataFrame(schema=MetageneSchema.polars)
    for batch in reader:
        processed = process_batch_lazy(
            batch.data.lazy(), genome.lazy(), fragment_expr, agg_expr, locus_expr
        )
        report_df.extend(processed.collect())

    return report_df
