from __future__ import annotations

import functools
import gzip
from typing import Annotated, Literal, Optional

import polars as pl
from polars import DataFrame
from pydantic import BaseModel, Field, field_validator, validate_call
from pyreadr import write_rds

from .IO import UniversalReader
from .misc.types import GenomeDf
from .misc.utils import (
    MetageneJoinedSchema,
    remove_extension,
)


class MetageneModel(BaseModel):
    upstream_windows: int = Field(ge=0, title="Upstream windows number", default=0)
    gene_windows: int = Field(gt=0, title="Region body windows number")
    downstream_windows: int = Field(ge=0, title="Downstream windows number", default=0)
    strand: Optional[Literal["+", "-", "."]] = Field(
        default=None,
        title="Metagene strand",
        description="Defines the strand if metagene was filtered by it.",
    )
    context: Optional[Literal["CG", "CHG", "CHH"]] = Field(
        default=None,
        title="Methylation context",
        description="Defines the context if metagene was filtered by it.",
    )

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class MetageneBase(MetageneModel):
    """
    Base class for :class:`Metagene` and plots.
    """

    report_df: DataFrame = Field(
        exclude=True,
        title="Methylation data",
        description="pl.DataFrame with cytosine methylation status.",
    )
    plot_data: Optional[pl.DataFrame] = Field(
        exclude=True, title="Plot data (optional)", default=None
    )

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def read_metagene(
        cls,
        reader: UniversalReader,
        genome: GenomeDf,
        upstream_windows: Annotated[int, Field(ge=0)] = 100,
        body_windows: Annotated[int, Field(gt=0)] = 200,
        downstream_windows: Annotated[int, Field(ge=0)] = 100,
        sumfunc: Literal["wmean", "mean", "min", "max", "median", "1pgeom"] = "wmean",
    ):
        genome = genome.rename({"strand": "gene_strand"})
        # POLARS EXPRESSIONS
        # Region position check
        UP_REGION = pl.col("position") < pl.col("start")
        DOWN_REGION = pl.col("position") > pl.col("end")

        # Fragment numbers calculation
        # 1e-10 is added (position is always < end)
        UP_FRAGMENT = (
            (
                (pl.col("position") - pl.col("upstream"))
                / (pl.col("start") - pl.col("upstream"))
            )
            * upstream_windows
        ).floor()
        BODY_FRAGMENT = (
            (
                (pl.col("position") - pl.col("start"))
                / (pl.col("end") - pl.col("start") + 1e-10)
            )
            * body_windows
        ).floor() + upstream_windows
        DOWN_FRAGMENT = (
            (
                (
                    (pl.col("position") - pl.col("end"))
                    / (pl.col("downstream") - pl.col("end") + 1e-10)
                )
                * downstream_windows
            ).floor()
            + upstream_windows
            + body_windows
        )

        # Firstly BismarkPlot was written so there were only one sum statistic - mean.
        # Sum and count of densities was calculated for further weighted mean analysis
        # in respect to fragment size
        # For backwards compatibility, for newly introduces statistics, column names are
        # kept the same.
        # Count is set to 1 and "sum" to actual statistics (e.g. median, min, e.t.c)

        AGG_EXPRS = [
            pl.lit(1).alias("count"),
            pl.first("gene_strand").alias("strand"),
            pl.first("gene"),
            pl.first("id"),
        ]

        if sumfunc == "median":
            AGG_EXPRS.append(pl.median("density").alias("sum"))
        elif sumfunc == "min":
            AGG_EXPRS.append(pl.min("density").alias("sum"))
        elif sumfunc == "max":
            AGG_EXPRS.append(pl.max("density").alias("sum"))
        elif sumfunc == "1pgeom":
            AGG_EXPRS.append((pl.col("density").log1p().mean().exp() - 1).alias("sum"))
        elif sumfunc == "mean":
            AGG_EXPRS.append(pl.mean("density").alias("sum"))
        else:
            AGG_EXPRS.append(pl.sum("density").alias("sum"))
            AGG_EXPRS.pop(0)
            AGG_EXPRS.insert(0, pl.count("density").alias("count"))

        GENE_LABEL_COLS = [
            pl.col("chr"),
            pl.concat_str(pl.col("start"), pl.col("end"), separator="-"),
        ]

        GROUP_BY_COLS = ["chr", "start", "context", "fragment"]

        def process_batch(df: pl.DataFrame):
            result = (
                df.lazy()
                .filter(pl.col("count_total") != 0)
                # Sort by position for joining
                .sort(["chr", "position"])
                # Join with nearest
                .join_asof(
                    genome.lazy(),
                    left_on="position",
                    right_on="upstream",
                    by="chr",
                    strategy="backward",
                )
                # Limit by end of region
                .filter(pl.col("position") <= pl.col("downstream"))
                # Filter by strand if it is defined
                .filter(
                    (
                        (pl.col("gene_strand") != ".")
                        & (pl.col("gene_strand") == pl.col("strand"))
                    )
                    | (pl.col("gene_strand") == ".")
                )
                # Calculate fragment ids
                .with_columns(
                    [
                        pl.when(UP_REGION)
                        .then(UP_FRAGMENT)
                        .when(DOWN_REGION)
                        .then(DOWN_FRAGMENT)
                        .otherwise(BODY_FRAGMENT)
                        .alias("fragment"),
                        pl.concat_str(GENE_LABEL_COLS, separator=":").alias("gene"),
                    ]
                )
                # Assign types
                .cast(
                    {
                        key: value
                        for key, value in MetageneJoinedSchema.items()
                        if key in GROUP_BY_COLS
                    }
                )
                # gather fragment stats
                .group_by(GROUP_BY_COLS)
                # Calculate sumfunc
                .agg(AGG_EXPRS)
                .drop_nulls(subset=["sum"])
                .cast(
                    {
                        key: value
                        for key, value in MetageneJoinedSchema.items()
                        if key not in GROUP_BY_COLS
                    }
                )
                .select(list(MetageneJoinedSchema.keys()))
                .collect()
            )
            return result

        print("Reading report from", reader.file)
        report_df = None

        for batch in reader:
            processed = process_batch(batch.data)

            if report_df is None:
                report_df = processed
            else:
                report_df.extend(processed)

        print("DONE\n")
        return cls(
            report_df=report_df,
            upstream_windows=upstream_windows,
            gene_windows=body_windows,
            downstream_windows=downstream_windows,
        )

    def save_rds(self, filename, compress: bool = False):
        """
        Save Metagene in RDS format.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        write_rds(
            filename, self.report_df.to_pandas(), compress="gzip" if compress else None
        )

    def save_tsv(self, filename, compress=False):
        """
        Save Metagene in TSV.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.report_df.write_csv(file, separator="\t")
        else:
            self.report_df.write_csv(filename, separator="\t")

    @functools.cached_property
    def total_windows(self):
        return self.upstream_windows + self.downstream_windows + self.gene_windows

    @functools.cached_property
    def _x_ticks(self):
        return [
            self.upstream_windows / 2,
            self.upstream_windows,
            self.total_windows / 2,
            self.gene_windows + self.upstream_windows,
            self.total_windows - (self.downstream_windows / 2),
        ]

    @functools.cached_property
    def _borders(self):
        return [
            self.upstream_windows,
            self.gene_windows + self.upstream_windows,
        ]

    def __len__(self):
        return len(self.report_df)


class MetageneFilesBase(BaseModel):
    samples: list[type[BaseModel]] = Field(
        exclude=True,
        title="List of samples",
        description="List of initialized Metagene instances.",
    )
    labels: Optional[list[str]] = Field(
        default_factory=list,
        title="Sample labels",
        description="Equal length list of samples' labels.",
    )

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: list[str], values) -> list[str]:
        if len(labels) == 0:
            return list(map(str, range(len(values["samples"]))))
        assert len(labels) == values["samples"], ValueError(
            "Samples and labels lists should be equal length!"
        )
        return labels

    @field_validator("samples")
    @classmethod
    def _validate_saples(cls, samples: list[type[BaseModel]]):
        assert (
            functools.reduce(
                lambda a, b: a == b,
                map(lambda sample: sample.upstream_windows, samples),
            )
            and functools.reduce(
                lambda a, b: a == b, map(lambda sample: sample.gene_windows, samples)
            )
            and functools.reduce(
                lambda a, b: a == b,
                map(lambda sample: sample.downstream_windows, samples),
            )
        ), ValueError("Samples have different number of windows!")
        return samples

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    def save_rds(self, base_filename, compress: bool = False, merge: bool = False):
        """
        Save Metagene in Rds.

        Parameters
        ----------
        base_filename
            Base path for file (final path will be ``base_filename+label.rds``).
        compress
            Whether to compress to gzip or not.
        merge
            Do samples need to be merged into single :class:`Metagene` before saving.
        """
        if merge:
            merged = pl.concat(
                [
                    sample.report_df.lazy().with_columns(pl.lit(label))
                    for sample, label in zip(self.samples, self.labels)
                ]
            )
            write_rds(
                base_filename, merged.to_pandas(), compress="gzip" if compress else None
            )
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_rds(
                    f"{remove_extension(base_filename)}_{label}.rds",
                    compress="gzip" if compress else None,
                )

    def save_tsv(self, base_filename, compress: bool = False, merge: bool = False):
        """
        Save Metagenes in TSV.

        Parameters
        ----------
        base_filename
            Base path for file (final path will be ``base_filename+label.tsv``).
        compress
            Whether to compress to gzip or not.
        merge
            Do samples need to be merged into single :class:`Metagene` before saving.
        """
        if merge:
            merged = pl.concat(
                [
                    sample.report_df.lazy().with_columns(pl.lit(label))
                    for sample, label in zip(self.samples, self.labels)
                ]
            )
            if compress:
                with gzip.open(base_filename + ".gz", "wb") as file:
                    # noinspection PyTypeChecker
                    merged.write_csv(file, separator="\t")
            else:
                merged.write_csv(base_filename, separator="\t")
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_tsv(
                    f"{remove_extension(base_filename)}_{label}.tsv", compress=compress
                )
