import gzip
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Optional, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, validate_call
from pyreadr import write_rds
from scipy.stats import stats

from ..misc.schemas import MetageneSchema
from ..misc.utils import reverse_strand
from .utils import NpArray, plot_stat_expr, savgol_line


class PlotData(ABC):
    def save_rds(self, filename, compress: bool = False):
        """
        Save Metagene in RDS format.

        Parameters
        ----------
        filename
            Path for file.
        compress
            Whether to compress to gzip or not.
        """
        write_rds(
            filename, self.dataframe.to_pandas(), compress="gzip" if compress else None
        )

    def save_tsv(self, filename, compress=False) -> None:
        """
        Save Metagene as TSV file.

        Parameters
        ----------
        filename
            Path for file.
        compress
            Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.dataframe.write_csv(file, separator="\t")
        else:
            self.dataframe.write_csv(filename, separator="\t")

    @classmethod
    @abstractmethod
    def from_metagene_df(cls, **kwargs): ...

    @property
    @abstractmethod
    def dataframe(self): ...


class LinePlotData(BaseModel, PlotData):
    x: NpArray
    y: NpArray
    x_ticks: Iterable[float]
    borders: Iterable[float]
    lower: Optional[NpArray] = Field(default=None)
    upper: Optional[NpArray] = Field(default=None)
    label: str = Field(default="")
    x_labels: Optional[list[str]] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, x, y, x_ticks, borders, lower, upper, label, x_labels):
        super().__init__(
            x=x,
            y=y,
            x_ticks=x_ticks,
            borders=borders,
            lower=lower,
            upper=upper,
            label=label,
            x_labels=x_labels,
        )

    @classmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def from_metagene_df(
        cls,
        metagene_df: pl.DataFrame,
        metagene_model: dict,
        smooth: Annotated[int, Field(ge=0)] = 0,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
        stat: str = "wmean",
        merge_strands: bool = True,
        label: str = "",
    ):
        # Merge strands
        if merge_strands:
            metagene_df = metagene_df.filter(pl.col("strand") != "-").extend(
                reverse_strand(metagene_df, metagene_df["fragment"].max())
            )

        # Apply stats expr
        res = (
            metagene_df.group_by("fragment")
            .agg(
                [
                    plot_stat_expr(stat).alias("density"),
                    pl.col("sum"),
                    pl.col("count"),
                    pl.sum("count").alias("n"),
                    (pl.col("sum") / pl.col("count")).mean().alias("average"),
                    (pl.col("sum") - (pl.col("sum") / pl.col("count")))
                    .mean()
                    .pow(2)
                    .alias("variance"),
                ]
            )
            .sort("fragment")
        )

        if 0 < confidence < 1 and stat in ["mean", "wmean"]:
            res = (
                res.with_columns(
                    (pl.col("variance") / pl.col("n")).sqrt().alias("scale")
                )
                .with_columns(
                    pl.struct(["n", "average", "scale"])
                    .map_elements(
                        lambda field: stats.t.interval(
                            confidence,
                            df=field["n"] - 1,
                            loc=field["average"],
                            scale=field["scale"],
                        ),
                        return_dtype=pl.List(pl.Float64),
                    )
                    .alias("interval")
                )
                .with_columns(
                    pl.col("interval").list.to_struct(fields=["lower", "upper"])
                )
                .unnest("interval")
            )
        elif 0 < confidence < 1 and stat not in ["mean", "wmean"]:
            raise ValueError(
                "Confidence bands available only for mean and wmean stat parameters."
            )

        # Fill empty windows
        template = pl.DataFrame(
            dict(fragment=list(range(metagene_model["total_windows"]))),
            schema=dict(fragment=res.schema["fragment"]),
        )

        joined = template.join(res, on="fragment", how="left")

        # Calculate CI
        lower = None
        upper = None
        if 0 < confidence < 1 and stat in ["mean", "wmean"]:
            upper = savgol_line(joined["upper"].to_numpy(), smooth) * 100
            lower = savgol_line(joined["lower"].to_numpy(), smooth) * 100

        # Smooth and convert to percents
        y = savgol_line(joined["density"].to_numpy(), smooth) * 100
        x = np.arange(len(y))

        return LinePlotData(
            x,
            y,
            metagene_model["_x_ticks"],
            metagene_model["_borders"],
            lower,
            upper,
            label,
            ["Upstream", "", "Body", "", "Downstream"],
        )

    @property
    def dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(dict(x=self.x, y=self.y))


class HeatMapData(BaseModel, PlotData):
    matrix: NpArray
    x_ticks: Iterable[float]
    borders: Iterable[float]
    label: str = Field(default="")
    x_labels: Optional[list[str]] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, matrix, x_ticks, borders, label, x_labels):
        super().__init__(
            matrix=matrix,
            x_ticks=x_ticks,
            borders=borders,
            label=label,
            x_labels=x_labels,
        )

    @classmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def from_metagene_df(
        cls,
        metagene_df: pl.DataFrame,
        metagene_model: dict,
        nrow: Annotated[int, Field(gt=0)] = 100,
        ncol: Annotated[int, Field(gt=0)] = 100,
        label: str = "",
    ):
        metagene_df = metagene_df.filter(pl.col("strand") != "-").extend(
            reverse_strand(metagene_df, metagene_df["fragment"].max())
        )

        # sort by rows and add row numbers
        hm_data = (
            metagene_df.lazy()
            .group_by("gene")
            .agg(
                [pl.col("fragment"), (pl.col("sum") / pl.col("count")).alias("density")]
            )
            .with_columns(
                (
                    pl.col("density").list.sum() / (metagene_model["total_windows"] + 1)
                ).alias("order")
            )
            .sort("order", descending=True)
            .with_row_count(name="row")
            .with_columns(
                (pl.col("row") / (pl.max("row") + 1) * nrow)
                .floor()
                .alias("row")
                .cast(pl.UInt16)
            )
            .explode(["fragment", "density"])
            .group_by(["row", "fragment"])
            .agg(pl.mean("density"))
        )

        # prepare full template
        template = (
            pl.LazyFrame(data={"row": list(range(nrow))})
            .with_columns(
                pl.lit(list(range(0, metagene_model["total_windows"]))).alias(
                    "fragment"
                )
            )
            .explode("fragment")
            .with_columns(
                [
                    pl.col("fragment").cast(MetageneSchema.polars.get("fragment")),
                    pl.col("row").cast(pl.UInt16),
                ]
            )
        )

        # join template with actual data
        hm_data = (  # template join with orig
            template.join(hm_data, on=["row", "fragment"], how="left")
            .fill_null(0)
            .sort(["row", "fragment"])
            .group_by("row", maintain_order=True)
            .agg(pl.col("density"))
            .collect()["density"]
            .to_list()
        )

        # convert to matrix and percents
        hm_data = np.array(hm_data, dtype=np.float32) * 100

        return cls(
            hm_data,
            metagene_model["_x_ticks"],
            metagene_model["_borders"],
            label,
            ["Upstream", "", "Body", "", "Downstream"],
        )

    @property
    def dataframe(self):
        return pl.from_numpy(self.matrix)


class BoxPlotData(BaseModel, PlotData):
    values: Iterable[int]
    label: str = Field(default="")
    locus: Optional[list[str]] = Field(default=None)
    id: Optional[list[str]] = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def empty(cls, label=None):
        return cls([], label if label is not None else "", [], [])

    def __init__(self, values, label, locus, id):
        super().__init__(values=values, label=label, locus=locus, id=id)

    @classmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def from_metagene_df(
        cls,
        metagene_df: pl.DataFrame,
        filter_context: Optional[Literal["CG", "CHG", "CHH"]] = None,
        label: str = "",
    ):
        df = (
            metagene_df.filter(context=filter_context)
            if filter_context is not None
            else metagene_df
        )

        if not df.is_empty():
            data = df.group_by(["chr", "start"]).agg(
                [
                    pl.first("gene").alias("locus"),
                    pl.first("strand"),
                    pl.first("id"),
                    (pl.sum("sum") / pl.sum("count")).alias("density"),
                ]
            )

            return cls(
                data["density"].to_list(),
                label,
                data["locus"].to_list(),
                data["id"].to_list(),
            )
        else:
            return BoxPlotData.empty(label)

    @property
    def dataframe(self):
        return pl.DataFrame(dict(values=self.values, locus=self.locus, id=self.id))
