from collections.abc import Iterable
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

from bsxplorer.plot.utils import NpArray


class LinePlotData(BaseModel):
    x: NpArray
    y: NpArray
    x_ticks: Iterable[int]
    borders: Iterable[int]
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


class HeatMapData(BaseModel):
    matrix: NpArray
    x_ticks: Iterable[int]
    borders: Iterable[int]
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


class BoxPlotData(BaseModel):
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
