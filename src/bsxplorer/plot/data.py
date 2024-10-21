from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LinePlotData:
    x: np.ndarray
    y: np.ndarray
    x_ticks: Iterable
    borders: Iterable
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None,
    label: str = ""
    x_labels: list[str] | None = None


@dataclass
class HeatMapData:
    matrix: np.ndarray
    x_ticks: list
    borders: list
    label: str = ""
    x_labels: list[str] | None = None


@dataclass
class BoxPlotData:
    values: list
    label: str
    locus: list = None
    id: list = None

    @classmethod
    def empty(cls, label=None):
        return cls([], label if label is not None else "", [], [])
