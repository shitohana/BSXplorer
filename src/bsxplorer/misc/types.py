from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import polars as pl
from pydantic import AfterValidator

from .utils import CONTEXTS, STRANDS, UniversalBatchSchema


def genome_cast(genome: pl.DataFrame):
    return genome.cast(
        dict(
            chr=UniversalBatchSchema["chr"],
            upstream=UniversalBatchSchema["position"],
            start=UniversalBatchSchema["position"],
            end=UniversalBatchSchema["position"],
            downstream=UniversalBatchSchema["position"],
        )
    )


def genome_check(genome: pl.DataFrame):
    assert all(
        col in genome.columns
        for col in ["chr", "upstream", "downstream", "start", "end"]
    ), pl.SchemaError(
        'Not all necessary fields are present in DataFrame (must have "chr", '
        '"upstream", "downstream", "start", "end")'
    )
    return genome


def path_cast(path: str | Path):
    if not isinstance(path, Path):
        path = Path(path)
    path = path.expanduser().absolute()
    return path


def path_check(path: Path):
    assert path.exists(), FileNotFoundError(str(path))
    return path


def context_check(context: str | None):
    if context is not None:
        assert context in CONTEXTS, ValueError(
            f"Context {context} is not supported ({CONTEXTS})"
        )
    return context


def strand_check(strand: str | None):
    if strand is not None:
        assert strand in STRANDS, ValueError(
            f"Context {strand} is not supported ({STRANDS})"
        )
    return strand


def mb2bytes(mb: int):
    return mb * (1024**2)


GenomeDf = Annotated[
    pl.DataFrame, AfterValidator(genome_check), AfterValidator(genome_cast)
]
ExistentPath = Annotated[
    Union[str, Path], AfterValidator(path_cast), AfterValidator(path_check)
]
Context = Annotated[
    Optional[Literal["CG", "CHG", "CHH"]], AfterValidator(context_check)
]
Strand = Annotated[Optional[Literal["+", "-"]], AfterValidator(strand_check)]
Mb2Bytes = Annotated[int, AfterValidator(mb2bytes)]
