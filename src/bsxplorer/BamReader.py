import time
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Literal

import numpy
import pysam
import numpy as np
from numba import prange, jit
import numba as nb
import polars as pl


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 2, "C"), nb.types.Array(nb.float64, 1, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_methylation_entropy(matrix, columns, window_length, min_depth):
    positions_matrix = np.zeros((matrix.shape[1] - window_length, window_length), dtype=np.int64)
    ME_array = np.zeros((matrix.shape[1] - window_length), dtype=np.float64)

    # Sliding window loop
    for i in prange(matrix.shape[1] - window_length):
        window = matrix[:, i:(i + window_length)]

        none_idx = np.not_equal(window, -1).sum(axis=1)

        # Filter empty values
        window = window[none_idx == window.shape[1], :]
        # Check minimal depth
        if window.shape[1] < min_depth:
            continue

        # Initialize signaures array
        signatures = np.zeros(window.shape[0], dtype=np.int64)
        # Fill signatures array
        for col in prange(window.shape[1]):
            signatures = np.multiply(signatures, 2)
            signatures += window[:, col]

        # Sort and find unique values
        sort_signatures = np.sort(signatures)
        nonzero_indicies = np.nonzero(sort_signatures - np.roll(sort_signatures, -1))[0]
        counts = nonzero_indicies + 1
        counts_shifted = np.roll(counts, 1)
        counts_shifted[0] = 0

        # Unique counts
        counts = counts - counts_shifted
        total_counts = counts.sum()

        ME_value = abs((1 / window_length) * sum([(count / total_counts) * np.log2(count / total_counts) for count in counts]))

        positions_matrix[i, :] = columns[i:(i + window_length)]
        ME_array[i] = ME_value

    return positions_matrix, ME_array


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 2, "C"), nb.types.Array(nb.float64, 1, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_epipolymorphism(matrix, columns, window_length, min_depth):
    positions_matrix = np.zeros((matrix.shape[1] - window_length, window_length), dtype=np.int64)
    PM_array = np.zeros((matrix.shape[1] - window_length), dtype=np.float64)

    # Sliding window loop
    for i in prange(matrix.shape[1] - window_length):
        window = matrix[:, i:(i + window_length)]

        none_idx = np.not_equal(window, -1).sum(axis=1)

        # Filter empty values
        window = window[none_idx == window.shape[1], :]
        # Check minimal depth
        if window.shape[1] < min_depth:
            continue

        # Initialize signaures array
        signatures = np.zeros(window.shape[0], dtype=np.int64)
        # Fill signatures array
        for col in prange(window.shape[1]):
            signatures = np.multiply(signatures, 2)
            signatures += window[:, col]

        # Sort and find unique values
        sort_signatures = np.sort(signatures)
        nonzero_indicies = np.nonzero(sort_signatures - np.roll(sort_signatures, -1))[0]
        counts = nonzero_indicies + 1
        counts_shifted = np.roll(counts, 1)
        counts_shifted[0] = 0

        # Unique counts
        counts = counts - counts_shifted
        total_counts = counts.sum()

        PM_value = 1 - sum([(count / total_counts) for count in counts])

        positions_matrix[i, :] = columns[i:(i + window_length)]
        PM_array[i] = PM_value

    return positions_matrix, PM_array


@jit(
    nb.types.Tuple((nb.types.Array(nb.int64, 1, "C"), nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.int64, 2, "C")))(
        nb.types.Array(nb.int8, 2, "F"),
        nb.types.Array(nb.int64, 1, "C"),
        nb.int64,
        nb.int64
    ),
    nopython=True,
    parallel=True,
    fastmath=True
)
def _jit_PDR(matrix, columns, min_cyt: int, min_depth: int):
    # Filter low cytosines reads
    low_cyt = np.sum(matrix != -1, axis=1) >= min_cyt
    matrix = matrix[low_cyt, :]

    n_cols = matrix.shape[1]
    position_array = np.empty(n_cols, dtype=np.int64)
    pdr_array = np.empty(n_cols, dtype=np.float64)
    count_matrix = np.empty((n_cols, 2), dtype=np.int64)

    for i in prange(n_cols):
        covering_reads = matrix[:, i] != -1

        # Skip if no reads cover this cytosine
        if covering_reads.sum() < min_depth:
            continue

        window = matrix[covering_reads, :]

        concordant_reads = np.logical_xor((window == 1).sum(axis=1), (window == 0).sum(axis=1))

        total = len(concordant_reads)
        concordant_count = concordant_reads.sum()
        discordant_count = total - concordant_count
        PDR_value = discordant_count / total

        pdr_array[i] = PDR_value
        count_matrix[i] = concordant_count, discordant_count
        position_array[i] = columns[i]

    return position_array, pdr_array, count_matrix


nuc_row = namedtuple(typename="nuc_row", field_names=["position", "context", "m", "qual", "id", "strand", "converted"])


class BAMOptions:
    def __init__(self, bamtype: Literal["bismark"]):
        self._bamtype = bamtype

    @property
    def calls_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "z": ("CG", 0),
                "Z": ("CG", 1),
                "x": ("CHG", 0),
                "X": ("CHG", 1),
                "h": ("CHH", 0),
                "H": ("CHH", 1),
                "u": ("U", 0),
                "U": ("U", 1)
            }

    @property
    def orientation_dict(self) -> dict:
        if self._bamtype == "bismark":
            return {
                "CT": True,
                "GA": False
            }


class BAMBatch:
    def __init__(self, methyl_df: pl.DataFrame):
        self.methyl_df = methyl_df

    schema = {
        "position": pl.UInt64,
        "context": pl.Categorical,
        "m": pl.Int8,
        "qual": pl.UInt8,
        "id": pl.Categorical,
        "strand": pl.Categorical,
        "converted": pl.Boolean
    }

    def filter(self, **kwargs):
        return self.__class__(self.methyl_df.filter(**kwargs))

    def filter_qual(self, min_qual: int):
        return self.__class__(self.methyl_df.filter(pl.col("qual") >= min_qual))

    def to_report(self, min_qual: int = 0):
        report = (
            self.methyl_df
            .filter(pl.col("qual") >= min_qual)
            .group_by(["position", "context"])
            .agg([
                pl.first("strand"),
                pl.sum("m").alias("count_m"),
                pl.count("m").alias("count_total")
            ])
            .sort("position")
        )

        return report

    def to_pivot(self):
        if len(self.methyl_df["strand"].cat.get_categories()) > 1:
            indexes = ["id", "converted", "strand"]
        else:
            indexes = ["id", "converted"]

        pivoted = self.methyl_df.pivot(index=indexes, columns="position", values="m").fill_null(-1)

        return PivotRegion(pivoted)


class PivotRegion:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    @property
    def matrix_df(self):
        return self.df.select(pl.all().exclude(["id", "converted", "strand"]))

    @property
    def _jit_compatitable(self):
        df = self.matrix_df
        matrix = df.to_numpy().astype(np.int8)
        columns = np.array(list(map(int, df.columns)), dtype=np.int64)

        return matrix, columns

    def methylation_entropy(self, window_length: int = 4, min_depth: int = 4):
        """
        Wrapper for _jit_methylation_entropy – JIT compiled ME calculating function

        Parameters
        ----------
        window_length
            Length of the sliding window
        min_depth
            Minimal depth of reads to consider this window for calculation

        Returns
        -------
            Matrix with position of cytosines from window and array with their ME values
        """
        matrix, columns = self._jit_compatitable
        return _jit_methylation_entropy(matrix, columns, window_length, min_depth)

    def epipolymorphism(self, window_length: int = 4, min_depth: int = 4):
        """
        Wrapper for _jit_epipolymorphism – JIT compiled PM calculating function

        Parameters
        ----------
        window_length
            Length of the sliding window
        min_depth
            Minimal depth of reads to consider this window for calculation

        Returns
        -------
            Matrix with position of cytosines from window and array with their ME values
        """
        matrix, columns = self._jit_compatitable
        return _jit_epipolymorphism(matrix, columns, window_length, min_depth)

    def PDR(self, min_cyt: int = 5, min_depth: int = 4):
        matrix, columns = self._jit_compatitable
        return _jit_PDR(matrix, columns, min_cyt, min_depth)

class BAMReader:
    def __init__(
            self,
            bamfile: pysam.AlignmentFile,
            bamtype: Literal["bismark"] = "bismark"
    ):
        self.bamfile = bamfile
        self._options = BAMOptions(bamtype)

    @staticmethod
    def _convert_XM(xm_str: str, qual: str, ref_start: int, length: int, converter: dict):
        """
        Returns (Position, Context, Is methylated, Phred quality int)
        """
        out = [(ref_start + shift + 1, xm_str[shift], qual[shift]) for shift in range(length) if xm_str[shift] != "."]
        data = tuple(map(list, zip(*out)))

        return (
                (data[0], ) +
                tuple(zip(*map(converter.get, data[1]))) +
                (list(map(lambda val: ord(val) - 33, data[2])), )
        )

    @staticmethod
    def _convert_orientation(xm: str, xg: str, converter):
        xm_conv, xg_conv = converter[xm], converter[xg]

        if xm_conv and xg_conv:
            return ("+", False)
        if xm_conv and not xg_conv:
            return ("-", False)
        if not xm_conv and xg_conv:
            return ("+", True)
        if not xm_conv and not xg_conv:
            return ("-", True)

    def _parse_alignments(self, alignments):
        calls_dict = self._options.calls_dict
        orientation_dict = self._options.orientation_dict

        region_stats = [
            nuc_row(
                *self._convert_XM(
                    xm_str=alignment.tags[2][1],
                    qual=alignment.qual,
                    ref_start=alignment.reference_start,
                    length=alignment.query_alignment_length,
                    converter=calls_dict
                ),
                alignment.qname,
                *self._convert_orientation(
                    xm=alignment.tags[3][1],
                    xg=alignment.tags[4][1],
                    converter=orientation_dict
                )
            )
            for alignment in alignments
            if len(set(alignment.tags[2][1])) != 1
        ]

        region_df = pl.DataFrame(region_stats)

        if region_df.is_empty():
            return None

        expanded = (
            region_df
            .explode(["position", "context", "m", "qual"])
            .cast(BAMBatch.schema)
        )

        return expanded

    def iter_regions(self, regions):
        for region in regions:
            yield self.region(*region)

    def region(
            self,
            chr: str,
            start: int,
            end: int,
            strand: str = None
    ):
        bam_alignments = self.bamfile.fetch(contig=chr, start=start, end=end)
        methyl_df = self._parse_alignments(bam_alignments)

        if methyl_df is None:
            return None

        if strand is not None:
            filters = (pl.col("strand") == strand) & (pl.col("position") >= start) & (pl.col("position") <= end)
        else:
            filters = (pl.col("position") >= start) & (pl.col("position") <= end)

        methyl_df = methyl_df.filter(filters)

        return BAMBatch(methyl_df)

