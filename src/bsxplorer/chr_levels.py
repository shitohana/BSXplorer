from __future__ import annotations

from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field, validate_call
from pyreadr import write_rds

from .plots import BoxPlot, BoxPlotData, LinePlot, LinePlotData, savgol_line
from .types import ExistentPath
from .universal_reader import UniversalReader
from .utils import interval_chr


class ChrLevels:
    """
    Read report and visualize chromosome methylation levels
    """

    def __init__(self, df: pl.DataFrame) -> None:
        """
        Read report and visualize chromosome methylation levels

        Parameters
        ----------
        df
        """
        self.report = df

        # delete this in future and change to calculation of plot data
        # when plot is drawn
        self.plot_data = self.__calculate_plot_data(df)

    @staticmethod
    def __calculate_plot_data(df: pl.DataFrame):
        group_cols = [pl.sum("sum"), pl.sum("count"), pl.min("start").alias("chr_pos")]
        mut_cols = [(pl.col("sum") / pl.col("count")).alias("density")]

        if "upper" in df.columns:
            group_cols += [pl.mean("upper"), pl.mean("lower")]

        return (
            df.sort(["chr", "window"])
            .group_by(["chr", "window", "context"], maintain_order=True)
            .agg(group_cols)
            .group_by(["chr", "window"], maintain_order=True)
            .agg(pl.all().exclude(["chr", "window"]))
            .with_row_count("fragment")
            .explode(pl.all().exclude(["chr", "window", "fragment"]))
            .with_columns(mut_cols)
        )

    @classmethod
    def _read_report(
        cls,
        reader: UniversalReader,
        chr_min_length=10**6,
        window_length: int = 10**6,
        confidence: float = None,
    ):
        def process_batch(df: pl.DataFrame, unfinished_windows_df, last=False):
            last_chr, last_position = df.select("chr", "position").row(-1)
            windows_df = df.with_columns(
                (pl.col("position") / window_length)
                .floor()
                .cast(pl.Int32)
                .alias("window"),
            )

            if unfinished_windows_df is not None:
                windows_df = unfinished_windows_df.vstack(windows_df)

            (last_window,) = windows_df.filter(chr=last_chr).select("window").row(-1)

            if not last:
                unfinished_windows_df = windows_df.filter(
                    chr=last_chr, window=last_window
                )
                finished_windows_df = windows_df.filter(
                    (pl.col("chr") != last_chr) | (pl.col("window") != last_window)
                )
            else:
                finished_windows_df = windows_df
                unfinished_windows_df = None

            AGG_COLS = [
                pl.sum("density").alias("sum"),
                pl.count("density").alias("count"),
                pl.min("position").alias("start"),
                pl.max("position").alias("end"),
            ]
            if confidence is not None and confidence > 0:
                AGG_COLS.append(
                    pl.struct(["density", "count_total"])
                    .map_elements(
                        lambda x: interval_chr(
                            x.struct.field("density"),
                            x.struct.field("count_total"),
                            confidence,
                        )
                    )
                    .alias("interval")
                )

            finished_group = finished_windows_df.group_by(
                ["chr", "strand", "context", "window"]
            ).agg(AGG_COLS)

            if confidence is not None and confidence > 0:
                finished_group = finished_group.unnest("interval")

            return finished_group, unfinished_windows_df

        print("Reading report from", reader.file)
        report_df = None
        unfinished = None
        for batch in reader:
            batch.filter_not_none()
            processed, unfinished = process_batch(batch.data, unfinished)

            if report_df is None:
                report_df = processed
            else:
                report_df.extend(processed)

        # Add last unfinished
        report_df.extend(process_batch(unfinished, None, last=True)[0])

        # Filter by chromosome lengths
        chr_stats = report_df.group_by("chr").agg(pl.min("start"), pl.max("end"))
        chr_short_list = chr_stats.filter(
            (pl.col("end") - pl.col("start")) < chr_min_length
        )["chr"].to_list()

        report_df = report_df.filter(~pl.col("chr").is_in(chr_short_list))

        return cls(report_df)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_bismark(
        cls,
        file: ExistentPath,
        chr_min_length: Annotated[int, Field(ge=1_000)] = 1_000_000,
        window_length: Annotated[int, Field(ge=1_000)] = 100_000,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = False,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
    ):
        """
        Initialize ChrLevels with CX_report file

        Parameters
        ----------
        file
            Path to the report file.
        chr_min_length
            Minimum length of chromosome to be analyzed
        window_length
            Length of windows in bp
        block_size_mb
            Size of batch in bytes, which will be read from report file (for report
            types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """

        reader = UniversalReader(**(locals() | dict(report_type="bismark")))
        return cls._read_report(reader, chr_min_length, window_length, confidence)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_cgmap(
        cls,
        file: ExistentPath,
        chr_min_length: Annotated[int, Field(ge=1_000)] = 1_000_000,
        window_length: Annotated[int, Field(ge=1_000)] = 100_000,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = False,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
    ):
        """
        Initialize ChrLevels with CGMap file

        Parameters
        ----------
        file
            Path to the report file.
        chr_min_length
            Minimum length of chromosome to be analyzed
        window_length
            Length of windows in bp
        block_size_mb
            Size of batch in bytes, which will be read from report file (for report
            types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="cgmap")))
        return cls._read_report(reader, chr_min_length, window_length, confidence)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_bedgraph(
        cls,
        file: ExistentPath,
        cytosine_file: ExistentPath,
        chr_min_length: Annotated[int, Field(ge=1_000)] = 1_000_000,
        window_length: Annotated[int, Field(ge=1_000)] = 100_000,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = False,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
    ):
        """
        Initialize ChrLevels with CGMap file

        Parameters
        ----------
        file
            Path to the report file.
        cytosine_file
            Path to preprocessed by :class:`Sequence` cytosine file.
        chr_min_length
            Minimum length of chromosome to be analyzed
        window_length
            Length of windows in bp
        block_size_mb
            Size of batch in bytes, which will be read from report file (for report
            types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="bedgraph")))
        return cls._read_report(reader, chr_min_length, window_length, confidence)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_coverage(
        cls,
        file: ExistentPath,
        cytosine_file: ExistentPath,
        chr_min_length: Annotated[int, Field(ge=1_000)] = 1_000_000,
        window_length: Annotated[int, Field(ge=1_000)] = 100_000,
        block_size_mb: Annotated[int, Field(gt=0)] = 100,
        use_threads: bool = False,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
    ):
        """
        Initialize ChrLevels with CGMap file

        Parameters
        ----------
        file
            Path to the report file.
        cytosine_file
            Path to preprocessed by :class:`Sequence` cytosine file.
        chr_min_length
            Minimum length of chromosome to be analyzed
        window_length
            Length of windows in bp
        block_size_mb
            Size of batch in bytes, which will be read from report file (for report
            types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="coverage")))
        return cls._read_report(reader, chr_min_length, window_length, confidence)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_binom(
        cls,
        file: ExistentPath,
        chr_min_length: Annotated[int, Field(ge=1_000)] = 1_000_000,
        window_length: Annotated[int, Field(ge=1_000)] = 100_000,
        confidence: Annotated[float, Field(ge=0, lt=1)] = 0.0,
        p_value: Annotated[float, Field(gt=0, lt=1)] = 0.05,
    ):
        """
        Initialize ChrLevels with .parquet file from :class:`Binom`.

        Parameters
        ----------
        file
            Path to the report file.
        chr_min_length
            Minimum length of chromosome to be analyzed
        window_length
            Length of windows in bp
        confidence
            Pvalue for confidence bands of the LinePlot.
        p_value
            Pvalue with which cytosine will be considered methylated.
        """
        reader = UniversalReader(
            **(locals() | dict(report_type="binom", methylation_pvalue=p_value))
        )
        return cls._read_report(reader, chr_min_length, window_length, confidence)

    def save_plot_rds(self, path, compress: bool = False):
        """
        Saves plot data in a rds DataFrame with columns:

        +----------+---------+
        | fragment | density |
        +==========+=========+
        | Int      | Float   |
        +----------+---------+
        """
        write_rds(
            path, self.plot_data.to_pandas(), compress="gzip" if compress else None
        )

    def filter(self, context: str = None, strand: str = None, chr: str = None):
        """
        Filter chromosome methylation levels data.

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH) to filter (only one).
        strand
            Strand to filter (+ or -).
        chr
            Chromosome name to filter.

        Returns
        -------
            :class:`ChrLevels`
        """
        context_filter = (
            self.report["context"] == context if context is not None else True
        )
        strand_filter = self.report["strand"] == strand if strand is not None else True
        chr_filter = self.report["chr"] == chr if chr is not None else True

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(
                self.report.filter(context_filter & strand_filter & chr_filter)
            )

    @property
    def _ticks_data(self):
        ticks_data = self.plot_data.group_by("chr", maintain_order=True).agg(
            pl.min("fragment")
        )

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()
        return x_ticks, x_labels, x_lines

    def line_plot_data(self, smooth: int = 0):
        y = self.plot_data["density"].to_numpy()

        upper, lower = None, None
        if (
            "upper" in self.plot_data.columns
            and np.isnan(self.plot_data["upper"].to_numpy()).sum() == 0
        ):
            upper = savgol_line(self.plot_data["upper"].to_numpy(), smooth) * 100
            lower = savgol_line(self.plot_data["lower"].to_numpy(), smooth) * 100

        y = savgol_line(y, smooth) * 100
        x = np.arange(len(y))

        x_ticks, x_labels, x_lines = self._ticks_data
        return LinePlotData(x, y, x_ticks, x_lines, lower, upper, x_labels=x_labels)

    def line_plot(self, smooth: int = 0):
        return LinePlot(self.line_plot_data(smooth))

    def box_plot_data(self):
        pd_df = (
            self.report.group_by(["chr"])
            .agg((pl.col("sum") / pl.col("count")).alias("density"))
            .sort("chr")
        )
        data = pd_df.to_dict()
        chroms = data["chr"]
        values = data["density"]
        return [BoxPlotData(value, chrom) for chrom, value in zip(chroms, values)]

    def box_plot(self):
        """

        Returns
        -------
        BoxPlot
        """
        return BoxPlot(self.box_plot_data())
