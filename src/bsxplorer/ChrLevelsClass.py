from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from pyreadr import write_rds

from .Base import read_chromosomes, validate_chromosome_args
from .Plots import savgol_line, LinePlotData, LinePlot, BoxPlotData, BoxPlot
from .UniversalReader_classes import UniversalReader


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
            df
            .sort(["chr", "window"])
            .group_by(["chr", "window", "context"], maintain_order=True)
            .agg(group_cols)
            .group_by(["chr", "window"], maintain_order=True)
            .agg(pl.all().exclude(["chr", "window"]))
            .with_row_count("fragment")
            .explode(pl.all().exclude(["chr", "window", "fragment"]))
            .with_columns(mut_cols)
        )

    @classmethod
    def from_bismark(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = False,
            confidence: float = None
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
            Size of batch in bytes, which will be read from report file (for report types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """

        reader = UniversalReader(**(locals() | dict(report_type="bismark")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_cgmap(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = False,
            confidence: float = None
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
            Size of batch in bytes, which will be read from report file (for report types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="cgmap")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_bedGraph(
            cls,
            file: str | Path,
            cytosine_file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = False,
            confidence: float = None
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
            Size of batch in bytes, which will be read from report file (for report types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="bedgraph")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_coverage(
            cls,
            file: str | Path,
            cytosine_file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = False,
            confidence: float = None
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
            Size of batch in bytes, which will be read from report file (for report types other than "binom").
        use_threads
            Will reading be multithreaded.
        confidence
            Pvalue for confidence bands of the LinePlot.
        """
        reader = UniversalReader(**(locals() | dict(report_type="coverage")))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    @classmethod
    def from_binom(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            confidence: float = None,
            p_value: float = .05
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
        reader = UniversalReader(**(locals() | dict(report_type="binom", methylation_pvalue=p_value)))
        args = validate_chromosome_args(chr_min_length, window_length, confidence)

        report_df = read_chromosomes(**(locals() | args))

        return cls(report_df)

    def save_plot_rds(self, path, compress: bool = False):
        """
        Saves plot data in a rds DataFrame with columns:

        +----------+---------+
        | fragment | density |
        +==========+=========+
        | Int      | Float   |
        +----------+---------+
        """
        write_rds(path, self.plot_data.to_pandas(),
                  compress="gzip" if compress else None)

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
        context_filter = self.report["context"] == context if context is not None else True
        strand_filter = self.report["strand"] == strand if strand is not None else True
        chr_filter = self.report["chr"] == chr if chr is not None else True

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.report.filter(context_filter & strand_filter & chr_filter))

    @property
    def _ticks_data(self):
        ticks_data = self.plot_data.group_by("chr", maintain_order=True).agg(pl.min("fragment"))

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()
        return x_ticks, x_labels, x_lines

    def line_plot_data(self, smooth: int = 0):
        y = self.plot_data["density"].to_numpy()

        upper, lower = None, None
        if "upper" in self.plot_data.columns and np.isnan(self.plot_data["upper"].to_numpy()).sum() == 0:
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
            self.report
            .group_by(["chr"])
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
