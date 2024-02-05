from __future__ import annotations

import os

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyreadr import write_rds
from scipy.signal import savgol_filter
import pyarrow as pa
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

from .utils import interval, decompress, ReportBar
from .ArrowReaders import CsvReader, BismarkOptions, ParquetReader


class ChrBaseReader(ABC):
    def __init__(self,
        file: str | Path,
        chr_min_length=10 ** 6,
        window_length: int = 10 ** 6,
        confidence: int = None
    ):
        self.report_file    = file
        self.chr_min_length = chr_min_length
        self.window_length  = window_length
        self.confidence     = confidence

        self._validate()

    def _validate(self):
        self.report_file = Path(self.report_file).expanduser().absolute()
        if not self.report_file.exists():
            raise FileNotFoundError(self.report_file)

        if self.chr_min_length < 1:
            self.chr_min_length = 0

        if self.window_length < 1:
            raise ValueError(f"Window length must be positive, not {self.window_length}")

        if self.confidence is not None:
            if not (0 <= self.confidence < 1):
                warnings.warn("Confidence value needs to be in [0;1) interval, not {}. Disabling confidence bands.")
                self.confidence = 0
        else:
            self.confidence = 0

    @abstractmethod
    def _get_reader(self) -> CsvReader | ParquetReader:
        ...

    @abstractmethod
    def _mutate_batch(self, batch) -> pl.DataFrame:
        ...

    @abstractmethod
    def _batch_size(self) -> int:
        ...

    @staticmethod
    def __process_batch(df: pl.DataFrame, window_length, chr_min_length, confidence):
        df = (
            df
            .filter((pl.col('count_m') + pl.col('count_um') != 0))
            .group_by(["strand", "chr"])
            # Preprocess
            .agg([
                pl.col("context"),
                (pl.col("position") / window_length).floor().alias("window").cast(pl.Int32),
                ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(pl.Float32),
                (pl.col('count_m') + pl.col('count_um')).alias('count').cast(pl.UInt32),
                (pl.max("position") - pl.min("position")).alias("length")
            ])
            .filter(pl.col("length") > chr_min_length)
        )

        if len(df) == 0:
            return None

        DATA_COLS = [
            pl.sum('density').alias('sum'),
            pl.count('density').alias('count'),
            (pl.first('window') * window_length).alias("chr_pos")
        ]
        if confidence is not None and confidence > 0:
            DATA_COLS.append(
                pl.struct(["density", "count"])
                .map_elements(lambda x: interval(x.struct.field("density"), x.struct.field("count"), confidence))
                .alias("interval")
            )
        df = (
            df
            .explode(["context", "window", "density", "count"])
            .group_by(by=['chr', 'strand', 'context', 'window'])
            .agg(DATA_COLS)
            .drop_nulls(subset=['sum'])
        )

        if confidence is not None and confidence > 0:
            df = df.unnest("interval")

        return df

    def read(self):
        decompressed = decompress(self.report_file)
        if decompressed != self.report_file:
            self.report_file = decompressed.name

        reader = self._get_reader()
        print(f"Reading report from {self.report_file}")

        file_size = os.stat(self.report_file).st_size
        bar = ReportBar(max=file_size)

        report_df = None
        pl.enable_string_cache()

        for batch in reader:
            df = self._mutate_batch(batch)

            processed = self.__process_batch(df, self.window_length, self.chr_min_length, self.confidence)

            if processed is not None:
                if report_df is None:
                    report_df = processed
                else:
                    report_df.extend(processed)

            bar.next(self._batch_size())

        bar.goto(bar.max)
        bar.finish()

        if not isinstance(decompressed, Path):
            decompressed.close()

        return report_df


class ChrBismarkReader(ChrBaseReader):
    def __init__(self, block_size_mb: int = 100, use_threads: bool = True, **kwargs):
        super().__init__(**kwargs)

        self._block_size_mb = block_size_mb
        self._use_threads = use_threads

    def _batch_size(self) -> int:
        return self._block_size_mb * (1024 ** 2)

    def _get_reader(self) -> CsvReader | ParquetReader:
        pool = pa.default_memory_pool()
        block_size = self._batch_size()
        reader = CsvReader(
            self.report_file,
            BismarkOptions(use_threads=self._use_threads, block_size=block_size),
            memory_pool=pool)
        pool.release_unused()
        return reader

    def _mutate_batch(self, batch) -> pl.DataFrame:
        return pl.from_arrow(batch)


class ChrParquetReader(ChrBaseReader):
    def _batch_size(self) -> int:
        return int(self.report_file.stat().st_size / self._get_reader().reader.num_row_groups)

    def _get_reader(self) -> CsvReader | ParquetReader:
        return ParquetReader(self.report_file)

    def _mutate_batch(self, batch) -> pl.DataFrame:
        return pl.from_arrow(batch)


class ChrLevels:
    def __init__(self, df: pl.DataFrame) -> None:
        self.bismark = df

        # delete this in future and change to calculation of plot data
        # when plot is drawn
        self.plot_data = self.__calculate_plot_data(df)

    @staticmethod
    def __calculate_plot_data(df: pl.DataFrame):
        return (
            df
            .sort(["chr", "window"])
            .group_by(["chr", "window"], maintain_order=True)
            .agg([pl.sum("sum"), pl.sum("count"), pl.first("chr_pos")])
            .with_row_count("fragment")
            .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
        )

    @classmethod
    def from_bismark(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            block_size_mb: int = 100,
            use_threads: bool = True,
            confidence: int = None
    ):
        """
        Initialize ChrLevels with CX_report file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        """
        chr_reader = ChrBismarkReader(
            file=file,
            chr_min_length=chr_min_length,
            window_length=window_length,
            block_size_mb=block_size_mb,
            use_threads=use_threads,
            confidence=confidence
        )
        report_df = chr_reader.read()

        return cls(report_df)


    @classmethod
    def from_parquet(
            cls,
            file: str | Path,
            chr_min_length=10 ** 6,
            window_length: int = 10 ** 6,
            confidence: int = None
    ):
        """
        Initialize ChrLevels with CX_report file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        """
        chr_reader = ChrParquetReader(
            file=file,
            chr_min_length=chr_min_length,
            window_length=window_length,
            confidence=confidence
        )

        report_df = chr_reader.read()

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
        :param context: Methylation context (CG, CHG, CHH) to filter (only one).
        :param strand: Strand to filter (+ or -).
        :param chr: Chromosome name to filter.
        :return: Filtered :class:`Bismark`.
        """
        context_filter = self.bismark["context"] == context if context is not None else True
        strand_filter = self.bismark["strand"] == strand if strand is not None else True
        chr_filter = self.bismark["chr"] == chr if chr is not None else True

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.bismark.filter(context_filter & strand_filter & chr_filter))

    @property
    def __ticks_data(self):
        ticks_data = self.plot_data.group_by("chr", maintain_order=True).agg(pl.min("fragment"))

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()
        return x_ticks, x_labels, x_lines

    def __add_flank_lines(self, axes):
        x_ticks, x_labels, x_lines = self.__ticks_data

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)

        for tick in x_lines:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.1)

    def __add_flank_lines_plotly(self, figure: go.Figure):
        x_ticks, x_labels, x_lines = self.__ticks_data

        figure.update_layout(
            xaxis = dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=x_labels)
        )

        for tick in x_lines:
            figure.add_vline(x=tick, line_dash="dash", line_color="rgba(0,0,0,0.2)")

    def draw_mpl(
            self,
            fig_axes: tuple = None,
            smooth: int = 10,
            label: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-'
    ) -> Figure:
        if fig_axes is None:
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        data = self.plot_data["density"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            _, _, lines = self.__ticks_data
            data_ranges = [data[lines[i]: lines[i + 1]] for i in range(len(lines) - 1)]
            data_ranges = [savgol_filter(r, window, 3, mode='nearest') for r in data_ranges]

            data = np.concatenate(data_ranges)

        x = np.arange(len(data))
        data = data * 100  # convert to percents
        axes.plot(x, data, label=label,
                  linestyle=linestyle, linewidth=linewidth)

        axes.legend()
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        self.__add_flank_lines(axes)

        fig.set_size_inches(12, 5)

        return fig

    def draw_plotly(self,
                    figure: tuple = None,
                    smooth: int = 10,
                    label: str = None
                    ):
        if figure is None:
            figure = go.Figure()

        y_data = self.plot_data["density"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            _, _, lines = self.__ticks_data
            lines[-1] = lines[-1] + 1
            data_ranges = [y_data[lines[i]: lines[i + 1]] for i in range(len(lines) - 1)]
            data_ranges = [savgol_filter(r, window, 3, mode='nearest') for r in data_ranges]

            y_data = np.concatenate(data_ranges)

        y_data = y_data * 100  # convert to percents
        plot_df = (
            self.plot_data
            .drop("density")
            .with_columns(pl.lit(y_data).alias("density"))
        )

        # trace = go.Scatter(x=x, y=y_data, mode="lines", name=label)
        trace_fig = px.line(
            plot_df.to_pandas(),
            x="fragment",
            y="density",
            hover_data={
                "fragment": False,
                "chr": True,
                "chr_pos": True,
                "density": ":.4f"
            },
        )



        figure.add_traces(trace_fig.data)

        figure.update_layout(
            xaxis_title="Position",
            yaxis_title="Methylation density, %",
            hovermode="x unified"
        )

        self.__add_flank_lines_plotly(figure)

        return figure
