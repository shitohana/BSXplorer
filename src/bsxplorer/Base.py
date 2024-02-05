from __future__ import annotations

import gzip
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import pyarrow as pa
from pyreadr import write_rds
from matplotlib.axes import Axes
import plotly.graph_objects as go

from .ArrowReaders import CsvReader, ParquetReader, BismarkOptions
from .utils import remove_extension, prepare_labels, MetageneSchema, ReportBar


class MetageneBase:
    """
    Base class for :class:`Metagene` and plots.
    """

    def __init__(self, bismark_df: pl.DataFrame, **kwargs):
        """
        Base class for Bismark data.

        :param bismark_df: pl.DataFrame with cytosine methylation status.
        :param upstream_windows: Number of upstream windows. Required.
        :param gene_windows: Number of gene windows. Required.
        :param downstream_windows: Number of downstream windows. Required.
        :param strand: Strand if filtered.
        :param context: Methylation context if filtered.
        :param plot_data: Data for plotting.
        """
        self.bismark: pl.DataFrame = bismark_df

        self.upstream_windows: int = kwargs.get("upstream_windows")
        self.downstream_windows: int = kwargs.get("downstream_windows")
        self.gene_windows: int = kwargs.get("gene_windows")
        self.plot_data: pl.DataFrame = kwargs.get("plot_data")
        self.context: str = kwargs.get("context")
        self.strand: str = kwargs.get("strand")

    @property
    def metadata(self) -> dict:
        """
        :return: Bismark metadata in dict
        """
        return {
            "upstream_windows": self.upstream_windows,
            "downstream_windows": self.downstream_windows,
            "gene_windows": self.gene_windows,
            "plot_data": self.plot_data,
            "context": self.context,
            "strand": self.strand
        }

    def save_rds(self, filename, compress: bool = False):
        """
        Save Metagene in Rds.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        write_rds(filename, self.bismark.to_pandas(),
                  compress="gzip" if compress else None)

    def save_tsv(self, filename, compress=False):
        """
        Save Metagene in TSV.

        :param filename: Path for file.
        :param compress: Whether to compress to gzip or not.
        """
        if compress:
            with gzip.open(filename + ".gz", "wb") as file:
                # noinspection PyTypeChecker
                self.bismark.write_csv(file, separator="\t")
        else:
            self.bismark.write_csv(filename, separator="\t")

    @property
    def total_windows(self):
        return self.upstream_windows + self.downstream_windows + self.gene_windows

    @property
    def tick_positions(self):
        return dict(
            up_mid=self.upstream_windows / 2,
            body_start=self.upstream_windows,
            body_mid=self.total_windows / 2,
            body_end=self.gene_windows + self.upstream_windows,
            down_mid=self.total_windows - (self.downstream_windows / 2)
        )

    def __len__(self):
        return len(self.bismark)


class MetageneFilesBase:
    def __init__(self, samples, labels: list[str] = None):
        self.samples = self.__check_metadata(samples if isinstance(samples, list) else [samples])

        if samples is None:
            raise Exception("Flank or gene windows number does not match!")

        self.labels = list(map(str, range(len(samples)))) if labels is None else labels

        if len(self.labels) != len(self.samples):
            raise Exception("Labels length doesn't match samples number")

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
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
            )
            write_rds(base_filename, merged.to_pandas(),
                      compress="gzip" if compress else None)
        if not merge:
            for sample, label in zip(self.samples, self.labels):
                sample.save_rds(
                    f"{remove_extension(base_filename)}_{label}.rds", compress="gzip" if compress else None)

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
                [sample.bismark.lazy().with_columns(pl.lit(label))
                 for sample, label in zip(self.samples, self.labels)]
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
                    f"{remove_extension(base_filename)}_{label}.tsv", compress=compress)

    @staticmethod
    def __check_metadata(samples: list[MetageneBase]):
        upstream_check = set([sample.metadata["upstream_windows"]
                             for sample in samples])
        downstream_check = set(
            [sample.metadata["downstream_windows"] for sample in samples])
        gene_check = set([sample.metadata["gene_windows"]
                         for sample in samples])

        if len(upstream_check) == len(gene_check) == len(downstream_check) == 1:
            return samples
        else:
            raise ValueError("Different windows number between samples")


class PlotBase(MetageneBase):
    def flank_lines(self, axes: Axes, major_labels: list, minor_labels: list, show_border=True):
        labels = prepare_labels(major_labels, minor_labels)

        if self.downstream_windows < 1:
            labels["down_mid"], labels["body_end"] = [""] * 2

        if self.upstream_windows < 1:
            labels["up_mid"], labels["body_start"] = [""] * 2

        ticks = self.tick_positions

        names = list(ticks.keys())
        x_ticks = [ticks[key] for key in names]
        x_labels = [labels[key] for key in names]

        axes.set_xticks(x_ticks, labels=x_labels)

        if show_border:
            for tick in [ticks["body_start"], ticks["body_end"]]:
                axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)

        return axes

    def _merge_strands(self, df: pl.DataFrame):
        return df.filter(pl.col("strand") == "+").extend(self._strand_reverse(df.filter(pl.col("strand") == "-")))

    @staticmethod
    def _strand_reverse(df: pl.DataFrame):
        max_fragment = df["fragment"].max()
        return df.with_columns((max_fragment - pl.col("fragment")).alias("fragment")).sort("fragment")

    def flank_lines_plotly(self, figure: go.Figure, major_labels: list, minor_labels: list, show_border=True):
        """
        Add flank lines to the given axis (for line plot)
        """
        labels = prepare_labels(major_labels, minor_labels)

        if self.downstream_windows < 1:
            labels["down_mid"], labels["body_end"] = [""] * 2

        if self.upstream_windows < 1:
            labels["up_mid"], labels["body_start"] = [""] * 2

        ticks = self.tick_positions

        names = list(ticks.keys())
        x_ticks = [ticks[key] for key in names]
        x_labels = [labels[key] for key in names]

        figure.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=x_labels)
        )

        if show_border:
            for tick in [ticks["body_start"], ticks["body_end"]]:
                figure.add_vline(x=tick, line_dash="dash", line_color="rgba(0,0,0,0.2)")

        return figure


class ReportReader(ABC):
    def __init__(
            self,
            report_file: str | Path,
            genome: pl.DataFrame,
            upstream_windows: int = 0,
            body_windows: int = 2000,
            downstream_windows: int = 0,
            use_threads: bool = True,
            sumfunc: str = "mean"
    ):
        self.report_file        = report_file
        self.genome             = genome
        self.upstream_windows   = upstream_windows
        self.body_windows       = body_windows
        self.downstream_windows = downstream_windows
        self.use_threads        = use_threads
        self.sumfunc            = sumfunc
        self.temp_file          = None

        self.validate()

    def validate(self):
        # Windows
        self.upstream_windows   = self.upstream_windows if self.upstream_windows > 0 else 0
        self.body_windows       = self.body_windows if self.body_windows > 0 else 0
        self.downstream_windows = self.downstream_windows if self.downstream_windows > 0 else 0

        # Genome
        if not isinstance(self.genome, pl.DataFrame):
            raise TypeError("Genome must be converted into DataFrame (e.g. via Genome.gene_body()).")

        # Report file
        self.report_file = Path(self.report_file).expanduser().absolute()

        if not self.report_file.exists():
            raise FileNotFoundError(f"Report file: {self.report_file} – not found!")

        self.report_file = self.__decompress(self.report_file)

        # todo add sumfunc validator

    def __decompress(self, path: str | Path) -> Path:
        if path.suffix == ".gz":
            temp_file = tempfile.NamedTemporaryFile()
            print(f"Temporarily unpack {path} to {temp_file.name}")

            with gzip.open(path, mode="rb") as file:
                shutil.copyfileobj(file, temp_file)

            self.temp_file = temp_file
            return Path(temp_file.name)
        else:
            return path

    @abstractmethod
    def get_reader(self) -> CsvReader | ParquetReader:
        ...

    @abstractmethod
    def mutate_batch(self, batch) -> pl.DataFrame:
        ...

    @abstractmethod
    def batch_size(self) -> int:
        ...

    @staticmethod
    def __process_batch(df: pl.DataFrame, genome: pl.DataFrame, up_win, body_win, down_win, sumfunc):
        # *** POLARS EXPRESSIONS ***
        # cast genome columns to type to join
        GENE_COLUMNS = [
            pl.col('strand').cast(MetageneSchema["strand"]),
            pl.col('chr').cast(MetageneSchema["chr"])
        ]
        # upstream region position check
        UP_REGION = pl.col('position') < pl.col('start')
        # body region position check
        BODY_REGION = (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
        # downstream region position check
        DOWN_REGION = (pl.col('position') > pl.col('end'))

        UP_FRAGMENT = (((pl.col('position') - pl.col('upstream')) / (
                    pl.col('start') - pl.col('upstream'))) * up_win).floor()
        # fragment even for position == end needs to be rounded by floor
        # so 1e-10 is added (position is always < end)
        BODY_FRAGMENT = (((pl.col('position') - pl.col('start')) / (
                    pl.col('end') - pl.col('start') + 1e-10)) * body_win).floor() + up_win
        DOWN_FRAGMENT = (((pl.col('position') - pl.col('end')) / (
                    pl.col('downstream') - pl.col('end') + 1e-10)) * down_win).floor() + up_win + body_win

        # Firstly BismarkPlot was written so there were only one sum statistic - mean.
        # Sum and count of densities was calculated for further weighted mean analysis in respect to fragment size
        # For backwards compatibility, for newly introduces statistics, column names are kept the same.
        # Count is set to 1 and "sum" to actual statistics (e.g. median, min, e.t.c)
        if sumfunc == "median":
            AGG_EXPR = [pl.median("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "min":
            AGG_EXPR = [pl.min("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "max":
            AGG_EXPR = [pl.max("density").alias("sum"), pl.lit(1).alias("count")]
        elif sumfunc == "geometric":
            AGG_EXPR = [pl.col("density").log().mean().exp().alias("sum"),
                        pl.lit(1).alias("count")]
        elif sumfunc == "1pgeometric":
            AGG_EXPR = [(pl.col("density").log1p().mean().exp() - 1).alias("sum"),
                        pl.lit(1).alias("count")]
        else:
            AGG_EXPR = [pl.sum('density').alias('sum'), pl.count('density').alias('count')]

        processed = (
            df.lazy()
            # assign types
            # calculate density for each cytosine
            .with_columns([
                pl.col('position').cast(MetageneSchema["position"]),
                pl.col('chr').cast(MetageneSchema["chr"]),
                pl.col('strand').cast(MetageneSchema["strand"]),
                pl.col('context').cast(MetageneSchema["context"]),
            ])
            # sort by position for joining
            .sort(['chr', 'strand', 'position'])
            # join with nearest
            .join_asof(
                genome.lazy().with_columns(GENE_COLUMNS),
                left_on='position', right_on='upstream', by=['chr', 'strand']
            )
            # limit by end of region
            .filter(pl.col('position') <= pl.col('downstream'))
            # calculate fragment ids
            .with_columns([
                pl.when(UP_REGION).then(UP_FRAGMENT)
                .when(BODY_REGION).then(BODY_FRAGMENT)
                .when(DOWN_REGION).then(DOWN_FRAGMENT)
                .alias('fragment'),
                pl.concat_str([
                    pl.col("chr"),
                    (pl.concat_str(pl.col("start"), pl.col("end"), separator="-"))
                ], separator=":").alias("gene")
            ])
            .with_columns([
                pl.col("fragment").cast(MetageneSchema["fragment"]),
                pl.col("gene").cast(MetageneSchema["gene"]),
                pl.col('id').cast(MetageneSchema["id"])
            ])
            # gather fragment stats
            .groupby(by=['chr', 'strand', 'start', 'gene', 'context', 'id', 'fragment'])
            .agg(AGG_EXPR)
            .drop_nulls(subset=['sum'])
        ).collect()
        return processed

    def read(self):
        print("Initializing report reader.")
        reader = self.get_reader()

        file_size = self.report_file.stat().st_size

        bar = ReportBar(max=file_size)

        print("Reading report from", self.report_file)
        report_df = None
        pl.enable_string_cache()

        for batch in reader:
            batch_df = self.mutate_batch(batch)
            processed = self.__process_batch(
                batch_df,
                self.genome,
                self.upstream_windows, self.body_windows, self.downstream_windows,
                self.sumfunc
            )

            if report_df is None:
                report_df = processed
            else:
                report_df.extend(processed)

            bar.next(self.batch_size())

        bar.goto(bar.max)
        bar.finish()
        print("DONE\n")

        if self.temp_file is not None:
            self.temp_file.close()

        return report_df


class BismarkReportReader(ReportReader):
    def __init__(self, block_size_mb: int = 50, **kwargs):
        self.block_size_mb = block_size_mb

        super().__init__(**kwargs)

    def get_reader(self) -> CsvReader | ParquetReader:
        pool = pa.default_memory_pool()
        reader = CsvReader(
            self.report_file,
            BismarkOptions(use_threads=self.use_threads,
                           block_size=self.batch_size()),
            memory_pool=pool
        )
        pool.release_unused()

        return reader

    def mutate_batch(self, batch) -> pl.DataFrame:
        mutated = (
            pl
            .from_arrow(batch)
            .filter((pl.col('count_m') + pl.col('count_um') != 0))
            .with_columns(
                ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density')
                .cast(MetageneSchema.sum)
            )
        )

        return mutated

    def batch_size(self):
        return self.block_size_mb * 1024**2


class ParquetReportReader(ReportReader):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def get_reader(self) -> CsvReader | ParquetReader:
        reader = ParquetReader(self.report_file, use_threads=self.use_threads)
        return reader

    def mutate_batch(self, batch) -> pl.DataFrame:
        # todo add metadata identifier for suitable file

        mutated = (
            batch
            .from_arrow(batch)
            .filter(pl.col("count_total") != 0)
            .with_columns(
                (pl.col('count_m') / pl.col('count_total')).alias('density').cast(MetageneSchema.sum)
            )
            .drop("count_total")
        )

        return mutated

    def batch_size(self):
        row_groups = self.get_reader().reader.num_row_groups
        file_size = self.report_file.stat().st_size

        return  int(file_size / row_groups)


class BinomReportReader(ReportReader):
    def __init__(self, p_value: float = .05, **kwargs):
        self.p_value = p_value

        super().__init__(**kwargs)

    def get_reader(self) -> CsvReader | ParquetReader:
        reader = ParquetReader(self.report_file, use_threads=self.use_threads)
        return reader

    def mutate_batch(self, batch) -> pl.DataFrame:
        # todo add metadata identifier for suitable file

        mutated = (
            pl.from_arrow(batch)
            .with_columns((pl.col("p_value") < self.p_value).cast(pl.Float32).alias("density"))
            .drop("count_total")
        )

        return mutated

    def batch_size(self):
        row_groups = self.get_reader().reader.num_row_groups
        file_size = self.report_file.stat().st_size

        return int(file_size / row_groups)