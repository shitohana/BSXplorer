from __future__ import annotations

import gzip
import warnings
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.axes import Axes
from pyreadr import write_rds
from scipy import stats

from .UniversalReader_batches import UniversalBatch
from .UniversalReader_classes import UniversalReader
from .utils import remove_extension, CYTOSINE_SUMFUNC, MetageneJoinedSchema, AvailableSumfunc, prepare_labels


class MetageneBase:
    """
    Base class for :class:`Metagene` and plots.
    """

    def __init__(self, report_df: pl.DataFrame, **kwargs):
        """
        Base class for Bismark data.

        :param report_df: pl.DataFrame with cytosine methylation status.
        :param upstream_windows: Number of upstream windows. Required.
        :param gene_windows: Number of gene windows. Required.
        :param downstream_windows: Number of downstream windows. Required.
        :param strand: Strand if filtered.
        :param context: Methylation context if filtered.
        :param plot_data: Data for plotting.
        """
        self.report_df: pl.DataFrame = report_df

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
        write_rds(filename, self.report_df.to_pandas(),
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
                self.report_df.write_csv(file, separator="\t")
        else:
            self.report_df.write_csv(filename, separator="\t")

    @property
    def total_windows(self):
        return self.upstream_windows + self.downstream_windows + self.gene_windows

    @property
    def _x_ticks(self):
        return [
            self.upstream_windows / 2,
            self.upstream_windows,
            self.total_windows / 2,
            self.gene_windows + self.upstream_windows,
            self.total_windows - (self.downstream_windows / 2)
        ]

    @property
    def _borders(self):
        return [self.upstream_windows, self.gene_windows + self.upstream_windows,]

    def __len__(self):
        return len(self.report_df)


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
                [sample.report_df.lazy().with_columns(pl.lit(label))
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
                [sample.report_df.lazy().with_columns(pl.lit(label))
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


def validate_metagene_args(
        genome: pl.DataFrame,
        upstream_windows: int = 0,
        body_windows: int = 2000,
        downstream_windows: int = 0,
        sumfunc: AvailableSumfunc = "wmean",
):
    # VALIDATION
    # Windows
    upstream_windows = upstream_windows if upstream_windows > 0 else 0
    body_windows = body_windows if body_windows > 0 else 0
    downstream_windows = downstream_windows if downstream_windows > 0 else 0
    # Genome
    if not isinstance(genome, pl.DataFrame):
        raise TypeError("Genome must be converted into DataFrame (e.g. via Genome.gene_body()).")
    # Sumfunc
    if sumfunc not in CYTOSINE_SUMFUNC:
        raise NotImplementedError("This summary function is not implemented yet")

    return locals()


def validate_chromosome_args(
        chr_min_length=10 ** 6,
        window_length: int = 10 ** 6,
        confidence: int = None
):
    if chr_min_length < 0:
        warnings.warn("Minimum length of chromosomes should be positive value. Setting it to 0.")
        chr_min_length = 0
    if window_length < 0:
        warnings.warn("Window length should be positive value. Setting it to 1e6.")
        window_length = 10 ** 6
    if confidence is None:
        confidence = 0
    else:
        if not (0 <= confidence < 1):
            warnings.warn("Confidence value needs to be in [0;1) interval, not {}. Disabling confidence bands.")
            confidence = 0

    return locals()


def read_chromosomes(
        reader: UniversalReader,
        chr_min_length=10 ** 6,
        window_length: int = 10 ** 6,
        confidence: float = None,
        **kwargs
):
    # Validate confidence param
    if confidence is None:
        confidence = 0
    else:
        if not (0 <= confidence < 1):
            warnings.warn("Confidence value needs to be in [0;1) interval, not {}. Disabling confidence bands.".format(
                confidence))
            confidence = 0

    def process_batch(df: pl.DataFrame, unfinished_windows_df, last=False):
        last_chr, last_position = df.select("chr", "position").row(-1)
        windows_df = (
            df.with_columns(
                (pl.col("position") / window_length).floor().cast(pl.Int32).alias("window"),
            )
        )

        if unfinished_windows_df is not None:
            windows_df = unfinished_windows_df.vstack(windows_df)

        last_window, = windows_df.filter(chr=last_chr).select("window").row(-1)

        if not last:
            unfinished_windows_df = windows_df.filter(chr=last_chr, window=last_window)
            finished_windows_df = windows_df.filter((pl.col("chr") != last_chr) | (pl.col("window") != last_window))
        else:
            finished_windows_df = windows_df
            unfinished_windows_df = None

        AGG_COLS = [
            pl.sum('density').alias('sum'),
            pl.count('density').alias('count'),
            pl.min("position").alias("start"),
            pl.max("position").alias("end")
        ]
        if confidence is not None and confidence > 0:
            AGG_COLS.append(
                pl.struct(["density", "count_total"])
                .map_elements(
                    lambda x: interval_chr(x.struct.field("density"), x.struct.field("count_total"), confidence))
                .alias("interval")
            )

        finished_group = (
            finished_windows_df
            .group_by(["chr", "strand", "context", "window"])
            .agg(AGG_COLS)
        )

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
    chr_short_list = chr_stats.filter((pl.col("end") - pl.col("start")) < chr_min_length)["chr"].to_list()

    report_df = report_df.filter(~pl.col("chr").is_in(chr_short_list))

    return report_df


def read_metagene(
        reader: UniversalReader,
        genome: pl.DataFrame,
        upstream_windows: int = 0,
        body_windows: int = 2000,
        downstream_windows: int = 0,
        sumfunc: Literal["wmean", "mean", "min", "max", "median", "1pgeom"] = "wmean",
        **kwargs
):
    batch_schema = UniversalBatch.pl_schema()
    genome = (
        genome
        .cast(dict(
            chr=batch_schema["chr"],
            upstream=batch_schema["position"],
            start=batch_schema["position"],
            end=batch_schema["position"],
            downstream=batch_schema["position"],
        ))
        .rename({"strand": "gene_strand"})
    )

    # BATCH SETUP
    def process_batch(df: pl.DataFrame, genome: pl.DataFrame, upstream_windows, body_windows, downstream_windows,
                      sumfunc):
        # POLARS EXPRESSIONS
        # Region position check
        UP_REGION = pl.col('position') < pl.col('start')
        DOWN_REGION = (pl.col('position') > pl.col('end'))

        # Fragment numbers calculation
        # 1e-10 is added (position is always < end)
        UP_FRAGMENT = (((pl.col('position') - pl.col('upstream')) / (
                    pl.col('start') - pl.col('upstream'))) * upstream_windows).floor()
        BODY_FRAGMENT = (((pl.col('position') - pl.col('start')) / (
                    pl.col('end') - pl.col('start') + 1e-10)) * body_windows).floor() + upstream_windows
        DOWN_FRAGMENT = (((pl.col('position') - pl.col('end')) / (pl.col('downstream') - pl.col(
            'end') + 1e-10)) * downstream_windows).floor() + upstream_windows + body_windows

        # Firstly BismarkPlot was written so there were only one sum statistic - mean.
        # Sum and count of densities was calculated for further weighted mean analysis in respect to fragment size
        # For backwards compatibility, for newly introduces statistics, column names are kept the same.
        # Count is set to 1 and "sum" to actual statistics (e.g. median, min, e.t.c)

        AGG_EXPRS = [pl.lit(1).alias("count"), pl.first("gene_strand").alias("strand")]

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
            AGG_EXPRS.append(pl.sum('density').alias('sum'))
            AGG_EXPRS.pop(0)
            AGG_EXPRS.insert(0, pl.count('density').alias("count"))

        GENE_LABEL_COLS = [
            pl.col("chr"),
            pl.concat_str(pl.col("start"), pl.col("end"), separator="-")
        ]

        GROUP_BY_COLS = ['chr', 'start', 'gene', 'context', 'id', 'fragment']

        processed = (
            df.lazy()
            .filter(pl.col("count_total") != 0)
            # Sort by position for joining
            .sort(['chr', 'position'])
            # Join with nearest
            .join_asof(genome.lazy(), left_on='position', right_on='upstream', by='chr', strategy="backward")
            # Limit by end of region
            .filter(pl.col('position') <= pl.col('downstream'))
            # Filter by strand if it is defined
            .filter(((pl.col("gene_strand") != ".") & (pl.col("gene_strand") == pl.col("strand"))) | (
                        pl.col("gene_strand") == "."))
            # Calculate fragment ids
            .with_columns([
                pl.when(UP_REGION).then(UP_FRAGMENT).when(DOWN_REGION).then(DOWN_FRAGMENT).otherwise(
                    BODY_FRAGMENT).alias('fragment'),
                pl.concat_str(GENE_LABEL_COLS, separator=":").alias("gene")
            ])
            # Assign types
            .cast({key: value for key, value in MetageneJoinedSchema.items() if key in GROUP_BY_COLS})
            # gather fragment stats
            .groupby(by=GROUP_BY_COLS)
            # Calculate sumfunc
            .agg(AGG_EXPRS)
            .drop_nulls(subset=['sum'])
            .cast({key: value for key, value in MetageneJoinedSchema.items() if key not in GROUP_BY_COLS})
            .select(list(MetageneJoinedSchema.keys()))
        ).collect()
        return processed

    print("Reading report from", reader.file)
    report_df = None

    for batch in reader:
        processed = process_batch(
            batch.data,
            genome,
            upstream_windows, body_windows, downstream_windows,
            sumfunc
        )

        if report_df is None:
            report_df = processed
        else:
            report_df.extend(processed)

    print("DONE\n")
    return report_df


def interval_chr(sum_density: list[int], sum_counts: list[int], alpha=.95):
    """
    Evaluate confidence interval for point

    :param sum_density: Sums of methylated counts in fragment
    :param sum_counts: Sums of all read cytosines in fragment
    :param alpha: Probability for confidence band
    """
    with np.errstate(invalid='ignore'):
        sum_density, sum_counts = np.array(sum_density), np.array(sum_counts)
        average = sum_density.sum() / len(sum_counts)

        variance = np.average((sum_density - average) ** 2)

        n = sum(sum_counts) - 1

        i = stats.t.interval(alpha, df=n, loc=average, scale=np.sqrt(variance / n))

        return {"lower": i[0], "upper": i[1]}
