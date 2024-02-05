from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import polars as pl
import progress.bar
import pyarrow as pa
from pyarrow import parquet as pq
from scipy.stats import binom

from .ArrowReaders import BismarkOptions, CsvReader, ParquetReader


class BinomialData:
    """
    Calculates P-value for cytosine residues.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def preprocess(
            cls,
            file: str | Path,
            report_type: Literal["bismark"] = "bismark",
            name: str | Path = None,
            min_coverage: int = 2,
            block_size_mb: int = 20,
            use_threads: bool = True,
            dir: str | Path = "./"
    ):
        """
        Method to preprocess BS-seq cytosine report data by calculating methylation P-value for every cytosine (assuming distribution is binomial) that passes `min_coverage` threshold.

        Parameters
        ----------
        file
            Path to cytosine report.
        report_type
            Type of report. Possible types: bismark.
        name
            Name with which preprocessed file will be saved. If not provided - input file name is being used.
        min_coverage
            Minimal coverage for cytosine.
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        dir
            Path to working dir, where file will be saved.

        Returns
        -------
        BinomialData
            Instance of Binom class.
        """
        file = Path(file).expanduser().absolute()
        if not file.exists(): raise FileNotFoundError()

        block_size = (1024 ** 2) * block_size_mb

        metadata = dict(
            cytosine_residues=0,
            density_sum=0
        )

        # Reading and calculating total stats
        print("Reading from", file.absolute())

        file_size = os.stat(file.absolute()).st_size
        bar = cls.__bar("Reading cytosines", file_size)

        if report_type == "bismark":
            options = BismarkOptions(use_threads=use_threads, block_size=block_size)

            get_reader = lambda: CsvReader(file, options)
        # todo check with another parquet...
        if report_type == "parquet":
            get_reader = lambda: ParquetReader(file)

        with get_reader() as reader:
            for batch in reader:
                # Update metadata with total distribution stats
                formatted = cls.__formatters["stats"][report_type](batch)
                metadata["cytosine_residues"] += len(batch)
                metadata["density_sum"] += formatted["density"].sum()

                bar.next(block_size)
            bar.goto(file_size)
            bar.finish()

        # Calculating total p_values
        save_path = Path(dir) / ((file.stem if name is None else name) + ".parquet")
        print("Writing p_values file into:", save_path.absolute())
        total_probability = metadata["density_sum"] / metadata["cytosine_residues"]

        with pq.ParquetWriter(save_path, cls.__schemas["arrow"]["p_value"]) as pq_writer:
            with get_reader() as reader:
                bar = cls.__bar("Calculating p-values", file_size)
                for batch in reader:
                    # Calculate density for each cytosine despite its methylation context.
                    formatted = (
                        cls.__formatters["p_value"][report_type](batch)
                        # Filter by coverage.
                        .filter(pl.col("total") >= min_coverage)
                    )
                    # Binomial test for cytosine methylation
                    cdf = 1 - binom.cdf(formatted["count_m"] - 1, formatted["total"], total_probability)
                    # Write to p_valued file
                    p_valued = (
                        formatted.with_columns(pl.lit(cdf).cast(pl.Float64).alias("p_value"))
                        .select(cls.__schemas["arrow"]["p_value"].names)
                        .cast(cls.__schemas["polars"]["p_value"])
                    )
                    pq_writer.write(
                        p_valued
                        .to_arrow()
                        .cast(cls.__schemas["arrow"]["p_value"])
                    )
                    bar.next(block_size)
            bar.goto(file_size)
            bar.finish()
        print("Done")

        print(f"\nTotal cytosine residues: {metadata['cytosine_residues']}.\nAverage proportion of methylated reads to total reads for cytosine residue: {metadata['density_sum'] / metadata['cytosine_residues']}")

        return cls(save_path)

    def region_pvalue(
            self,
            genome: pl.DataFrame,
            methylation_pvalue: float = .05,
            use_threads: bool = True,
            save: bool = False,
            save_path: str = None
    ):
        """
        Map cytosines with provided annotation and calculate region methylation P-value (assuming distribution is binomial).

        Parameters
        ----------
        genome
            DataFrame with annotation (e.g. from `Genome` class)
        methylation_pvalue
            P-value of cytosine methylation for it to be considered methylated.
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        save
            Does processed file need to be saved as BED-like TSV with gene methylation stats.
        save_path
            Path, where file needs to be saved (If save option is set True).

        Returns
        -------
        RegionStat
            Instance of RegionStat class.

        Examples
        --------
        If there no preprocessed file:

        >>> import source as bp
        >>> report_path = "/path/to/report.txt"
        >>> genome_path = "/path/to/genome.gff"
        >>> c_binom = bp.BinomialData.preprocess(report_path, report_type="bismark")

        >>> genome = bp.Genome.from_gff(genome_path).gene_body()
        >>> region_stats = c_binom.region_pvalue(genome)
        >>> region_stats
        shape: (3, 11)
        ┌─────────┬────────┬─────────┬───────┬───────┬────────┬────────┬────────┬────────┬────────┬────────┐
        │ chr     ┆ strand ┆ id      ┆ start ┆ end   ┆ p_valu ┆ p_valu ┆ p_valu ┆ total  ┆ total  ┆ total  │
        │ ---     ┆ ---    ┆ ---     ┆ ---   ┆ ---   ┆ e_cont ┆ e_cont ┆ e_cont ┆ contex ┆ contex ┆ contex │
        │ cat     ┆ cat    ┆ str     ┆ u64   ┆ u64   ┆ ext_CG ┆ ext_CH ┆ ext_CH ┆ t_CG   ┆ t_CHG  ┆ t_CHH  │
        │         ┆        ┆         ┆       ┆       ┆ ---    ┆ G      ┆ H      ┆ ---    ┆ ---    ┆ ---    │
        │         ┆        ┆         ┆       ┆       ┆ f64    ┆ ---    ┆ ---    ┆ i64    ┆ i64    ┆ i64    │
        │         ┆        ┆         ┆       ┆       ┆        ┆ f64    ┆ f64    ┆        ┆        ┆        │
        ╞═════════╪════════╪═════════╪═══════╪═══════╪════════╪════════╪════════╪════════╪════════╪════════╡
        │ NC_0030 ┆ +      ┆ gene-AT ┆ 3631  ┆ 5899  ┆ 1.0    ┆ 1.0    ┆ 1.0    ┆ 60     ┆ 82     ┆ 251    │
        │ 70.9    ┆        ┆ 1G01010 ┆       ┆       ┆        ┆        ┆        ┆        ┆        ┆        │
        │ NC_0030 ┆ -      ┆ gene-AT ┆ 6788  ┆ 9130  ┆ 0.9992 ┆ 1.0    ┆ 1.0    ┆ 31     ┆ 55     ┆ 295    │
        │ 70.9    ┆        ┆ 1G01020 ┆       ┆       ┆ 65     ┆        ┆        ┆        ┆        ┆        │
        │ NC_0030 ┆ +      ┆ gene-AT ┆ 11101 ┆ 11372 ┆ 1.0    ┆ 1.0    ┆ 1.0    ┆ 1      ┆ 8      ┆ 43     │
        │ 70.9    ┆        ┆ 1G03987 ┆       ┆       ┆        ┆        ┆        ┆        ┆        ┆        │
        └─────────┴────────┴─────────┴───────┴───────┴────────┴────────┴────────┴────────┴────────┴────────┘

        If preprocessed file exists:

        >>> preprocessed_path = "/path/to/preprocessed.parquet"
        >>> c_binom = bp.BinomialData(preprocessed_path)
        >>> region_stats = c_binom.region_pvalue(genome)
        """

        # Format genome
        pl.enable_string_cache()
        genome = genome.cast({k: v for k, v in self.__schemas["polars"]["p_value"].items() if k in genome.columns})

        metadata = pl.DataFrame(schema={"context":    self.__schemas["polars"]["p_value"]["context"],
                                        "methylated": pl.Int64,
                                        "total":      pl.Int64})

        gene_stats = None

        print("From", self.path)
        with ParquetReader(self.path, use_threads=use_threads) as pq_reader:
            bar = self.__bar("Reading p_values", max=len(pq_reader))

            for batch in pq_reader:
                methyl = (
                    pl.from_arrow(batch)
                    .with_columns(
                        (pl.col("p_value") <= methylation_pvalue).alias("methylated").cast(pl.Boolean)
                    )
                    .drop("p_value")
                )

                metadata = metadata.extend(
                    methyl.group_by("context").agg([
                        pl.col("methylated").sum().cast(pl.Int64),
                        pl.count("position").alias("total").cast(pl.Int64)
                    ])
                )

                mapped = (
                    methyl.lazy()
                    .join_asof(genome.lazy(),
                               left_on="position", right_on="start",
                               strategy="backward",
                               by=["chr", "strand"])
                    .filter(pl.col("position") <= pl.col("end"))
                    .filter(pl.col("start").is_not_nan())
                    .group_by(["chr", "strand", "start", "end", "id", "context"], maintain_order=True)
                    .agg([
                        pl.col("methylated").sum().cast(pl.Int64),
                        pl.count("position").alias("total").cast(pl.Int64)
                    ])
                    .collect()
                )

                if gene_stats is None: gene_stats = mapped
                else: gene_stats.extend(mapped)

                bar.next()
            bar.finish()

        metadata = metadata.group_by("context").agg([pl.sum("methylated"), pl.sum("total")])

        result = pl.DataFrame(schema=gene_stats.schema | {"p_value": pl.Float64})
        for row in metadata.iter_rows(named=True):
            filtered = gene_stats.filter(pl.col("context") == row["context"])
            total_probability = row["methylated"] / row["total"]

            cdf = 1 - binom.cdf(filtered["methylated"] - 1, filtered["total"], total_probability)
            result.extend(
                filtered.with_columns(pl.lit(cdf, pl.Float64).alias("p_value"))
            )

            print(f"{row['context']}\tTotal sites: {row['total']}, methylated: {row['methylated']}\t({round(total_probability * 100, 2)}%)")
        result = result.select(["chr", "start", "end", "id", "p_value", "strand", "context", "total"])

        if save:
            if save_path is not None:
                save_path = Path(save_path)
            else:
                save_path = self.path.parent / (self.path.stem + "_genes.tsv")
            if save_path.suffix == "":
                save_path = save_path.with_suffix(".tsv")

            result.write_csv(save_path.absolute(), has_header=False, separator="\t")

            print("Saved into:", save_path)

        return RegionStat.from_expanded(result)

    __formatters = {
        "stats": {
            "bismark": lambda batch:
                (
                    pl.from_arrow(batch).lazy()
                    .with_columns([
                        (pl.col("count_m") / (pl.col("count_m") + pl.col("count_um")))
                        .cast(pl.Float64)
                        .alias("density")
                    ])
                    .filter(pl.col("density").is_not_nan())
                ).collect(),
            "parquet": lambda batch:
                (
                    pl.from_arrow(batch).lazy()
                    .with_columns([
                        (pl.col("count_m") / (pl.col("count_m") + pl.col("count_um")))
                        .cast(pl.Float64)
                        .alias("density")
                    ])
                    .filter(pl.col("density").is_not_nan())
                ).collect(),
        },
        "p_value": {
            "bismark": lambda batch:
                (
                    pl.from_arrow(batch).lazy()
                    .with_columns([
                        (pl.col("count_m") + pl.col("count_um")).cast(pl.Int32).alias("total"),
                        pl.col("count_m").cast(pl.Int32),
                    ])
                ).collect(),
            "parquet": lambda batch:
            (
                pl.from_arrow(batch).lazy()
                .with_columns([
                    (pl.col("count_m") + pl.col("count_um")).cast(pl.Int32).alias("total"),
                    pl.col("count_m").cast(pl.Int32),
                ])
            ).collect()
        }
    }

    __schemas = {
        "polars": {
            "p_value": dict(
                chr=pl.Categorical,
                strand=pl.Categorical,
                position=pl.UInt64,
                context=pl.Categorical,
                p_value=pl.Float64
            )
        },
        "arrow": {
            "p_value": pa.schema([
                ("chr", pa.dictionary(pa.int16(), pa.utf8())),
                ("strand", pa.dictionary(pa.int8(), pa.utf8())),
                ("position", pa.uint64()),
                ("context", pa.dictionary(pa.int8(), pa.utf8())),
                ("p_value", pa.float64()),
            ])
        }

    }

    @staticmethod
    def __bar(name, max):
        suffix = "%(percent).1f%%"
        return progress.bar.Bar(name, suffix=suffix, max=max)


# TODO add CSV init support
class RegionStat:
    """
    Class for manipulation with region P-value data.

    Attributes
    ----------
    region_stats : polars.DataFrame
        Region stats DataFrame
    """
    def __init__(self, region_stats: pl.DataFrame = None):
        """
        Class for manipulation with region P-value data.

        Warnings
        --------
        Do not call this method directly.

        Parameters
        ----------
        region_stats : polars.DataFrame
            Region stats DataFrame
        """

        self.region_stats = region_stats

    @classmethod
    def from_expanded(cls, df: pl.DataFrame):
        """
        Generate Instance of RegionStat class from DataFrame with context column expanded (e.g. output of `Binom.gene_pvalue()` function)

        Parameters
        ----------
        df
            DataFrame with context column expanded

        Returns
        -------
        RegionStat
            Instance of RegionStat class
        """
        gene_stats = (
            df
            .filter(df.select(["chr", "strand", "start", "end", "context"]).is_duplicated().not_())
            .pivot(values=["p_value", "total"],
                   columns="context",
                   index=list(set(df.columns) - set(["p_value", "context", "total"])))
        )
        return cls(gene_stats)

    def filter(
            self,
            context: Literal["CG", "CHG", "CHH"] = None,
            op: Literal["<=", "<", ">", ">="] = None,
            p_value: float = .05,
            min_n: int = 0
    ):
        """
        Filter RegionStat class by P-value of methylation in selected context or minimal region counts.

        E.g. :math:`P_{CG}\leq0.05` or :math:`N_{CG}\geq20`

        Minimal counts are compared only with :math:`\geq` operation.

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH).
        op
            Comparative operation (<=, <, >, >=).
        p_value
            P-value for operation.
        min_n
            Minimal counts for cytosines methylated in selected context.

        Returns
        -------
        RegionStat
            Filtered class.

        Examples
        --------
        >>> region_stats = BinomialData("./A_thaliana.parquet").region_pvalue(genome)
        >>> region_stats
        shape: (3, 11)
        ┌─────────┬────────┬─────────┬───────┬───────┬────────┬────────┬────────┬────────┬────────┬────────┐
        │ chr     ┆ strand ┆ id      ┆ start ┆ end   ┆ p_valu ┆ p_valu ┆ p_valu ┆ total_ ┆ total_ ┆ total_ │
        │ ---     ┆ ---    ┆ ---     ┆ ---   ┆ ---   ┆ e_cont ┆ e_cont ┆ e_cont ┆ contex ┆ contex ┆ contex │
        │ cat     ┆ cat    ┆ str     ┆ u64   ┆ u64   ┆ ext_CG ┆ ext_CH ┆ ext_CH ┆ t_CG   ┆ t_CHG  ┆ t_CHH  │
        │         ┆        ┆         ┆       ┆       ┆ ---    ┆ G      ┆ H      ┆ ---    ┆ ---    ┆ ---    │
        │         ┆        ┆         ┆       ┆       ┆ f64    ┆ ---    ┆ ---    ┆ i64    ┆ i64    ┆ i64    │
        │         ┆        ┆         ┆       ┆       ┆        ┆ f64    ┆ f64    ┆        ┆        ┆        │
        ╞═════════╪════════╪═════════╪═══════╪═══════╪════════╪════════╪════════╪════════╪════════╪════════╡
        │ NC_0030 ┆ +      ┆ gene-AT ┆ 3631  ┆ 5899  ┆ 1.0    ┆ 1.0    ┆ 1.0    ┆ 60     ┆ 82     ┆ 251    │
        │ 70.9    ┆        ┆ 1G01010 ┆       ┆       ┆        ┆        ┆        ┆        ┆        ┆        │
        │ NC_0030 ┆ -      ┆ gene-AT ┆ 6788  ┆ 9130  ┆ 0.9992 ┆ 1.0    ┆ 1.0    ┆ 31     ┆ 55     ┆ 295    │
        │ 70.9    ┆        ┆ 1G01020 ┆       ┆       ┆ 65     ┆        ┆        ┆        ┆        ┆        │
        │ NC_0030 ┆ +      ┆ gene-AT ┆ 11101 ┆ 11372 ┆ 1.0    ┆ 1.0    ┆ 1.0    ┆ 1      ┆ 8      ┆ 43     │
        │ 70.9    ┆        ┆ 1G03987 ┆       ┆       ┆        ┆        ┆        ┆        ┆        ┆        │
        └─────────┴────────┴─────────┴───────┴───────┴────────┴────────┴────────┴────────┴────────┴────────┘
        >>> region_stats.filter("CG", "<", 0.05, 20)
        shape: (3, 11)
        ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
        │ chr    ┆ strand ┆ id     ┆ start  ┆ end    ┆ p_valu ┆ p_valu ┆ p_valu ┆ total_ ┆ total_ ┆ total_ │
        │ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ e_cont ┆ e_cont ┆ e_cont ┆ contex ┆ contex ┆ contex │
        │ cat    ┆ cat    ┆ str    ┆ u64    ┆ u64    ┆ ext_CG ┆ ext_CH ┆ ext_CH ┆ t_CG   ┆ t_CHG  ┆ t_CHH  │
        │        ┆        ┆        ┆        ┆        ┆ ---    ┆ G      ┆ H      ┆ ---    ┆ ---    ┆ ---    │
        │        ┆        ┆        ┆        ┆        ┆ f64    ┆ ---    ┆ ---    ┆ i64    ┆ i64    ┆ i64    │
        │        ┆        ┆        ┆        ┆        ┆        ┆ f64    ┆ f64    ┆        ┆        ┆        │
        ╞════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╡
        │ NC_003 ┆ +      ┆ gene-A ┆ 23121  ┆ 31227  ┆ 9.9920 ┆ 1.0    ┆ 1.0    ┆ 123    ┆ 171    ┆ 604    │
        │ 070.9  ┆        ┆ T1G010 ┆        ┆        ┆ e-16   ┆        ┆        ┆        ┆        ┆        │
        │        ┆        ┆ 40     ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        │
        │ NC_003 ┆ -      ┆ gene-A ┆ 121067 ┆ 130577 ┆ 0.0048 ┆ 1.0    ┆ 1.0    ┆ 233    ┆ 338    ┆ 1169   │
        │ 070.9  ┆        ┆ T1G013 ┆        ┆        ┆ 71     ┆        ┆        ┆        ┆        ┆        │
        │        ┆        ┆ 20     ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        │
        │ NC_003 ┆ -      ┆ gene-A ┆ 192634 ┆ 193670 ┆ 0.0005 ┆ 0.2612 ┆ 0.9915 ┆ 20     ┆ 22     ┆ 143    │
        │ 070.9  ┆        ┆ T1G015 ┆        ┆        ┆ 71     ┆ 87     ┆ 5      ┆        ┆        ┆        │
        │        ┆        ┆ 30     ┆        ┆        ┆        ┆        ┆        ┆        ┆        ┆        │
        └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
        """
        if context is not None and op is not None:
            if op == "<":
                expr = pl.col(f"p_value_context_{context}") < p_value
            elif op == "<=":
                expr = pl.col(f"p_value_context_{context}") <= p_value
            elif op == ">=":
                expr = pl.col(f"p_value_context_{context}") >= p_value
            else:
                expr = pl.col(f"p_value_context_{context}") > p_value
        else:
            expr = True

        return self.__class__(
            self.region_stats
                .filter(expr)
                .filter(pl.col(f"total_context_{context}") >= min_n)
        )

    def save(self, path: str | Path):
        """
        Save regions as BED-like file.

        Parameters
        ----------
        path
            Path where file needs to be saved.
        """
        path = Path(path)

        (
            self.region_stats
            .select(["chr", "start", "end", "id", "strand"])
            .write_csv(path, has_header=False, separator="\t")
        )

    def categorise(
            self,
            context: Literal["CG", "CHG", "CHH"] = None,
            p_value: float = .05,
            min_n: int = 0,
            save: str | Path = None
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Categorise regions as BM (Body Methylation), IM (Intermediate Methylated) and UM (Undermethylated) according
        to `Takuno and Gaut <https://doi.org/10.1073/pnas.1215380110>`_

        E.g. for CG: :math:`P_{CG}<0.05, P_{CG}\geq0.05, P_{CG}\geq0.05\ and\ N_{CG}\geq20`

        Parameters
        ----------
        context
            Methylation context (CG, CHG, CHH).
        p_value
            P-value for operation.
        min_n
            Minimal counts for cytosines methylated in selected context.
        save
            Path where files with BM, IM, UM genes will be saved (None if saving not needed).

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            BM, IM, UM :class:`pl.DataFrame`
        """
        other_contexts = list({"CG", "CHG", "CHH"} - set([context]))
        bm = (
            self.filter(context, "<", p_value, min_n)
            .filter(other_contexts[0], ">=", 1 - p_value)
            .filter(other_contexts[1], ">=", 1 - p_value)
        )
        im = (
            self.filter(context, ">=", p_value, min_n)
            .filter(context, "<", 1 - p_value, min_n)
            .filter(other_contexts[0], ">=", 1 - p_value)
            .filter(other_contexts[1], ">=", 1 - p_value)
        )
        um = (
            self.filter(context, ">=", p_value, 1 - min_n)
            .filter(other_contexts[0], ">=", 1 - p_value)
            .filter(other_contexts[1], ">=", 1 - p_value)
        )

        if save is not None:
            for df, df_type in zip([bm, im, um], ["BM", "IM", "UM"]):
                save_path = Path(save)
                df.save(save_path.with_name(save_path.name + "_" + df_type).with_suffix(".tsv"))
        return bm.region_stats, im.region_stats, um.region_stats

    def __len__(self):
        return len(self.region_stats)

    def __repr__(self):
        return self.region_stats.__repr__()

    def __str__(self):
        return self.region_stats.__str__()

