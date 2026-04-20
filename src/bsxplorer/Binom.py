from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from pyarrow import parquet as pq
from scipy.stats import binom

from .UniversalReader_batches import ARROW_SCHEMAS, ReportTypes
from .UniversalReader_classes import UniversalReader
from .utils import arrow2polars_convert


class BinomialData:
    """Calculates P-value for cytosine residues."""
    def __init__(self, path):
        self.preprocessed_path = Path(path)

    @classmethod
    def read_total_stats(
            cls,
            file: str | Path,
            report_type: ReportTypes,
            block_size_mb: int = 20,
            use_threads: bool = True,
            **kwargs
    ):
        """
        Get methylation stats from methylation reports file.

        Parameters
        ----------
        file
            Path to cytosine report.
        report_type
            Type of report. Possible types: "bismark", "cgmap", "bedgraph", "coverage".
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        kwargs
            Keyword agruements for reader (e.g. sequence)

        Returns
        -------

        """
        with UniversalReader(file, report_type, use_threads, block_size_mb=block_size_mb, **kwargs) as reader:
            metadata = dict(cytosine_residues=0, density_sum=0)
            for batch in reader:
                batch.filter_not_none()
                metadata["cytosine_residues"] += len(batch)
                metadata["density_sum"] += batch.data["density"].sum()

        return metadata

    @classmethod
    def from_report(
            cls,
            file: str | Path,
            report_type: ReportTypes,
            block_size_mb: int = 20,
            use_threads: bool = True,
            min_coverage: int = 2,
            save: str | Path | bool = None,
            dir: str | Path = Path.cwd(),
            **kwargs
    ):
        """
        Method to preprocess BS-seq cytosine report data by calculating methylation P-value for every cytosine (assuming distribution is binomial) that passes `min_coverage` threshold.

        Parameters
        ----------
        file
            Path to cytosine report.
        report_type
            Type of report. Possible types: "bismark", "cgmap", "bedgraph", "coverage".
        save
            Name with which preprocessed file will be saved. If not provided - input file name is being used.
        min_coverage
            Minimal coverage for cytosine.
        block_size_mb
            Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)
        use_threads
            Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`
        dir
            Path to working dir, where file will be saved.
        kwargs
            Keyword agruements for reader (e.g. sequence)

        Returns
        -------
        BinomialData
            Instance of Binom class.
        """
        file = Path(file)
        filename = "{tmp}{name}.binom.pq".format(
            tmp="tmp_" if save is False else "",
            name=file.stem if save is None else save
        )
        save_path = Path(dir) / filename

        # Update metadata with total distribution stats
        metadata = cls.read_total_stats(file, report_type, block_size_mb, use_threads, **kwargs)

        # Calculating total p_values
        print("Writing p_values file into:", save_path.absolute())
        total_probability = metadata["density_sum"] / metadata["cytosine_residues"]

        arrow_pvalue_schema = ARROW_SCHEMAS["binom"]
        polars_pvalue_schema = arrow2polars_convert(arrow_pvalue_schema)

        with pq.ParquetWriter(save_path, arrow_pvalue_schema) as pq_writer:
            with UniversalReader(file, report_type, use_threads, block_size_mb=block_size_mb, **kwargs) as reader:
                for batch in reader:
                    filtered = batch.data.filter(pl.col("count_total") >= min_coverage)
                    # Binomial test for cytosine methylation
                    cdf_col = 1 - binom.cdf(filtered["count_m"].cast(pl.Int64) - 1, filtered["count_total"], total_probability)
                    # Write to p_valued file
                    p_valued = (
                        filtered.with_columns(pl.lit(cdf_col).cast(pl.Float64).alias("p_value"))
                        .select(arrow_pvalue_schema.names)
                        .cast(polars_pvalue_schema)
                    )
                    pq_writer.write(p_valued.to_arrow().cast(arrow_pvalue_schema))

        print("DONE")
        print(f"\nTotal cytosine residues: {metadata['cytosine_residues']}.\nAverage proportion of methylated reads to total reads for cytosine residue: {round(metadata['density_sum'] / metadata['cytosine_residues'] * 100, 3)}%")

        return cls(save_path)

    def region_pvalue(
            self,
            genome: pl.DataFrame,
            methylation_pvalue: float = .05,
            use_threads: bool = True,
            save: str | Path | bool = None,
            dir: str | Path = Path.cwd()
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
            Name with which preprocessed file will be saved. If not provided - input file name is being used.
        dir
            Path to working dir, where file will be saved.

        Returns
        -------
        RegionStat
            Instance of RegionStat class.

        Examples
        --------
        If there no preprocessed file:

        >>> report_path = "/path/to/report.txt"
        >>> genome_path = "/path/to/genome.gff"
        >>> c_binom = bsxplorer.BinomialData.preprocess(report_path, report_type="bismark")

        >>> genome = bsxplorer.Genome.from_gff(genome_path).gene_body()
        >>> data = c_binom.region_pvalue(genome)
        >>> data
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

        >>> preprocessed_path = "/path/to/preprocessed.binom.pq"
        >>> c_binom = bsxplorer.BinomialData(preprocessed_path)
        >>> data = c_binom.region_pvalue(genome)
        """
        polars_pvalue_schema = arrow2polars_convert(ARROW_SCHEMAS["binom"])
        genome = genome.cast({k: v for k, v in polars_pvalue_schema.items() if k in genome.columns})
        genome = genome.rename({"strand": "gene_strand"})

        context_metadata = pl.DataFrame(
            schema={"context": polars_pvalue_schema["context"],
                    "count_m": pl.Int64,
                    "count_total": pl.Int64}
        )

        gene_stats = None
        with UniversalReader(
                self.preprocessed_path,
                "binom",
                use_threads,
                methylation_pvalue=methylation_pvalue
        ) as reader:
            for full_batch in reader:
                # Extend context metadata
                context_metadata.extend(
                    full_batch.data.group_by("context").agg([
                        pl.sum("count_m").cast(pl.Int64),
                        pl.sum("count_total").cast(pl.Int64)
                    ])
                )

                # Map on genome
                mapped = (
                    full_batch.data.lazy()
                    .join_asof(
                        genome.lazy(), left_on="position", right_on="start", strategy="backward", by="chr"
                    )
                    .filter(
                        # Limit by end of region
                        pl.col('position') <= pl.col('downstream'),
                        # Filter by strand if it is defined
                        ((pl.col("gene_strand") != ".") & (pl.col("gene_strand") == pl.col("strand"))) | (pl.col("gene_strand") == "."),
                        pl.col("start").is_not_nan()
                    )
                    .group_by(["chr", "start", "context"], maintain_order=True)
                    .agg([
                        pl.first("gene_strand"),
                        pl.first("end"),
                        pl.first("id"),
                        pl.sum("count_m"),
                        pl.sum("count_total")
                    ])
                    .rename({"gene_strand": "strand"})
                    .collect()
                )

                if gene_stats is None:
                    gene_stats = mapped
                else:
                    gene_stats.extend(mapped)

        context_metadata = context_metadata.group_by("context").agg([pl.sum("count_m"), pl.sum("count_total")])

        # Calculate region pvalues
        result = pl.DataFrame(schema=gene_stats.schema | {"p_value": pl.Float64})
        for metadata_row in context_metadata.iter_rows(named=True):
            total_probability = metadata_row["count_m"] / metadata_row["count_total"]

            filtered = gene_stats.filter(pl.col("context") == metadata_row["context"])
            cdf = 1 - binom.cdf(filtered["count_m"].cast(pl.Int64) - 1, filtered["count_total"], total_probability)
            result.extend(filtered.with_columns(pl.lit(cdf, pl.Float64).alias("p_value")))

            print(f"{metadata_row['context']}\tTotal sites: {metadata_row['count_total']}, methylated: {metadata_row['count_m']}\t({round(total_probability * 100, 2)}%)")

        result = result.select(["chr", "start", "end", "id", "strand", "context", "p_value", "count_m", "count_total"])

        filename = "{tmp}{name}.tsv".format(
            tmp="tmp_" if save is False else "",
            name=self.preprocessed_path.stem if save is None else save
        )
        save_path = Path(dir) / filename

        if save:
            result.write_csv(save_path.absolute(), include_header=False, separator="\t")
            print("Saved into:", save_path)

        return RegionStat.from_expanded(result)


class Filter:
    __slots__ = ["expr"]

    def __le__(self, other):
        self.expr = self.expr.le(other)
        return self

    def __lt__(self, other):
        self.expr = self.expr.lt(other)
        return self

    def __ge__(self, other):
        self.expr = self.expr.ge(other)
        return self

    def __gt__(self, other):
        self.expr = self.expr.gt(other)
        return self

    def __eq__(self, other):
        self.expr = self.expr.eq(other)
        return self

    def __and__(self, other: PValueFilter | CountFilter):
        self.expr = self.expr.and_(other.expr)
        return self

    def __or__(self, other: PValueFilter | CountFilter):
        self.expr = self.expr.or_(other.expr)
        return self


class PValueFilter(Filter):
    def __init__(self, context: str):
        self.context = context
        self.expr = pl.col(f"p_value_context_{context}")


class CountFilter(Filter):
    def __init__(self, context: str):
        super().__init__()
        self.context = context
        self.expr = pl.col(f"count_total_context_{context}")


class RegionStat:
    """
    Class for manipulation with region P-value data.

    Attributes
    ----------
    data : polars.DataFrame
        Region stats DataFrame
    """

    schema = {
        'chr': pl.Utf8,
        'start': pl.UInt64,
        'end': pl.UInt64,
        'id': pl.Utf8,
        'strand': pl.Utf8,
        'context': pl.Utf8,
        'p_value': pl.Float64,
        'count_m': pl.UInt32,
        'count_total': pl.UInt32
    }

    def __init__(self, data: pl.DataFrame = None):
        """
        Class for manipulation with region P-value data.

        Warnings
        --------
        Do not call this method directly.

        Parameters
        ----------
        data : polars.DataFrame
            Region stats DataFrame
        """

        self.data = data

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
            .pivot(values=["p_value", "count_m", "count_total"],
                   columns="context",
                   index=list(set(df.columns) - {"p_value", "context", "count_m", "count_total"}))
        )
        return cls(gene_stats)

    @classmethod
    def from_csv(cls, file: str | Path):
        """
        Read RegionStat data from preprocessed and saved with :method:`RegionStat.save`

        Parameters
        ----------
        file
            Path to file.

        """
        if not file.exists(): raise FileNotFoundError(file)
        df = pl.read_csv(file, has_header=False, separator="\t", schema=cls.schema)
        return cls.from_expanded(df)

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
        >>> data = BinomialData("./A_thaliana.binom.pq").region_pvalue(genome)
        >>> data
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
        >>> data.filter("CG", "<", 0.05, 20)
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
            self.data
            .filter(expr)
            .filter(pl.col(f"count_total_context_{context}") >= min_n)
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
            self.data
            .select(["chr", "start", "end", "id", "strand"])
            .write_csv(path, include_header=False, separator="\t")
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
        other_contexts = [
            c for c in
            map(lambda name: name.replace("p_value_context_", ""), self.data.select("^p_value_context_.+$").columns)
            if c != context
        ]

        assert isinstance(context, str)
        bm_filter = (PValueFilter(context) < p_value) & (CountFilter(context) >= min_n)
        im_filter = (PValueFilter(context) >= p_value) & (PValueFilter(context) < 1 - p_value) & (CountFilter(context) >= min_n)
        um_filter = (PValueFilter(context) >= 1 - p_value) & (CountFilter(context) >= min_n)

        for other_context in other_contexts:
            bm_filter = bm_filter & (PValueFilter(other_context) >= p_value)
            im_filter = im_filter & (PValueFilter(other_context) >= p_value)
            um_filter = um_filter & (PValueFilter(other_context) >= p_value)

        bm = self.__class__(self.data.filter(bm_filter.expr))
        im = self.__class__(self.data.filter(im_filter.expr))
        um = self.__class__(self.data.filter(um_filter.expr))

        if save is not None:
            for df, df_type in zip([bm, im, um], ["BM", "IM", "UM"]):
                save_path = Path(save)
                df.save(save_path.with_name(save_path.name + "_" + df_type).with_suffix(".tsv"))
        return bm.data, im.data, um.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

