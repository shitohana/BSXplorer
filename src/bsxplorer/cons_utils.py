from __future__ import annotations

import argparse
import base64
import time
import warnings
from abc import ABC, abstractmethod

from dataclasses import dataclass, field, asdict
from gc import collect
from io import BytesIO
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pcsv
from jinja2 import Template
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from . import Genome, Metagene, PCA, MetageneFiles, BinomialData, ChrLevels, HeatMap
from .BamReader import BAMReader
from .SeqMapper import init_tempfile
from .UniversalReader_batches import REPORT_TYPES_LIST
from .UniversalReader_classes import UniversalWriter, UniversalReader, UniversalReplicatesReader


def render_template(
        template: str | Path,
        params: dict,
        output: str | Path
):
    with open(template) as template_file:
        j2_template = Template(template_file.read())

        with open(output, "w") as output_file:
            output_file.write(
                j2_template.render(params)
            )


@dataclass
class TemplatePlot:
    title: str = field(default_factory=str)
    data: str = field(default_factory=str)


@dataclass
class TemplateContext:
    heading: str = field(default_factory=str)
    caption: str = field(default_factory=str)
    plots: list[TemplatePlot] = field(default_factory=list)


@dataclass
class TemplateBody:
    title: str = field(default_factory=str)
    context_reports: list[TemplateContext] = field(default_factory=list)


_config = {
    "MERGE_BATCH_SIZE": 10 ** 6
}


def _config_path(arg) -> Path:
    config_path = Path(arg).expanduser().absolute()
    if not config_path.exists():
        raise FileNotFoundError(arg)
    return config_path


def _working_dir(arg) -> Path:
    dir_path = Path(arg).expanduser().absolute()
    if dir_path.exists():
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a folder!")
    else:
        print(f"{dir_path} does not exists. Creating folder there.")
        dir_path.mkdir()
    return dir_path


def _bbin(arg) -> int:
    if int(arg) < 1:
        raise ValueError("There should be at least 1 body window")
    return int(arg)


def _quantile(arg) -> float:
    if not (0 <= float(arg) < 1):
        raise ValueError("Value should be in interval [0, 1).")
    else:
        return float(arg)


def _ticks(arg) -> str:
    return str(arg).replace("\\", "")


def metagene_parser():
    parser = argparse.ArgumentParser(
        prog='bsxplorer-metagene',
        description='Metagene report creation tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config',             help='Path to config file', type=_config_path)
    parser.add_argument('-o', '--out',        help='Output filename', default=f"Metagene_Report_{time.strftime('%d-%m-%y_%H-%M-%S')}", metavar='NAME')
    parser.add_argument('--dir',              help='Output and working dir', default=Path.cwd(), metavar='DIR', type=_working_dir)

    parser.add_argument('-m', '--block_mb',   help='Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)', type=int, default=50)
    parser.add_argument('-t', '--threads',    help='Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`', action="store_true")
    parser.add_argument('-s', '--sumfunc',    help='Summary function to calculate density for bin with.', type=str, default="wmean", choices=["wmean", "mean", "median", "min", "max", "1pgeom"])
    parser.add_argument('-u', '--ubin',       help='Number of windows for upstream region', type=int, default=50)
    parser.add_argument('-d', '--dbin',       help='Number of windows for downstream downstream', type=int, default=50)
    parser.add_argument('-b', '--bbin',       help='Number of windows for body region', type=_bbin, default=100)

    parser.add_argument('-q', '--quantile',   help='Quantile of most varying genes to draw on clustermap', type=_quantile, default=.75)
    parser.add_argument('-C', '--confidence', help='Probability for confidence bands for line-plot. 0 if disabled', type=_quantile, default=.95)
    parser.add_argument('-S', '--smooth',     help='Windows for SavGol function.', type=float, default=10)
    parser.add_argument('-H',                 help='Vertical resolution for heat-map', type=int, default=100, dest="vresolution")
    parser.add_argument('-V',                 help='Vertical resolution for heat-map', type=int, default=100, dest="hresolution")

    parser.add_argument('--separate_strands', help='Do strands need to be processed separately', action='store_true')
    parser.add_argument('--export',           help='Export format for plots (set none to disable)', type=str, default='pdf', choices=['pdf', 'svg', 'none'])
    parser.add_argument('--ticks',            help='Names of ticks (- character should be escaped with double reverse slash)', type=_ticks, nargs=5)

    return parser


def chr_parser():
    parser = argparse.ArgumentParser(
        prog='bsxplorer-chr',
        description='Chromosome methylation levels visualisation tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config', help='Path to config file', type=_config_path)
    parser.add_argument('-o', '--out', help='Output filename', default=f"Metagene_Report_{time.strftime('%d-%m-%y_%H-%M-%S')}", metavar='NAME')
    parser.add_argument('--dir', help='Output and working dir', default=str(Path.cwd()), metavar='DIR', type=_working_dir)

    parser.add_argument('-m', '--block_mb', help='Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)', type=int, default=50)
    parser.add_argument('-t', '--threads', help='Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`', type=int, default=True)
    parser.add_argument('-w', '--window', help="Length of windows in bp", type=int, default=10**6)
    parser.add_argument('-l', '--min_length', help="Minimum length of chromosome to be analyzed", type=int, default=10**6)
    parser.add_argument('-C', '--confidence', help='Probability for confidence bands for line-plot. 0 if disabled', type=_quantile, default=.95)
    parser.add_argument('-S', '--smooth', help='Windows for SavGol function.', type=float, default=100)

    parser.add_argument('--export',           help='Export format for plots (set none to disable)', type=str, default='pdf', choices=['pdf', 'svg', 'none'])

    parser.add_argument('--separate_strands', help='Do strands need to be processed separately', action='store_true')

    return parser


class ConsoleScript(ABC):
    def __init__(self, args: list[str] = None):
        warnings.filterwarnings('ignore')

        self.args = self._parser.parse_args(args)

        config = self._read_config(self.args.config)
        self._validate_config(config)
        self.config = config

        self._plots_folder()

        self._init_additional()

        self._print_args(self.args)

    _GENOME_TYPES = ["gff"]
    _REPORT_TYPES = ["bismark"]

    _CONFIG_SCHEMA = {
        "name": pl.Utf8,
        "report_file": pl.Utf8,
        "genome_file": pl.Utf8,
        "flank_length": pl.Int32,
        "min_length": pl.Int32,
        "region_type": pl.Utf8,
        "genome_type": pl.Utf8,
        "report_type": pl.Utf8
    }

    def _read_config(self, path) -> pl.DataFrame:
        report_args = pl.read_csv(
            path,
            has_header=False,
            separator="\t",
            schema=self._CONFIG_SCHEMA,
            truncate_ragged_lines=True,
            comment_char="#"
        )

        report_args = (
            report_args
            .with_row_count("id")
            .with_columns([
                pl.col("report_file").map_elements(lambda path: str(Path(path).expanduser().absolute())),
                pl.col("genome_file").map_elements(lambda path: str(Path(path).expanduser().absolute())),
                pl.col("flank_length").fill_null(0),
                pl.col("min_length").fill_null(0),
                pl.col("genome_type").fill_null("gff"),
                pl.col("report_type").fill_null("bismark"),
            ])
            .drop("id")
        )

        return report_args

    def _validate_config(self, config: pl.DataFrame):
        # files exist check
        for path in set(config["report_file"]) | set(config["genome_file"]):
            if not Path(path).expanduser().absolute().exists():
                raise FileNotFoundError(path)

        # negative flank length check
        if (config["flank_length"].to_numpy() < 0).sum() > 0:
            raise ValueError("Flank length cant be negative.")

        # unsupported genome type check
        if sum([t not in self._GENOME_TYPES for t in config["genome_type"]]) > 0:
            raise ValueError(f"Not all genome types are supported. (Supported: {self._GENOME_TYPES})")

        # unsupported report type check
        if sum([t not in self._REPORT_TYPES for t in config["report_type"]]) > 0:
            raise ValueError(f"Not all genome types are supported. (Supported: {self._REPORT_TYPES})")

        # unique genome for sample group check
        if config.group_by("name").agg(pl.col("genome_file").unique().len() != 1)["genome_file"].sum() > 0:
            raise ValueError("Different genome paths for same sample group.")

    def _plots_folder(self):
        if self.args.export != 'none':
            plot_dir = self.args.dir / "plots"

            if not plot_dir.exists():
                plot_dir.mkdir()

            print("Plots will be saved in directory:", plot_dir)

    @property
    def sample_list(self) -> list:
        mapping = (
            self.config
            .group_by("name")
            .agg([pl.col("report_file"), pl.col("report_type"), pl.col("genome_file").first()])
            .to_dicts()
        )

        return mapping

    def genome_mapping(self) -> dict:
        mapping = {}

        constructors = {
            "gff": Genome.from_gff,
        }

        grouped = (
            self.config
            .group_by(["genome_file", "genome_type", "min_length", "flank_length", "region_type"])
            .agg("name")
        )

        for row in grouped.to_dicts():
            genome: Genome = constructors[row["genome_type"]](row["genome_file"])

            kwargs = dict(min_length=row["min_length"], flank_length=row["flank_length"])
            if row["genome_type"] is None or row["genome_type"] in ["none", "all"]:
                annotation: pl.DataFrame = genome.all(**kwargs)
            else:
                annotation: pl.DataFrame = genome.other(region_type=row["region_type"], **kwargs)

            names = row["name"] if isinstance(row["name"], list) else list(row["name"])

            mapping |= {name: annotation for name in names}

        return mapping

    def _init_additional(self):
        pass

    @property
    @abstractmethod
    def _parser(self) -> argparse.ArgumentParser:
        ...

    @abstractmethod
    def _print_args(self, args: argparse.Namespace):
        ...

    @abstractmethod
    def main(self):
        ...


class BamScript:
    @property
    def _parser(self) -> argparse.ArgumentParser:
        def file_path_type(arg) -> Path:
            if arg is not None:
                arg = Path(arg).expanduser().absolute()
                if not arg.exists():
                    raise FileNotFoundError(arg)
            return arg

        BAM_TYPES = ["bismark"]
        CONTEXTS = ["CG", "CHG", "CHH", "all"]
        MODES = ["report", "stats"]
        STATS = ["ME", "EPM", "PDR"]

        parser = argparse.ArgumentParser(
            prog='bsxplorer-bam',
            description='BAM to report reader converter tool.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument('output', help="Path to output file.", type=Path)
        parser.add_argument("--bam", help="Path to SORTED .bam file with alignments", type=file_path_type, required=True)
        parser.add_argument("--bai", help="Path to .bai index file", type=file_path_type, required=True)
        parser.add_argument("-f", "--fasta", help="Path to .fasta file with reference sequence for full cytosine report.", default=None, type=file_path_type)
        parser.add_argument("--bamtype", help="Type of aligner which was used for generating BAM.", default="bismark", type=str, choices=BAM_TYPES)
        parser.add_argument('-m', '--mode', default="report", choices=MODES)
        parser.add_argument('--to_type', choices=REPORT_TYPES_LIST, help="Specifies the output file type if mode is set to 'report'.", default="bismark")
        parser.add_argument('--stat', choices=STATS, help="Specifies the BAM stat type if mode is set to 'stats'", default="ME")
        parser.add_argument('--stat_param', type=int, default=4, help="See docs for specifical stat parameters.")
        parser.add_argument('--stat_md', type=int, default=4, help="Minimum number of reads for cytosine to be analysed (if mode is 'stats')")
        parser.add_argument('-g', '--gff', help="Path to regions genome coordinates .gff file, if cytosines need to be filtered.", default=None, type=file_path_type)
        parser.add_argument("-c", "--context", help="Filter cytosines by specific methylation context", type=str, default="all", choices=CONTEXTS)
        parser.add_argument("-q", "--min_qual", help="Filter cytosines by read Phred score quality", type=int, choices=range(0, 43), default=None)
        parser.add_argument('-s', '--skip_converted', help="Skip reads aligned to converted sequence", action='store_true')
        parser.add_argument('--no_qc', help="Do not calculate QC stats", action='store_true')
        parser.add_argument('-t', '--threads', help="How many threads will be used for reading the BAM file.", type=int, default=1)
        parser.add_argument('-n', '--batch_n', help="Number of reads per batch.", default=1e4, type=int)
        parser.add_argument('-a', '--readahead', help="Number of batches to be read before processing.", default=5, type=int)

        return parser

    def __init__(self, args: list[str] = None):

        warnings.filterwarnings('ignore')
        self.args = self._parser.parse_args(args)

    def main(self):
        args = self.args

        if args.gff is not None:
            regions = Genome.from_gff(args.gff).all()
        else:
            regions = None

        output = Path(args.output).expanduser().absolute()

        reader = BAMReader(
            bam_filename=args.bam,
            index_filename=args.bai,
            cytosine_file=args.fasta,
            bamtype=args.bamtype,
            regions=regions,
            batch_num=args.batch_n,
            min_qual=args.min_qual,
            threads=args.threads,
            context=args.context,
            keep_converted=not args.skip_converted,
            qc=not args.no_qc,
            readahead=args.readahead
        )
        if args.mode == "report":
            with UniversalWriter(
                    file=output,
                    report_type=args.to_type
            ) as writer:
                for batch in reader.report_iter():
                    writer.write(batch)

        elif args.mode == "stats":
            if args.stat in ["ME", "EPM"]:
                out_schema = pa.schema(
                    [pa.field("chr", pa.string())] +
                    [pa.field(f"pos{pos_n}", pa.uint64()) for pos_n in range(args.stat_param)] +
                    [pa.field("value", pa.float64())]
                )

                with pcsv.CSVWriter(
                        args.output, schema=out_schema,
                        write_options=pcsv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
                ) as writer:
                    for batch in reader.stats_iter():
                        if batch is not None:
                            positions_matrix, value = batch.methylation_entropy(args.stat_param, args.stat_md)
                            if value.size != 0:
                                non_null = value != 0

                                positions_matrix = positions_matrix[non_null, :]
                                value = value[non_null]

                                if len(value) > 0:
                                    res_df = (
                                        pl.DataFrame(
                                            [list(positions_matrix), value],
                                            schema={"positions": pl.List(pl.UInt64), "value": pl.Float64}
                                        )
                                        .with_columns(pl.col("positions").list.to_struct())
                                        .with_columns(pl.lit(batch.chr).alias("chr"))
                                        .select(["chr", "positions", "value"])
                                        .unnest("positions")
                                    )
                                    writer.write(res_df.to_arrow())

            elif args.stat == "PDR":
                out_schema = pa.schema([
                    pa.field("chr", pa.string()),
                    pa.field("pos", pa.uint64()),
                    pa.field("value", pa.float64()),
                    pa.field("ccount", pa.float64()),
                    pa.field("dcount", pa.float64())
                ])

                with pcsv.CSVWriter(
                        output, schema=out_schema,
                        write_options=pcsv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
                ) as writer:
                    for batch in reader.stats_iter():
                        if batch is not None:
                            position_array, pdr_array, count_matrix = batch.PDR(args.stat_param, args.stat_md)

                            if pdr_array.size != 0:

                                non_null = pdr_array != 0
                                position_array = position_array[non_null]
                                pdr_array = pdr_array[non_null]
                                count_matrix = count_matrix[non_null, :]

                                if len(position_array) > 0:
                                    res_df = (
                                        pl.DataFrame(
                                            data=[position_array, pdr_array, list(count_matrix)],
                                            schema={"pos": pl.UInt64, "value": pl.Float64, "counts": pl.List(pl.UInt64)}
                                        )
                                        .with_columns(pl.col("counts").list.to_struct())
                                        .with_columns(pl.lit(batch.chr).alias("chr"))
                                        .select(["chr", "pos", "value", "counts"])
                                        .unnest("counts")
                                    )
                                    writer.write(res_df.to_arrow())

        else:
            raise KeyError(f"Unknown mode {args.mode}")

        if not args.no_qc:
            fig = reader.plot_qc()
            fig.set_size_inches(10, 7)
            fig.savefig(output.parent / (output.stem + ".pdf"))


class MetageneScript(ConsoleScript):
    @property
    def _parser(self) -> argparse.ArgumentParser:
        return metagene_parser()

    def _print_args(self, args: argparse.Namespace):
        print(
            'BSXplorer MetageneReport Run:\n\n'
            
            'Config:\n',
            self.config,
            "\n"
            f"Working directory: {args.dir}\n"
            f"Plots directory: {args.dir / 'plots'}\n"
            f"Report name: {args.out}\n"
            f"Block size (Mb): {args.block_mb}\n"
            f"Summary function: {args.sumfunc}\n"
            f"Threads: {args.threads}\n"
            f"\nUpstream | Body | Downstream (bins): {args.ubin} | {args.bbin} | {args.dbin}\n\n"
            
            f"Clustermap variance filtering quantile: {args.quantile}\n"
            f"Confidence band alpha for line_plot: {args.confidence}\n"
            f"Smoothing function window width: {args.smooth}\n"
            f"Heat_map dimensions (hor, ver): ({args.hresolution}, {args.vresolution})\n\n"
            
            f"Separate strands? - {args.separate_strands}\n"
            f"Export format: {args.export}\n"
            f"Tick_names: {'default' if args.ticks is None else args.ticks}\n\n"
            
            f"RUN STARTED at {time.strftime('%d/%m/%y %H:%M:%S')}\n\n"
        )

    _metagene_constructors = {
        "bismark": Metagene.from_bismark,
        "cgmap": Metagene.from_cgmap
    }

    def main(self):
        genome_mapping = self.genome_mapping()

        sample_metagenes = []
        sample_names = []

        # one genome check
        one_genome = self.config["genome_file"].unique().len() == 1
        if one_genome:
            pca = PCA()
        else:
            pca = None

        for row in self.sample_list:
            report_files = row["report_file"] if isinstance(row["report_file"], list) else list(row["report_file"])
            report_types = row["report_type"] if isinstance(row["report_type"], list) else list(row["report_type"])

            replicates = []
            for report_file, report_type in zip(report_files, report_types):
                kwargs = dict(
                    file=report_file,
                    genome=genome_mapping[row["name"]],
                    up_windows=self.args.ubin,
                    body_windows=self.args.bbin,
                    down_windows=self.args.dbin,
                    use_threads=self.args.threads,
                    sumfunc=self.args.sumfunc
                )

                if report_type not in ["parquet"]:
                    kwargs |= dict(block_size_mb=self.args.block_mb)

                report_metagene = self._metagene_constructors[report_type](**kwargs)

                if pca is not None:
                    pca.append_metagene(report_metagene, Path(report_file).stem, row["name"])
                replicates.append(report_metagene)

            sample_metagenes.append(
                MetageneFiles(replicates).merge()
            )
            sample_names.append(row["name"])

        metagene_files = MetageneFiles(sample_metagenes, sample_names)
        pass
        rendered = Renderer(self.args).metagene(metagene_files, pca, draw_cm=one_genome)

        return rendered


# Only report_type = bismark is supported rn
# To add more supported types:
#   - bsxplorer.utils.merge_replicates
#   - bsxplorer.Binom.BinomialData
# need to be updated
class CategoryScript(ConsoleScript):
    @property
    def _parser(self) -> argparse.ArgumentParser:
        parser = metagene_parser()

        parser.prog = "bsxplorer-categorise"
        parser.description = 'BM, UM categorisation tool'

        parser.add_argument("--cytosine_p", help="P-value for binomial test to consider cytosine methylated", default=".05", type=float)
        parser.add_argument("--min_cov", help="Minimal coverage for cytosine to keep", default="2", type=int)
        parser.add_argument("--region_p", help="P-value for binomial test to consider region methylated", default=".05", type=float)
        parser.add_argument("--save_cat", help="Does categories need to be saved", default=True, type=bool, action=argparse.BooleanOptionalAction)

        return parser

    def _print_args(self, args: argparse.Namespace):
        print(
            'BSXplorer MetageneReport Run:\n\n'
            'Config:\n',
            self.config,
            "\n"
            f"Working directory: {args.dir}\n"
            f"Plots directory: {args.dir / 'plots'}\n"
            f"Report name: {args.out}\n"
            f"Block size (Mb): {args.block_mb}\n"
            f"Summary function: {args.sumfunc}\n"
            f"Threads: {args.threads}\n"
            f"\nUpstream | Body | Downstream (bins): {args.ubin} | {args.bbin} | {args.dbin}\n\n"
            
            f"P-value for cytosines: {args.cytosine_p}\n"
            f"P-value for region: {args.region_p}\n"
            f"Minimal coverage: {args.min_cov}\n"
            f"Save categories? - {args.save_cat}\n\n"
            
            f"Clustermap variance filtering quantile: {args.quantile}\n"
            f"Confidence band alpha for line_plot: {args.confidence}\n"
            f"Smoothing function window width: {args.smooth}\n"
            f"Heat_map dimensions (hor, ver): ({args.hresolution}, {args.vresolution})\n\n"
            
            f"Separate strands? - {args.separate_strands}\n"
            f"Export format: {args.export}\n"
            f"Tick_names: {'default' if args.ticks is None else args.ticks}\n\n"
            
            f"RUN STARTED at {time.strftime('%d/%m/%y %H:%M:%S')}\n\n"
        )

    def main(self):
        genome_mapping = self.genome_mapping()

        sample_metagenes = []
        sample_names = []
        sample_region_stats = {}
        for row in self.sample_list:
            report_files = row["report_file"] if isinstance(row["report_file"], list) else list(row["report_file"])
            report_types = row["report_type"] if isinstance(row["report_type"], list) else list(row["report_type"])

            if len(report_files) > 1:
                temp = init_tempfile(self.args.dir, row["name"] + "_merged", delete=False, suffix=".txt")
                print(f"Merged replicates will be saved as {temp}")

                with UniversalWriter(temp, report_types[0]) as writer:
                    readers = [UniversalReader(file, report_type, self.args.threads,
                                               block_size_mb=self.args.block_mb, bar=False)
                               for file, report_type in zip(report_files, report_types)]

                    with UniversalReplicatesReader(readers) as reader:
                        for batch in reader:
                            writer.write(batch)

                merged_path = Path(temp)
                merged_type = report_types[0]
            else:
                merged_path = Path(report_files[0])
                merged_type = report_types[0]

            binom = BinomialData.from_report(
                file=merged_path,
                report_type=merged_type,
                block_size_mb=self.args.block_mb,
                use_threads=self.args.threads,
                min_coverage=self.args.min_cov,
                dir=self.args.dir
            )

            region_pvalues = binom.region_pvalue(
                genome_mapping[row["name"]],
                self.args.cytosine_p,
                self.args.threads
            )

            metagene = Metagene.from_binom(
                file=binom.preprocessed_path,
                genome=genome_mapping[row["name"]],
                up_windows=self.args.ubin,
                body_windows=self.args.bbin,
                down_windows=self.args.dbin,
                use_threads=self.args.threads,
                p_value=self.args.cytosine_p
            )

            sample_region_stats[row["name"]] = region_pvalues
            sample_metagenes.append(metagene)
            sample_names.append(row["name"])

        metagene_files = MetageneFiles(sample_metagenes, sample_names)

        rendered = Renderer(self.args).category(metagene_files, sample_region_stats)

        return rendered


class ChrLevelsScript(ConsoleScript):
    @property
    def _parser(self) -> argparse.ArgumentParser:
        parser = chr_parser()
        return parser

    def _print_args(self, args: argparse.Namespace):
        print(
            'BSXplorer MetageneReport Run:\n\n'
            'Config:\n',
            self.config,
            "\n"
            f"Working directory: {args.dir}\n"
            f"Plots directory: {args.dir / 'plots'}\n"
            f"Report name: {args.out}\n"
            f"Block size (Mb): {args.block_mb}\n"
            # f"Summary function: {args.sumfunc}\n"
            f"Threads: {args.threads}\n"
            
            f"Length of window: {args.window}\n"
            f"Minimum length of chromosome: {args.min_length}\n"
            
            f"Confidence band alpha for line_plot: {args.confidence}\n"
            f"Smoothing function window width: {args.smooth}\n"
            f"Separate strands? - {args.separate_strands}\n"
            f"Export format: {args.export}\n"
            
            f"RUN STARTED at {time.strftime('%d/%m/%y %H:%M:%S')}\n\n"
        )

    def _init_additional(self):
        self.config = self.config.select(["name", "report_file", "report_type"])

    @property
    def sample_list(self) -> list:
        mapping = (
            self.config
            .group_by("name")
            .agg([pl.col("report_file"), pl.col("report_type")])
            .to_dicts()
        )

        return mapping

    _constructors = {
        "bismark": ChrLevels.from_bismark,
        "binom": ChrLevels.from_binom,
        "cgmap": ChrLevels.from_cgmap
    }

    def main(self):
        sample_levels = []
        sample_labels = []

        for row in self.sample_list:
            report_files = row["report_file"] if isinstance(row["report_file"], list) else list(row["report_file"])
            report_types = row["report_type"] if isinstance(row["report_type"], list) else list(row["report_type"])

            temp = None

            if len(report_files) > 1:
                temp = init_tempfile(self.args.dir, row["name"] + "_merged", delete=False, suffix=".txt")
                print(f"Merged replicates will be saved as {temp}")

                with UniversalWriter(temp, report_types[0]) as writer:
                    readers = [UniversalReader(file, report_type, self.args.threads,
                                               block_size_mb=self.args.block_mb, bar=False)
                               for file, report_type in zip(report_files, report_types)]

                    with UniversalReplicatesReader(readers) as reader:
                        for batch in reader:
                            writer.write(batch)

                merged_path = Path(temp)
                merged_type = report_types[0]
            else:
                merged_path = Path(report_files[0])
                merged_type = report_types[0]

            kwargs = dict(
                file=merged_path,
                chr_min_length=self.args.min_length,
                window_length=self.args.window,
                confidence=self.args.confidence,
                use_threads=self.args.threads
            )

            if merged_type not in ["parquet"]:
                kwargs |= dict(
                    block_size_mb=self.args.block_mb,
                    use_threads=self.args.threads
                )

            levels = self._constructors[merged_type](**kwargs)
            sample_levels.append(levels)
            sample_labels.append(row["name"])

        rendered = Renderer(self.args).chr_levels(sample_levels, sample_labels)

        return rendered


class Renderer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.__add_plotlyjs = True

    def _save_mpl(self, fig: Figure, name: str):
        path = self.args.dir / "plots" / (name + f'.{self.args.export}')
        fig.savefig(path)

    @staticmethod
    def _p2html(fig: go.Figure, full_html=False, include_plotlyjs=False, default_width="900px", default_height="675px"):
        return fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, default_width=default_width, default_height=default_height)

    @property
    def _get_filters(self):
        if self.args.separate_strands:
            return [
                dict(context=context, strand=strand)
                for strand in ["+", "-"]
                for context in ["CG", "CHG", "CHH"]
            ]
        else:
            return [
                dict(context=context)
                for context in ["CG", "CHG", "CHH"]
            ]

    @staticmethod
    def _format_filters(filters: dict):
        if filters.get("strand") is not None:
            return f"{filters['context']}{filters['strand']}"
        else:
            return f"{filters['context']}"

    def category_context_block(self, bm, im, um, filters, metadata):
        filter_name = self._format_filters(filters)
        heading_text = f"Context {filter_name}"
        caption_text = "<p>Group regions count:</p>\n" + "\n".join(
            [f"<p>{label} – " + ", ".join([f"{group}: {number}" for group, number in metadata[label].items()]) + " regions</p>"
             for label in metadata.keys()]
        )
        context_block = TemplateContext(
            heading=heading_text,
            caption=caption_text
        )
        subplot_titles = ["BM", "IM", "UM"]

        lp_subplot_args = dict(
            rows=1, cols=3,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
            horizontal_spacing=.05
        )
        lp_fig = make_subplots(**lp_subplot_args)

        hm_subplot_args = dict(
            rows=len(metadata.keys()), cols=3,
            shared_yaxes=True,
            horizontal_spacing=.05
        )
        hm_fig = make_subplots(**hm_subplot_args)

        bp_fig = make_subplots(**(lp_subplot_args | dict(shared_yaxes=False)))

        for mfiles, mtype, plot_idx in zip([bm, im, um], subplot_titles, range(1, 4)):
            name = filter_name + f"_{mtype}" + "_{type}"

            # Plots
            line_plot = mfiles.line_plot(
                merge_strands=not self.args.separate_strands,
                smooth=self.args.smooth,
                confidence=self.args.confidence
            )
            heat_map = mfiles.heat_map(
                nrow=self.args.vresolution,
                ncol=self.args.hresolution
            )
            box_plot = mfiles.trim_flank().box_plot()

            # Matplotlib
            if self.args.export not in ["none"]:
                self._save_mpl(line_plot.draw_mpl(tick_labels=self.args.ticks), name.format(type="line_plot"))
                self._save_mpl(heat_map.draw_mpl(tick_labels=self.args.ticks), name.format(type="heat_map"))
                self._save_mpl(box_plot.draw_mpl(), name.format(type="box_plot"))

            # Plotly
            axis_name = f"xaxis{'' if plot_idx == 1 else plot_idx}"
            lp_args = {axis_name: dict(
                tickmode="array",
                tickvals=line_plot.data[0].x_ticks,
                ticktext=self.args.ticks
            )}
            hm_args = lp_args.copy()
            hm_args[axis_name] |= dict(tickvals=heat_map.data[0].x_ticks)
            # Line plot
            line_plot.draw_plotly(lp_fig, tick_labels=self.args.ticks, fig_rows=1, fig_cols=plot_idx)
            # Heat map
            for count, (hm_data, label) in enumerate(zip(heat_map.data, metadata.keys())):
                HeatMap(hm_data).draw_plotly(hm_fig, tick_labels=self.args.ticks, row=count + 1, col=plot_idx, title=label+mtype)
            # Box plot
            box_plot.draw_plotly(bp_fig, points=False, fig_rows=1, fig_cols=plot_idx)

        context_block.plots.append(
            TemplatePlot("Line plot", self._p2html(lp_fig, include_plotlyjs=self.__add_plotlyjs, default_width='80%'))
        )
        if self.__add_plotlyjs: self.__add_plotlyjs = False
        context_block.plots.append(
            TemplatePlot("Heatmap", self._p2html(hm_fig, include_plotlyjs=self.__add_plotlyjs, default_width='80%'))
        )
        context_block.plots.append(
            TemplatePlot("Box plot", self._p2html(bp_fig, include_plotlyjs=self.__add_plotlyjs))
        )

        return context_block

    def metagene_context_block(self, filtered: MetageneFiles, filters: dict, draw_cm: bool = True):

        filter_name = self._format_filters(filters)
        name = filter_name + "_{type}"

        context_block = TemplateContext(
            heading=f"Context {filter_name}"
        )

        # Clustermap
        if len(filtered.samples) > 1 and draw_cm:
            tmpfile = BytesIO()

            fig = filtered.dendrogram(q=self.args.quantile)

            # Plotly block
            fig.savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html_plot = TemplatePlot(
                title=f"Clustermap for {self.args.quantile} quantile",
                data="<img src=\'data:image/png;base64,{}\'>".format(encoded)
            )
            context_block.plots.append(html_plot)

            # Matplotlib
            if self.args.export not in ["none"]:
                self._save_mpl(fig, name.format(type="sample-cluster"))

        line_plot = filtered.line_plot(merge_strands=not self.args.separate_strands, smooth=self.args.smooth, confidence=self.args.confidence, stat="mean")
        heat_map = filtered.heat_map(nrow=self.args.vresolution, ncol=self.args.hresolution)
        box_plot = filtered.trim_flank().box_plot()

        # Matplotlib
        if self.args.export not in ["none"]:
            self._save_mpl(line_plot.draw_mpl(tick_labels=self.args.ticks), name.format(type="line_plot"))
            self._save_mpl(heat_map.draw_mpl(tick_labels=self.args.ticks), name.format(type="heat-map"))
            self._save_mpl(box_plot.draw_mpl(), name.format(type="box_plot"))
            plt.close()

        # Plotly
        context_block.plots.append(TemplatePlot(
            "Line plot",
            self._p2html(line_plot.draw_plotly(tick_labels=self.args.ticks), include_plotlyjs=self.__add_plotlyjs)
        ))

        if self.__add_plotlyjs: self.__add_plotlyjs = False

        context_block.plots.append(TemplatePlot(
            "Heat map",
            self._p2html(heat_map.draw_plotly(tick_labels=self.args.ticks), include_plotlyjs=self.__add_plotlyjs)
        ))

        context_block.plots.append(TemplatePlot(
            "Box plot",
            self._p2html(box_plot.draw_plotly(), include_plotlyjs=self.__add_plotlyjs)
        ))
        collect()

        return context_block

    def metagene(self, metagene_files: MetageneFiles, pca: PCA, draw_cm: bool = True):
        html_body = TemplateBody("Metagene Report")

        # PCA
        if pca is not None:
            try:
                pca_plot_data = self._p2html(pca.draw_plotly(), include_plotlyjs=self.__add_plotlyjs)
                pca_plot = TemplatePlot(
                    data=pca_plot_data
                )
                pca_block = TemplateContext(
                    heading="PCA plot for all replicates",
                    caption="",
                    plots=[pca_plot]
                )
                html_body.context_reports.append(pca_block)

                if self.__add_plotlyjs:
                    self.__add_plotlyjs = False
            except ValueError:
                print("Got different annotations. No PCA will be drawn")

        # Other
        filters_list = self._get_filters
        for filters in filters_list:
            filtered_metagenes = metagene_files.filter(**filters)

            context_block = self.metagene_context_block(filtered_metagenes, filters, draw_cm)
            html_body.context_reports.append(context_block)

        return asdict(html_body)

    def category(self, metagene_files: MetageneFiles, region_pvalues: dict):
        html_body = TemplateBody("Category Report")

        filters_list = self._get_filters

        for filters in filters_list:
            filtered_metagenes = metagene_files.filter(**filters)

            # Get categorised
            bm_metagenes, um_metagenes, im_metagenes = [], [], []
            metadata = {}

            for sample, label in zip(filtered_metagenes.samples, filtered_metagenes.labels):
                save_name = self.args.dir / f"{label}_{self._format_filters(filters)}" if self.args.save_cat else None

                # todo add filtering not only by id
                bm_ids, im_ids, um_ids = region_pvalues[label].categorise(
                    p_value=self.args.region_p,
                    save=save_name,
                    context=filters["context"]
                )

                metadata[label] = {"BM": len(bm_ids), "IM": len(im_ids), "UM": len(um_ids)}

                bm_metagenes.append(sample.filter(genome=bm_ids))
                im_metagenes.append(sample.filter(genome=im_ids))
                um_metagenes.append(sample.filter(genome=um_ids))

            bm_metagene_files = MetageneFiles(bm_metagenes, [label + "_BM" for label in filtered_metagenes.labels])
            im_metagenes_files = MetageneFiles(im_metagenes, [label + "_IM" for label in filtered_metagenes.labels])
            um_metagene_files = MetageneFiles(um_metagenes, [label + "_UM" for label in filtered_metagenes.labels])

            html_body.context_reports.append(self.category_context_block(bm_metagene_files, im_metagenes_files, um_metagene_files, filters, metadata))

        return asdict(html_body)

    def chr_levels(self, chr_levels_list: list[ChrLevels], labels: list[str]):
        html_body = TemplateBody("Chromosome levels report")

        filters_list = self._get_filters

        for filters in filters_list:
            context_report = TemplateContext(heading=f"Context {self._format_filters(filters)}")

            lp_fig = make_subplots()
            # bp_fig = make_subplots(rows=len(chr_levels_list), subplot_titles=labels)
            for count, (level, label) in enumerate(zip(chr_levels_list, labels)):
                line_plot = level.filter(**filters).line_plot(smooth=self.args.smooth)
                # box_plot = level.filter(**filters).box_plot()

                line_plot.draw_plotly(lp_fig, label=label)
                # box_plot.draw_plotly(bp_fig, fig_rows=count + 1, fig_cols=1)

            context_report.plots.append(TemplatePlot(
                "Line plot",
                lp_fig.to_html(full_html=False, include_plotlyjs=self.__add_plotlyjs)
            ))
            if self.__add_plotlyjs: self.__add_plotlyjs = False
            # context_report.plots.append(TemplatePlot(
            #     "Box plot",
            #     bp_fig.to_html(full_html=False, include_plotlyjs=self.__add_plotlyjs)
            # ))
            html_body.context_reports.append(context_report)

            collect()

        return asdict(html_body)
