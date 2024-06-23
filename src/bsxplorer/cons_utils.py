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
from tempfile import NamedTemporaryFile

import polars as pl
from jinja2 import Template
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from . import Genome, Metagene, PCA, MetageneFiles, BinomialData, ChrLevels
from .SeqMapper import init_tempfile
from .UniversalReader_classes import UniversalWriter, UniversalReader, UniversalReplicatesReader
from .utils import merge_replicates, decompress


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
        prog='BSXplorer',
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
        prog='BSXplorer-ChrLevels',
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
    parser.add_argument('-S', '--smooth', help='Windows for SavGol function.', type=float, default=10)

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

        parser.prog = "BSXplorer-Categorise"
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

            temp = None
            decompressed = None

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

            if decompressed is not None:
                decompressed.close()

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
        "parquet": ChrLevels.from_parquet,
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
                temp = merge_replicates(report_files, report_types[0], _config["MERGE_BATCH_SIZE"])

                merged_path = Path(temp.name)
                merged_type = "parquet"
            else:
                merged_path = Path(report_files[0])
                merged_type = report_types[0]

            kwargs = dict(
                file=merged_path,
                chr_min_length=self.args.min_length,
                window_length=self.args.window,
                confidence=self.args.confidence
                # threads = self.args.threads
            )

            if merged_type not in ["parquet"]:
                kwargs |= dict(
                    block_size_mb=self.args.block_mb,
                    threads=self.args.threads
                )

            levels = self._constructors[merged_type](**kwargs)
            sample_levels.append(levels)
            sample_labels.append(row["name"])

            if temp is not None: temp.close()

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
    def _p2html(fig: go.Figure, full_html=False, include_plotlyjs=False):
        return fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, default_width="900px", default_height="675px")

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

    def category_context_block(self, bm, other, um, filters):
        filter_name = self._format_filters(filters)

        context_block = TemplateContext(
            heading=f"Context {filter_name}"
        )

        tick_args = dict(
            major_labels=[self.args.ticks[i] for i in [1, 3]],
            minor_labels=[self.args.ticks[i] for i in [0, 2, 4]]
        )

        # Matplotlib block
        if self.args.export not in ["none"]:
            for mfiles, mtype in zip([bm, other, um], ["BM", "other", "UM"]):
                line_plot = mfiles.line_plot(merge_strands=not self.args.separate_strands)
                heat_map = mfiles.heat_map(nrow=self.args.vresolution, ncol=self.args.hresolution)

                name = filter_name + f"_{mtype}"

                fig = line_plot.draw_mpl(smooth=self.args.smooth, confidence=self.args.confidence, **tick_args)
                self._save_mpl(fig, name.format(type="line_plot"))

                fig = heat_map.draw_mpl(**tick_args)
                self._save_mpl(fig, name.format(type="heat-map"))

                fig = mfiles.box_plot()
                self._save_mpl(fig, name.format(type="box"))

                fig = mfiles.trim_flank().box_plot()
                self._save_mpl(fig, name.format(type="box_trimmed"))

                plt.close()

        # Plotly block
        fig = make_subplots(rows=1, cols=3, subplot_titles=["BM", "other", "UM"], shared_yaxes=True, horizontal_spacing=.05)

        for mfiles, plot_idx in zip([bm, other, um], range(1, 4)):
            lp = mfiles.line_plot().draw_plotly()

            for data in lp.data:
                fig.add_trace(data, row=1, col=plot_idx)
            # fig.update_layout({f"xaxis{plot_idx if plot_idx != 1 else ''}": lp.layout["xaxis"]})
            # fig.update_layout({f"xaxis{plot_idx if plot_idx != 1 else ''}": {"anchor": f"y{plot_idx if plot_idx != 1 else ''}"}})

        [fig.add_trace(data, row=1, col=1) for data in bm.line_plot().draw_plotly().data]
        [fig.add_trace(data, row=1, col=2) for data in other.line_plot().draw_plotly().data]
        [fig.add_trace(data, row=1, col=3) for data in um.line_plot().draw_plotly().data]

        pass





    def metagene_context_block(self, filtered: MetageneFiles, filters: dict, draw_cm: bool = True):

        filter_name = self._format_filters(filters)

        context_block = TemplateContext(
            heading=f"Context {filter_name}"
        )

        tick_args = dict(
            major_labels=[self.args.ticks[i] for i in [1, 3]],
            minor_labels=[self.args.ticks[i] for i in [0, 2, 4]]
        )

        line_plot = filtered.line_plot(merge_strands=not self.args.separate_strands)
        heat_map = filtered.heat_map(nrow=self.args.vresolution, ncol=self.args.hresolution)

        # Matplotlib block
        if self.args.export not in ["none"]:
            name = filter_name + "_{type}"

            fig = line_plot.draw_mpl(smooth=self.args.smooth, confidence=self.args.confidence, **tick_args)
            self._save_mpl(fig, name.format(type="line_plot"))

            fig = heat_map.draw_mpl(**tick_args)
            self._save_mpl(fig, name.format(type="heat-map"))

            fig = filtered.box_plot()
            self._save_mpl(fig, name.format(type="box"))

            fig = filtered.trim_flank().box_plot()
            self._save_mpl(fig, name.format(type="box_trimmed"))

            plt.close()

            if len(filtered.samples) > 1 and draw_cm:
                fig = filtered.dendrogram(q=self.args.quantile)
                self._save_mpl(fig, name.format(type="sample-cluster"))

        # Plotly block
        # Clustermap
        if len(filtered.samples) > 1 and draw_cm:
            tmpfile = BytesIO()

            fig = filtered.dendrogram(q=self.args.quantile)
            fig.savefig(tmpfile, format='png')

            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

            html_plot = TemplatePlot(
                title=f"Clustermap for {self.args.quantile} quantile",
                data="<img src=\'data:image/png;base64,{}\'>".format(encoded)
            )

            context_block.plots.append(html_plot)

        # Other
        fig = line_plot.draw_plotly(smooth=self.args.smooth, confidence=self.args.confidence, **tick_args)
        html_plot = TemplatePlot("Line plot", self._p2html(fig, include_plotlyjs=self.__add_plotlyjs))
        context_block.plots.append(html_plot)

        if self.__add_plotlyjs:
            self.__add_plotlyjs = False

        fig = heat_map.draw_plotly(**tick_args)
        html_plot = TemplatePlot("Heat map", self._p2html(fig))
        context_block.plots.append(html_plot)

        fig = filtered.box_plot_plotly()
        html_plot = TemplatePlot("Box plot with flanking regions", self._p2html(fig))
        context_block.plots.append(html_plot)

        fig = filtered.trim_flank().box_plot_plotly()
        html_plot = TemplatePlot("Box plot without flanking regions", self._p2html(fig))
        context_block.plots.append(html_plot)

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
            bm_metagenes, um_metagenes, other_metagenes = [], [], []

            for sample, label in zip(filtered_metagenes.samples, filtered_metagenes.labels):
                save_name = self.args.dir / (label + self._format_filters(filters)) if self.args.save_cat else None

                # todo add filtering not only by id
                bm_ids, _, um_ids = region_pvalues[label].categorise(
                    p_value=self.args.region_p,
                    save=save_name,
                    context=filters["context"]
                )

                bm_metagenes.append(sample.filter(genome=bm_ids))
                um_metagenes.append(sample.filter(genome=um_ids))

                other_genes = set(sample.bismark["gene"].to_list()) - set(bm_metagenes[-1].bismark["gene"].to_list() + um_metagenes[-1].bismark["gene"].to_list())
                other_metagenes.append(sample.filter(coords=other_genes))

            bm_metagene_files = MetageneFiles(bm_metagenes, filtered_metagenes.labels)
            um_metagene_files = MetageneFiles(um_metagenes, filtered_metagenes.labels)
            other_metagene_files = MetageneFiles(other_metagenes, filtered_metagenes.labels)

            html_body.context_reports.append(self.category_context_block(bm_metagene_files, other_metagene_files, um_metagene_files, filters))

            # for categorised, name in zip([bm_metagene_files, um_metagene_files], ["BM", "UM"]):
            #
            #     context_block = self.metagene_context_block(categorised, filters, draw_cm=False)
            #     context_block.caption = f"Category: {name}"
            #
            #     html_body.context_reports.append(context_block)

        return asdict(html_body)

    def chr_levels(self, chr_levels_list: list[ChrLevels], labels: list[str]):
        html_body = TemplateBody("Chromosome levels report")

        filters_list = self._get_filters

        for filters in filters_list:
            context_report = TemplateContext(heading=f"Context {self._format_filters(filters)}")
            fig = go.Figure()

            for level, label in zip(chr_levels_list, labels):
                level.filter(**filters).draw_plotly(fig, smooth=self.args.smooth, label=label)

            plot_html = TemplatePlot("Line plot", fig.to_html(full_html=False, include_plotlyjs=self.__add_plotlyjs))
            if self.__add_plotlyjs:
                self.__add_plotlyjs = False

            context_report.plots.append(plot_html)
            html_body.context_reports.append(context_report)

            collect()

        return asdict(html_body)
