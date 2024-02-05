from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import warnings
from collections import namedtuple
from dataclasses import asdict
from gc import collect
from io import BytesIO
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt

sys.path.insert(0, os.getcwd())
from src.bsxplorer import Genome, Metagene, MetageneFiles
from src.bsxplorer.Plots import PCA
from cons_utils import render_template, TemplateMetagenePlot, TemplateMetageneContext, TemplateMetageneBody

# TODO add plot data export option

ReportRow = namedtuple(
    "ReportRow",
    ["name", "report_file", "genome_file", "flank_length", "min_length", "region_type", "genome_type",
     "report_type"]
)

def get_metagene_parser():
    parser = argparse.ArgumentParser(
        prog='BSXplorer',
        description='Metagene report creation tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('-o', '--out', help='Output filename', default=f"Metagene_Report_{time.strftime('%d-%m-%y_%H-%M-%S')}", metavar='NAME')
    parser.add_argument('--dir', help='Output and working dir', default=os.path.abspath(os.getcwd()), metavar='DIR')
    parser.add_argument('-m', '--block_mb', help='Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)', type=int, default=50)
    parser.add_argument('-t', '--threads', help='Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`', type=int, default=True)
    parser.add_argument('-s', '--sumfunc', help='Summary function to calculate density for window with.', type=str, default="wmean")
    parser.add_argument('-u', '--ubin', help='Number of windows for upstream region', type=int, default=50)
    parser.add_argument('-d', '--dbin', help='Number of windows for downstream downstream', type=int, default=50)
    parser.add_argument('-b', '--bbin', help='Number of windows for body region', type=int, default=100)
    parser.add_argument('-q', '--quantile', help='Quantile of most varying genes to draw on clustermap', type=float, default=.75)

    parser.add_argument('-S', '--smooth', help='Windows for SavGol function.', type=float, default=10)
    parser.add_argument('-C', '--confidence', help='Probability for confidence bands for line-plot. 0 if disabled', type=float, default=.95)
    parser.add_argument('-H', help='Vertical resolution for heat-map', type=int, default=100, dest="vresolution")
    parser.add_argument('-V', help='Vertical resolution for heat-map', type=int, default=100, dest="hresolution")

    parser.add_argument('--separate_strands', help='Do strands need to be processed separately', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--export', help='Export format for plots (set none to disable)', type=str, default='pdf', choices=['pdf', 'svg', 'none'])
    parser.add_argument('--ticks', help='Names of ticks (5 labels with ; separator in " brackets)', type=lambda val: str(val).replace("\\", ""), nargs=5)

    return parser


def parse_config(path: str | Path):
    report_args = pl.read_csv(
        path,
        has_header=False,
        separator="\t",
        schema={
            "name": pl.Utf8,
            "report_file": pl.Utf8,
            "genome_file": pl.Utf8,
            "flank_length": pl.Int32,
            "min_length": pl.Int32,
            "region_type": pl.Utf8,
            "genome_type": pl.Utf8,
            "report_type": pl.Utf8
        },
        truncate_ragged_lines=True,
        comment_char="#"
    )

    report_args = (
        report_args
        .with_row_count("id")
        .with_columns([
            pl.when(pl.col("name").is_null()).then(pl.col("id")).otherwise(pl.col("name")),
            pl.col("flank_length").fill_null(2000),
            pl.col("min_length").fill_null(0),
            pl.col("genome_type").fill_null("gff"),
            pl.col("report_type").fill_null("bismark"),
        ])
        .drop("id")
    )

    if report_args["report_file"].is_null().sum() + report_args["genome_file"].is_null().sum() > 0:
        raise ValueError("You should specify both report and genome paths for all samples")

    for path in report_args["report_file"].to_list() + report_args["genome_file"].to_list():
        if not Path(path).expanduser().absolute().exists():
            raise FileNotFoundError(path)

    return report_args


def render_metagene_report(metagene_files: MetageneFiles, args: argparse.Namespace, pca: PCA):
    body = TemplateMetageneBody("Metagene Report", [])

    if len(metagene_files.samples) > 1:
        try:
            body.context_reports.append(
                TemplateMetageneContext(
                    heading="PCA plot for all replicates",
                    caption="",
                    plots=[TemplateMetagenePlot(
                        title="",
                        data=pca.draw_plotly().to_html(full_html=False)
                    )]
                )
            )
        except ValueError:
            print("Got different annotations. No PCA will be drawn")

    if not args.separate_strands:
        filters = [("CG", None), ("CHG", None), ("CHH", None)]
    else:
        filters = [("CG", "+"), ("CHG", "+"), ("CHH", "+"), ("CG", "-"), ("CHG", "-"), ("CHH", "-")]

    for metagene_filter in filters:
        context_report = TemplateMetageneContext(
            heading=f"Context {metagene_filter[0]}{metagene_filter[1] if metagene_filter[1] is not None else ''}",
            caption="",
            plots=[]
        )

        major_ticks = [args.ticks[i] for i in [1, 3]]
        minor_ticks = [args.ticks[i] for i in [0, 2, 4]]

        filtered: MetageneFiles = metagene_files.filter(context=metagene_filter[0], strand=metagene_filter[1])

        lp = filtered.line_plot(merge_strands=not args.separate_strands)

        hm = filtered.heat_map(nrow=args.vresolution, ncol=args.hresolution)

        bp_plotly = filtered.box_plot_plotly()
        bp_trimmed_plotly = filtered.trim_flank().box_plot_plotly()

        cm = None

        if len(metagene_files.samples) > 1:
            try:
                cm = lambda: filtered.dendrogram(q=args.quantile)
            except ValueError:
                print("Got different annotations. No ClusterMap will be drawn")

        if args.export != 'none':

            base_name = "".join(map(str, filter(lambda val: val is not None, metagene_filter)))
            if cm is not None:
                cm().savefig(
                    Path(args.dir) / "plots" / (base_name + "_cm" + f".{args.export}")
                )
            lp.draw_mpl(smooth=args.smooth, confidence=args.confidence, major_labels=major_ticks, minor_labels=minor_ticks).savefig(
                Path(args.dir) / "plots" / (base_name + "_lp" + f".{args.export}")
            )
            hm.draw_mpl(major_labels=major_ticks, minor_labels=minor_ticks).savefig(
                Path(args.dir) / "plots" / (base_name + "_hm" + f".{args.export}")
            )
            filtered.box_plot().savefig(
                Path(args.dir) / "plots" / (base_name + "_bp" + f".{args.export}")
            )
            filtered.trim_flank().box_plot().savefig(
                Path(args.dir) / "plots" / (base_name + "_bp_trimmed" + f".{args.export}")
            )

            plt.close()

        if cm is not None:
            tmpfile = BytesIO()
            cm().savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            context_report.plots.append(
                TemplateMetagenePlot(
                    f"Clustermap for {args.quantile} quantile",
                    "<img src=\'data:image/png;base64,{}\'>".format(encoded)
                )
            )

        context_report.plots += [
            TemplateMetagenePlot(
                "Line plot",
                lp.draw_plotly(smooth=args.smooth, confidence=args.confidence).to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Heat map",
                hm.draw_plotly().to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Box plot with flanking regions",
                bp_plotly.to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Box plot without flanking regions",
                bp_trimmed_plotly.to_html(full_html=False)
            )
        ]

        body.context_reports.append(context_report)

        collect()

    return asdict(body)


def main():
    warnings.filterwarnings('ignore')

    parser = get_metagene_parser()
    args = parser.parse_args()
    # args = parser.parse_args('-o SingleMetageneReport --dir /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/supp_data/SingleMetagene -u 250 -d 250 -b 500 -S 50 --ticks \-2000kb TSS Body TES \+2000kb -C 0 -V 100 -H 100  /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test/new_conf.tsv'.split())

    report_args = parse_config(args.config)

    metagenes = []
    pca = PCA()
    last_genome_path = None
    last_genome = None

    print('BSXplorer MetageneReport Run:\n\n'
          'Config:\n',
          report_args,
          "\n"
          f"Working directory: {args.dir}\n"
          f"Output filename: {args.out}\n"
          f"Block size (Mb): {args.block_mb}\n"
          f"Threads: {args.threads}\n"
          f"Summary function: {args.sumfunc}\n"
          f"\nUpstream | Body | Downstream (bins): {args.ubin} | {args.bbin} | {args.dbin}\n"
          f"RUN STARTED at {time.strftime('%d/%m/%y %H:%M:%S')}\n\n"
          )

    if args.export != 'none':
        if not (Path(args.dir) / "plots").exists():
            (Path(args.dir) / "plots").mkdir()

            print("Plots will be saved in directory:", (Path(args.dir) / "plots"))

    unique_samples = report_args["name"].unique().to_list()

    for sample in unique_samples:
        reports = report_args.filter(pl.col('name') == sample)
        sample_metagenes = []

        for report in reports.iter_rows():
            report = ReportRow(*report)

            # Genome init
            if last_genome_path == report.genome_file:
                genome = last_genome
            else:
                if report.genome_type == "gff":
                    genome = Genome.from_gff(report.genome_file)

                    last_genome_path = report.genome_file
                    last_genome = genome

            # Genome filter
            if report.region_type is not None:
                genome_df = genome.other(region_type=report.region_type, min_length=report.min_length,
                                         flank_length=report.flank_length)
            else:
                genome_df = genome.all(min_length=report.min_length, flank_length=report.flank_length)

            # Metagene init
            # todo add more types
            if report.report_type == "bismark":
                metagene = Metagene.from_bismark(report.report_file, genome_df, args.ubin, args.bbin, args.dbin,
                                                 args.block_mb, args.threads, args.sumfunc)

            # PCA
            pca.append_metagene(metagene, Path(report.report_file).stem, sample)

            sample_metagenes.append(metagene)
        metagenes.append((MetageneFiles(sample_metagenes, list(map(str, range(len(sample_metagenes))))).merge(), sample))

        sample_metagenes = None
        collect()

    metagene_files = MetageneFiles([m[0] for m in metagenes], [m[1] for m in metagenes])
    collect()

    rendered = render_metagene_report(metagene_files, args, pca)

    out = Path(args.dir) / (args.out + ".html")
    # render_template(Path.cwd() / "html/MetageneTemplate.html", rendered, out)
    render_template(Path.cwd() / "src/templates/html/MetageneTemplate.html", rendered, out)

