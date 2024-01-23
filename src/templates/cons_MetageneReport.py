from __future__ import annotations

import argparse
import os
import sys
import time
from collections import namedtuple
from dataclasses import asdict
from gc import collect
from pathlib import Path

import polars as pl

sys.path.insert(0, os.getcwd())
from src.bismarkplot import Genome, Metagene, MetageneFiles
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

    parser.add_argument('-S', '--smooth', help='Windows for SavGol function.', type=float, default=10)
    parser.add_argument('-C', '--confidence', help='Probability for confidence bands for line-plot. 0 if disabled', type=float, default=.95)
    parser.add_argument('-H', help='Vertical resolution for heat-map', type=int, default=100, dest="vresolution")
    parser.add_argument('-V', help='Vertical resolution for heat-map', type=int, default=100, dest="hresolution")

    parser.add_argument('--separate_strands', help='Do strands need to be processed separately', type=bool, default=False)

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


def render_metagene_report(metagene_files: MetageneFiles, args: argparse.Namespace):
    body = TemplateMetageneBody("Metagene Report", [])

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

        filtered: MetageneFiles = metagene_files.filter(context=metagene_filter[0], strand=metagene_filter[1])

        lp = filtered.line_plot(merge_strands=not args.separate_strands).draw_plotly(smooth=args.smooth,
                                                                                     confidence=args.confidence)
        hm = filtered.heat_map(nrow=args.vresolution, ncol=args.hresolution).draw_plotly()

        bp = filtered.box_plot_plotly()
        bp_trimmed = filtered.trim_flank().box_plot_plotly()

        context_report.plots += [
            TemplateMetagenePlot(
                "Line plot",
                lp.to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Heat map",
                hm.to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Box plot with flanking regions",
                bp.to_html(full_html=False)
            ),
            TemplateMetagenePlot(
                "Box plot without flanking regions",
                bp_trimmed.to_html(full_html=False)
            )
        ]

        body.context_reports.append(context_report)

        collect()

    return asdict(body)


def main():
    parser = get_metagene_parser()
    args = parser.parse_args()
    # args = parser.parse_args("-o F39_metagene --dir /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test -u 50 -d 50 -b 100 -S 5 -C 0 -V 50 /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test/F39conf.tsv".split())

    report_args = parse_config(args.config)

    for path in report_args["genome_file"].to_list() + report_args["genome_file"].to_list():
        if not Path(path).expanduser().absolute().exists():
            raise FileNotFoundError(path)

    metagenes = []
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

            sample_metagenes.append(metagene)
        metagenes.append((MetageneFiles(sample_metagenes, list(map(str, range(len(sample_metagenes))))).merge(), sample))

        sample_metagenes = None
        collect()

    metagene_files = MetageneFiles([m[0] for m in metagenes], [m[1] for m in metagenes])
    collect()

    rendered = render_metagene_report(metagene_files, args)

    out = Path(args.out).with_suffix(".html")
    # render_template(Path.cwd() / "html/MetageneTemplate.html", rendered, out)
    render_template(Path.cwd() / "src/templates/html/MetageneTemplate.html", rendered, out)


if __name__ == '__main__':
    main()