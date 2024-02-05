from __future__ import annotations

import argparse
import re
import time
import os
from pathlib import Path
from dataclasses import asdict
from gc import collect

import polars as pl
from plotly.express.colors import qualitative as PALETTE

from src.bsxplorer import ChrLevels
from cons_utils import render_template, TemplateMetagenePlot, TemplateMetageneContext, TemplateMetageneBody
from src.bsxplorer.utils import merge_replicates


def get_chr_parser():
    parser = argparse.ArgumentParser(
        prog='BSXplorer-ChrLevels',
        description='Chromosome methylation levels visualisation tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('-o', '--out', help='Output filename',
                        default=f"Metagene_Report_{time.strftime('%d-%m-%y_%H-%M-%S')}", metavar='NAME')
    parser.add_argument('--dir', help='Output and working dir', default=os.path.abspath(os.getcwd()), metavar='DIR')
    parser.add_argument('-m', '--block_mb',
                        help='Block size for reading. (Block size ≠ amount of RAM used. Reader allocates approx. Block size * 20 memory for reading.)',
                        type=int, default=50)
    parser.add_argument('-t', '--threads',
                        help='Do multi-threaded or single-threaded reading. If multi-threaded option is used, number of threads is defined by `multiprocessing.cpu_count()`',
                        type=int, default=True)

    parser.add_argument('-w', '--window', help="Length of windows in bp", type=int, default=10**6)
    parser.add_argument('-l', '--min_length', help="Minimum length of chromosome to be analyzed", type=int, default=10**6)
    parser.add_argument('-C', '--confidence', help='Probability for confidence bands for line-plot. 0 if disabled',
                        type=float, default=.95)
    parser.add_argument('-S', '--smooth', help='Windows for SavGol function.', type=float, default=10)


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
        },
        truncate_ragged_lines=True,
        comment_char="#"
    )

    report_args = (
        report_args
        .with_row_count("id")
        .with_columns([
            pl.when(pl.col("name").is_null()).then(pl.col("id")).otherwise(pl.col("name")),
        ])
        .drop("id")
    )

    if report_args["report_file"].is_null().sum() > 0:
        raise ValueError("You should specify both report and genome paths for all samples")

    for path in report_args["report_file"].to_list():
        if not Path(path).expanduser().absolute().exists():
            raise FileNotFoundError(path)

    return report_args


def render_chr_report(chr_levels: list[ChrLevels], args, labels: list[str]):
    body = TemplateMetageneBody("Chromosome levels report", [])

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

        figure = chr_levels[0].filter(context=metagene_filter[0], strand=metagene_filter[1]).draw_plotly(smooth=args.smooth)
        for levels in chr_levels[1:]:
            lp = levels.filter(context=metagene_filter[0], strand=metagene_filter[1]).draw_plotly(smooth=args.smooth)
            figure.add_traces(lp.data)

        for i in range(len(figure.data)):
            figure.data[i]["name"] = labels[i]
            figure.data[i].showlegend = True
            figure.data[i].line.color = PALETTE.Dark24[i]
            figure.data[i].hovertemplate = re.sub(
                '^<b>.+?</b>',
                f'<b>{labels[i]}</b>',
                figure.data[i].hovertemplate
            )

        context_report.plots.append(
            TemplateMetagenePlot(
                "Line plot",
                figure.to_html(full_html=False)
            )
        )

        body.context_reports.append(context_report)

        collect()

    return asdict(body)


def main():
    parser = get_chr_parser()
    args = parser.parse_args("-o test /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test/new_conf.tsv -l 10000 -w 50000 -C 0 -S 200".split())
    # args = parser.parse_args()

    report_args = parse_config(args.config)

    chr_levels = []

    print(
        'BSXplorer ChrLevels Run:\n\n'
        'Config:\n',
        report_args,
        "\n"
        f"Working directory: {args.dir}\n"
        f"Output filename: {args.out}\n"
        f"Block size (Mb): {args.block_mb}\n"
        f"Threads: {args.threads}\n"
        f"\nWindow size (bp): {args.window}\n"
        f"Minimal chromosome length (bp): {args.min_length}\n"
        f"RUN STARTED at {time.strftime('%d/%m/%y %H:%M:%S')}\n\n"
    )

    unique_samples = report_args["name"].unique().to_list()

    for sample in unique_samples:
        reports = report_args.filter(pl.col('name') == sample)

        if len(reports) > 1:
            report_paths = reports["report_file"].to_list()

            temp = merge_replicates(report_paths, "bismark", batch_size=10 ** 6)
            report_path = temp.name
            report_type = "parquet"
        else:
            report_path = reports.row(0)[1]
            report_type = "bismark"

        if report_type == "parquet":
            levels = ChrLevels.from_parquet(report_path, args.min_length, args.window, args.confidence)
        else:
            levels = ChrLevels.from_bismark(report_path, args.min_length, args.window, args.block_mb, args.threads, args.confidence)

        chr_levels.append(levels)

    rendered = render_chr_report(chr_levels, args, report_args["name"].to_list())

    out = Path(args.dir) / (args.out + ".html")
    # render_template(Path.cwd() / "html/ChrLevelsTemplate.html", rendered, out)
    render_template(Path.cwd() / "src/templates/html/MetageneTemplate.html", rendered, out)
