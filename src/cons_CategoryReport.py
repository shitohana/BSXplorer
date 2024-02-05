import os
from pathlib import Path
import time
from gc import collect
from dataclasses import asdict
import argparse

import polars as pl
from matplotlib import pyplot as plt

from cons_MetageneReport import get_metagene_parser, parse_config, ReportRow

from source.GenomeClass import Genome
from source.MetageneClasses import Metagene, MetageneFiles
from source.Binom import BinomialData
from source.utils import merge_replicates, decompress
from cons_utils import render_template, TemplateMetagenePlot, TemplateMetageneContext, TemplateMetageneBody


def render_category_report(metagene_files: MetageneFiles, args: argparse.Namespace, region_pvalues):
    body = TemplateMetageneBody("Metagene Report", [])

    if not args.separate_strands:
        filters = [("CG", None), ("CHG", None), ("CHH", None)]
    else:
        filters = [("CG", "+"), ("CHG", "+"), ("CHH", "+"), ("CG", "-"), ("CHG", "-"), ("CHH", "-")]

    for metagene_filter in filters:
        filtered: MetageneFiles = metagene_files.filter(context=metagene_filter[0], strand=metagene_filter[1])

        bm_metagene_files = []
        um_metagene_files = []

        for metagene, label in zip(filtered.samples, filtered.labels):
            if args.save_cat:
                save_name = Path(args.dir) / (label + metagene_filter[0])
            else:
                save_name = None

            bm, _, um = region_pvalues[label].categorise(context=metagene_filter[0], p_value=args.region_p, save=save_name)

            bm_metagene_files.append(metagene.filter(genome=bm))
            um_metagene_files.append(metagene.filter(genome=um))

        bm_metagene_files = MetageneFiles(bm_metagene_files, filtered.labels)
        um_metagene_files = MetageneFiles(um_metagene_files, filtered.labels)

        for category, name in zip([bm_metagene_files, um_metagene_files], ["BM", "UM"]):

            context_report = TemplateMetageneContext(
                heading=f"Context {metagene_filter[0]}{metagene_filter[1] if metagene_filter[1] is not None else ''}",
                caption=f"Category: {name}",
                plots=[]
            )

            lp = category.line_plot(merge_strands=not args.separate_strands)
            hm = category.heat_map(nrow=args.vresolution, ncol=args.hresolution)

            major_ticks = [args.ticks[i] for i in [1, 3]]
            minor_ticks = [args.ticks[i] for i in [0, 2, 4]]

            bp = category.box_plot_plotly()
            bp_trimmed = category.trim_flank().box_plot_plotly()

            if args.export != 'none':
                base_name = name + "".join(map(str, filter(lambda val: val is not None, metagene_filter)))
                lp.draw_mpl(smooth=args.smooth, confidence=args.confidence, major_labels=major_ticks, minor_labels=minor_ticks).savefig(
                    Path(args.dir) / "plots" / (base_name + "_lp" + f".{args.export}")
                )
                hm.draw_mpl(major_labels=major_ticks, minor_labels=minor_ticks).savefig(
                    Path(args.dir) / "plots" / (base_name + "_hm" + f".{args.export}")
                )
                category.box_plot().savefig(
                    Path(args.dir) / "plots" / (base_name + "_bp" + f".{args.export}")
                )
                category.trim_flank().box_plot().savefig(
                    Path(args.dir) / "plots" / (base_name + "_bp_trimmed" + f".{args.export}")
                )

                plt.close()

            context_report.plots += [
                TemplateMetagenePlot(
                    "Line plot",
                    lp.draw_plotly(smooth=args.smooth, confidence=args.confidence, major_labels=major_ticks, minor_labels=minor_ticks).to_html(full_html=False)
                ),
                TemplateMetagenePlot(
                    "Heat map",
                    hm.draw_plotly(major_labels=major_ticks, minor_labels=minor_ticks).to_html(full_html=False)
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

    parser.prog = "BSXplorer-Categorise"
    parser.description = 'BM, UM categorisation tool'

    parser.add_argument("--cytosine_p", help="P-value for binomial test to consider cytosine methylated", default=".05", type=float)
    parser.add_argument("--min_cov", help="Minimal coverage for cytosine to keep", default="2", type=int)
    parser.add_argument("--region_p", help="P-value for binomial test to consider region methylated", default=".05", type=float)
    parser.add_argument("--save_cat", help="Does categories need to be saved", default=True, type=bool, action=argparse.BooleanOptionalAction)

    # args = parser.parse_args("-o F39_test --dir /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test -u 50 -d 50 -b 100 -S 20 -C 0 -V 50 /Users/shitohana/Desktop/PycharmProjects/BismarkPlot/test/F39conf.tsv".split())
    args = parser.parse_args()

    report_args = parse_config(args.config)

    for path in report_args["genome_file"].to_list() + report_args["genome_file"].to_list():
        if not Path(path).expanduser().absolute().exists():
            raise FileNotFoundError(path)

    metagenes = []
    region_stats = {}
    last_genome_path = None
    last_genome = None

    print('BSXplorer CategoriseReport Run:\n\n'
          'Config:\n',
          report_args,
          "\n"
          f"Working directory: {args.dir}\n"
          f"Output filename: {args.out}\n"
          f"Block size (Mb): {args.block_mb}\n"
          f"Threads: {args.threads}\n"
          f"Summary function: {args.sumfunc}\n\n"
          f"Minimal coverage: {args.min_cov}\n"
          f"Cytosine P-value: {args.cytosine_p}\n"
          f"Region P-value: {args.region_p}\n\n"
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

        if len(reports) > 1:
            report_paths = reports["report_file"].to_list()
            report_type = reports["report_type"].to_list()[0]

            temp = merge_replicates(report_paths, report_type, batch_size=10**6)
            report = ReportRow(*reports.row(0))

            report = report._replace(report_file=temp.name)
            report = report._replace(report_type="parquet")
        else:
            report = ReportRow(*reports.row(0))

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

        if Path(report.report_file).suffix == ".gz":
            decompressed = decompress(Path(report.report_file).expanduser().absolute())
            report = report._replace(report_file=decompressed.name)

        # Metagene init
        binom = BinomialData.preprocess(
            file=report.report_file,
            report_type=report.report_type,
            block_size_mb=args.block_mb,
            use_threads=args.threads,
            min_coverage=args.min_cov
        )

        region_pvalues = binom.region_pvalue(genome_df, args.cytosine_p, args.threads)

        metagene = Metagene.from_binom(binom.path, genome_df, args.ubin, args.bbin, args.dbin, args.cytosine_p, args.threads)

        if binom.path.exists():
            os.remove(binom.path)

        region_stats[sample] = region_pvalues
        metagenes.append((metagene, sample))
        collect()

    metagene_files = MetageneFiles([m[0] for m in metagenes], [m[1] for m in metagenes])
    collect()

    rendered = render_category_report(metagene_files, args, region_stats)

    out = Path(args.dir) / (args.out + ".html")
    # render_template(Path.cwd() / "html/CategoryTemplate.html", rendered, out)
    render_template(Path.cwd() / "src/templates/html/CategoryTemplate.html", rendered, out)

    pass

if __name__ == "__main__":
    main()