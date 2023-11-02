import argparse
import os
import traceback
from datetime import datetime
from matplotlib.pyplot import close

parser = argparse.ArgumentParser(
    prog='BismarkPlot.',
    description='Metagene visualizing tool.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('filename', help='path to bismark methylation_extractor files', nargs='+')
parser.add_argument('-o', '--out', help='output base name', default="plot", metavar='NAME')
parser.add_argument('--dir', help='output dir', default=os.path.abspath(os.getcwd()), metavar='DIR')
parser.add_argument('-g', '--genome', help='path to GFF genome file')
parser.add_argument('-r', '--region', help='path to GFF genome file', default="gene", choices=["gene", "exon", "tss", "tes"])
parser.add_argument('-b', '--batch', help='number of rows to be read from bismark file by batch', type=int, default=10**6)
parser.add_argument('-c', '--cores', help='number of cores to use', type=int, default=None)
parser.add_argument('-f', '--flength', help='length in bp of flank regions', type=int, default=2000)
parser.add_argument('-u', '--uwindows', help='number of windows for upstream', type=int, default=50)
parser.add_argument('-d', '--dwindows', help='number of windows for downstream', type=int, default=50)
parser.add_argument('-m', '--mlength', help='minimal length in bp of gene', type=int, default=4000)
parser.add_argument('-w', '--gwindows', help='number of windows for genes', type=int, default=100)

parser.add_argument('--line', help='line-plot enabled', action='store_true', default=False)
parser.add_argument('--heatmap', help='heat-map enabled', action='store_true', default=False)
parser.add_argument('--box', help='box-plot enabled', action='store_true', default=False)
parser.add_argument('--violin', help='violin-plot enabled', action='store_true', default=False)

parser.add_argument('-S', '--smooth', help='windows for smoothing', type=float, default=10)
parser.add_argument('-L', '--labels', help='labels for plots', nargs='+')
parser.add_argument('-C', '--confidence', help='probability for confidence bands for line-plot. 0 if disabled', type=float, default=0)
parser.add_argument('-H', help='vertical resolution for heat-map', type=int, default=100, dest="vresolution")
parser.add_argument('-V', help='vertical resolution for heat-map', type=int, default=100, dest="hresolution")
parser.add_argument("--dpi", help="dpi of output plot", type=int, default=200)

parser.add_argument('-F', '--format', help='format of output plots', choices=['png', 'pdf', 'svg'], default='pdf', dest='file_format')


def main():
    args = parser.parse_args()
    if args.genome is None:
        print("You need to specify genome path")
        exit()

    try:
        from .BismarkPlot import MetageneFiles, Genome
        genome = Genome.from_gff(
            file=args.genome
        )
        if args.region == "tss":
            genome = genome.near_TSS(min_length = args.mlength, flank_length= args.flength)
        elif args.region == "tes":
            genome = genome.near_TES(min_length = args.mlength, flank_length= args.flength)
        elif args.region == "exon":
            genome = genome.exon(min_length = args.mlength)
        else:
            genome = genome.gene_body(min_length = args.mlength, flank_length= args.flength)

        bismark = MetageneFiles.from_list(
            filenames=args.filename,
            genome=genome,
            labels=args.labels,
            gene_windows=args.gwindows,
            upstream_windows=args.uwindows,
            downstream_windows=args.dwindows,
            batch_size=args.batch,
            cpu=args.cores
        )

        filename = args.dir + "/" + args.out
        print(f"Base name for saving: {filename}_<...>.{args.file_format}")

        for context in ["CG", "CHG", "CHH"]:
            for strand in ["+", "-"]:

                filtered = bismark.filter(context=context, strand=strand)
                base_name = filename + "_" + context + strand + "_{type}." + args.file_format

                if args.line:
                    fig = filtered.line_plot().draw(smooth=args.smooth, confidence=args.confidence)
                    fig.savefig(base_name.format(type="line-plot"), dpi = args.dpi)
                    close()
                if args.heatmap:
                    fig = filtered.heat_map(args.hresolution, args.vresolution).draw()
                    fig.savefig(base_name.format(type="heat-map"), dpi=args.dpi)
                    close()
                if args.box:
                    fig = filtered.trim_flank().box_plot()
                    fig.savefig(base_name.format(type="box-plot"), dpi=args.dpi)
                    close()
                if args.violin:
                    fig = filtered.trim_flank().violin_plot()
                    fig.savefig(base_name.format(type="violin-plot"), dpi=args.dpi)
                    close()


    except Exception:
        filename = args.dir + '/' + f'error{datetime.now().strftime("%m_%d_%H:%M")}.txt'
        with open(filename, 'w') as f:
            f.write(traceback.format_exc())
        print(f'Error happened. Please open an issue at GitHub with Traceback from file: {filename}')


if __name__ == "__main__":
    main()