import argparse
import os
import traceback
from datetime import datetime

parser = argparse.ArgumentParser(
    prog='BismarkPlot Metagene visualizing tool',
    description='A small library to plot Bismark methylation_extractor reports.'   # TODO CHANGE
)

parser.add_argument('filename', help='path to bismark methylation_extractor files', nargs='+', metavar='path/to/txt')
parser.add_argument('-o', '--out', help='output base name', default=os.path.abspath(os.getcwd()), metavar='DIR')
parser.add_argument('-g', '--genome', help='path to GFF genome file', metavar='/path/to/gff')
parser.add_argument('-r', '--region', help='path to GFF genome file', metavar='/path/to/gff', default = "gene", choices=["gene", "exon", "tss", "tes"])
parser.add_argument('-b', '--batch', help='number of rows to be read from bismark file by batch', type=int, default=10**6, metavar='N')
parser.add_argument('-c', '--cores', help='number of cores to use', type=int, default=None)
parser.add_argument('-f', '--flength', help='length in bp of flank regions', type=int, default=2000, metavar='LENGTH')
parser.add_argument('-u', '--uwindows', help='number of windows for upstream', type=int, default=50, metavar='N')
parser.add_argument('-d', '--dwindows', help='number of windows for downstream', type=int, default=50, metavar='N')
parser.add_argument('-m', '--mlength', help='minimal length in bp of gene', type=int, default=4000, metavar='LENGTH')
parser.add_argument('-w', '--gwindows', help='number of windows for genes', type=int, default=100, metavar='N')

parser.add_argument('-LP', '--line-plot', help='line-plot enabled', action='store_true')
parser.add_argument('-HM', '--heat-map', help='heat-map enabled', action='store_true')
parser.add_argument('-BX', '--box-plot', help='box-plot enabled', action='store_true')
parser.add_argument('-VL', '--violin-plot', help='bar-plot enabled', action='store_true')

parser.add_argument('-S', '--smooth', help='windows for smoothing (0 - no smoothing, 1 - straight line', type=float, default=10, metavar='FLOAT')
parser.add_argument('-L', '--labels', help='labels for plots', nargs='+', metavar='NAME')
parser.add_argument('-HR', '--hresolution', help='vertical resolution for heat-map', type=int, metavar='N', default=100)
parser.add_argument('-VR', '--vresolution', help='vertical resolution for heat-map', type=int, metavar='N', default=100)
parser.add_argument("--dpi", help="dpi of output plot", type=int, metavar="N", default=200)

parser.add_argument('-F', '--format', help='format of output plots', choices=['png', 'pdf', 'svg'], default='pdf', dest='file_format')


def main():
    args = parser.parse_args()

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

        for context in ["CG", "CHG", "CHH"]:
            for strand in ["+", "-"]:

                filtered = bismark.filter(context=context, strand=strand)
                base_name = args.out + "_" + context + strand + "_{type}." + args.format

                if args.line_plot:
                    filtered.line_plot().draw(smooth=args.smooth).savefig(base_name.format(type = "line-plot"), dpi = args.dpi)
                if args.heat_map:
                    filtered.heat_map(args.hresolution, args.vresolution).draw().savefig(base_name.format(type = "heat-map"), dpi = args.dpi)
                if args.box_plot:
                    filtered.trim_flank().box_plot().savefig(base_name.format(type = "box-plot"), dpi = args.dpi)
                if args.violin_plot:
                    filtered.trim_flank().violin_plot().savefig(base_name.format(type = "violin-plot"), dpi = args.dpi)

    except Exception:
        filename = f'error{datetime.now().strftime("%m_%d_%H:%M")}.txt'
        with open(args.out + '/' + filename, 'w') as f:
            f.write(traceback.format_exc())
        print(f'Error happened. Please open an issue at GitHub with Traceback from file: {f}')
