import argparse
import os
import traceback
from datetime import datetime

parser = argparse.ArgumentParser(
    prog='BismarkPlot',
    description='A small library to plot Bismark methylation_extractor reports.'
)

parser.add_argument('filename', help='path to bismark methylation_extractor files', nargs='+', metavar='path/to/txt')
parser.add_argument('-o', '--out', help='dir, where output will be stored', default=os.path.abspath(os.getcwd()), metavar='DIR')
parser.add_argument('-g', '--genome', help='path to GFF genome file', metavar='/path/to/gff')
parser.add_argument('-b', '--batch', help='number of rows to be read from bismark file by batch', type=int, default=10**6, metavar='N')
parser.add_argument('-f', '--flength', help='length in bp of flank regions', type=int, default=2000, metavar='LENGTH')
parser.add_argument('-i', '--fwindows', help='number of windows for flank regions (for each)', type=int, default=50, metavar='N')
parser.add_argument('-m', '--mlength', help='minimal length in bp of gene', type=int, default=4000, metavar='LENGTH')
parser.add_argument('-w', '--gwindows', help='number of windows for genes', type=int, default=100, metavar='N')

parser.add_argument('-L', '--line-plot', help='line-plot enabled', action='store_true')
parser.add_argument('-H', '--heat-map', help='heat-map enabled', action='store_true')
parser.add_argument('-BX', '--box-plot', help='box-plot enabled', action='store_true')
parser.add_argument('-BR', '--bar-plot', help='bar-plot enabled', action='store_true')

parser.add_argument('-S', '--smooth', help='plot smoothness rate (0 - no smoothing, 1 - straight line', type=float, default=.05, metavar='FLOAT')
parser.add_argument('-LB', '--labels', help='labels for plots', nargs='+', metavar='NAME')
parser.add_argument('-RS', '--resolution', help='vertical resolution for heat-map', type=int, metavar='N', default=100)
parser.add_argument('-SS', '--sspecific', help='strand specificity for box plot', action='store_true')

parser.add_argument('-FF', '--format', help='format of output plots', choices=['png', 'pdf', 'svg'], default='pdf', dest='file_format')


def main():
    args = parser.parse_args()

    try:
        from .BismarkFiles import BismarkFiles
        from .read_genome import read_genome
        genome = read_genome(
            file=args.genome,
            flank_length=args.flength,
            min_length=args.mlength
        )

        bismark = BismarkFiles(
            files=args.filename,
            genome=genome,
            flank_windows=args.fwindows,
            gene_windows=args.gwindows,
            batch_size=args.batch,
            line_plot=args.line_plot,
            heat_map=args.heat_map,
            box_plot=args.box_plot,
            bar_plot=args.bar_plot
        )

        if args.line_plot:
            bismark.draw_line_plots_all(
                smooth=args.smooth,
                labels=args.labels,
                out_dir=args.out,
                file_format=args.file_format
            )
        if args.heat_map:
            bismark.draw_heat_maps_all(
                resolution=args.resolution,
                labels=args.labels,
                out_dir=args.out,
                file_format=args.file_format
            )
        if args.box_plot:
            bismark.draw_box_plot(
                strand_specific=args.sspecific,
                labels=args.labels,
                out_dir=args.out,
                file_format=args.file_format
            )
        if args.bar_plot:
            bismark.draw_bar_plot(
                labels=args.labels,
                out_dir=args.out,
                file_format=args.file_format
            )

    except Exception:
        filename = f'error{datetime.now().strftime("%m_%d_%H:%M")}.txt'
        with open(args.out + '/' + filename, 'w') as f:
            f.write(traceback.format_exc())
        print(f'Error happened. Please open an issue at GitHub with Traceback from file: {f}')
