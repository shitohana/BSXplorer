import argparse
import os
import traceback
from datetime import datetime

parser = argparse.ArgumentParser(
    prog='BismarkPlot',
    description='Chromosome methylation levels visualization.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('filename', help='path to bismark methylation_extractor file', metavar='path/to/txt')
parser.add_argument('-o', '--out', help='output base name', default="plot", metavar='NAME')
parser.add_argument('-d', '--dir', help='output dir', default=os.path.abspath(os.getcwd()), metavar='DIR')
parser.add_argument('-b', '--batch', help='number of rows to be read from bismark file by batch', type=int, default=10**6, metavar='N')
parser.add_argument('-c', '--cores', help='number of cores to use', type=int, default=None)
parser.add_argument('-w', '--wlength', help='number of windows for chromosome', type=int, default=10**5, metavar='N')
parser.add_argument('-m', '--mlength', help='minimum chromosome length', type=int, default=10**6, metavar='N')
parser.add_argument('-S', '--smooth', help='windows for smoothing (0 - no smoothing, 1 - straight line', type=float, default=50, metavar='FLOAT')
parser.add_argument('-F', '--fmt', help='format of output plots', choices=['png', 'pdf', 'svg'], default='pdf', dest='file_format')
parser.add_argument("--dpi", help="dpi of output plot", type=int, default=200)



def main():
    args = parser.parse_args()

    try:
        from .BismarkPlot import ChrLevels
        import matplotlib.pyplot as plt

        chr = ChrLevels.from_file(
            args.filename,
            window_length=args.wlength,
            chr_min_length=args.mlength,
            batch_size=args.batch,
            cpu=args.cores
        )

        for strand in ["+", "-"]:
            fig, axes = plt.subplots()

            for context in ["CG", "CHG", "CHH"]:
                chr.filter(strand=strand, context=context).draw((fig, axes), smooth=args.smooth, label=context)

            save_path = f"{args.dir}/{args.out}_{strand}.{args.file_format}"

            print(f"Saving to: {save_path}")

            fig.savefig(save_path, dpi=args.dpi)

    except Exception:
        filename = f'error{datetime.now().strftime("%m_%d_%H:%M")}.txt'
        file_dir = args.dir + '/' + filename
        with open(file_dir, 'w') as f:
            f.write(traceback.format_exc())
        print(f'Error happened. Please open an issue at GitHub with Traceback from file: {file_dir}')


if __name__ == "__main__":
    main()