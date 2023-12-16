from multiprocessing import cpu_count

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyreadr import write_rds
from scipy.signal import savgol_filter

from src.bismarkplot.utils import approx_batch_num


class ChrLevels:
    def __init__(self, df: pl.DataFrame) -> None:
        self.bismark = df
        self.plot_data = (
            df
            .sort(["chr", "window"])
            .with_row_count("fragment")
            .group_by(["chr", "fragment"], maintain_order=True)
            .agg([pl.sum("sum"), pl.sum("count")])
            .with_columns((pl.col("sum") / pl.col("count")).alias("density"))
        )

    @classmethod
    def from_file(
        cls,
        file: str,
        chr_min_length = 10**6,
        window_length: int = 10**6,
        batch_size: int = 10 ** 6,
        cpu: int = cpu_count()
    ):
        """
        Initialize ChrLevels with CX_report file

        :param file: Path to file
        :param chr_min_length: Minimum length of chromosome to be analyzed
        :param window_length: Length of windows in bp
        :param cpu: How many cores to use. Uses every physical core by default
        :param batch_size: Number of rows to read by one CPU core
        """
        cpu = cpu if cpu is not None else cpu_count()

        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand',
                         'count_m', 'count_um', 'context'],
            columns=[0, 1, 2, 3, 4, 5],
            batch_size=batch_size,
            n_threads=cpu
        )
        read_approx = approx_batch_num(file, batch_size)
        read_batches = 0

        total = None

        batches = bismark.next_batches(cpu)
        print(f"Reading from {file}")
        while batches:
            for df in batches:
                df = (
                    df.lazy()
                    .filter((pl.col('count_m') + pl.col('count_um') != 0))
                    .group_by(["strand", "chr"])
                    .agg([
                        pl.col("context"),
                        (pl.col("position") / window_length).floor().alias("window").cast(pl.Int32),
                        ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density').cast(pl.Float32),
                        (pl.max("position") - pl.min("position")).alias("length")
                    ])
                    .filter(pl.col("length") > chr_min_length)
                    .explode(["context", "window", "density"])
                    .group_by(by=['chr', 'strand', 'context', 'window'])
                    .agg([
                        pl.sum('density').alias('sum'),
                        pl.count('density').alias('count')
                    ])
                    .drop_nulls(subset=['sum'])
                ).collect()
                if total is None and len(df) == 0:
                    raise Exception(
                        "Error reading Bismark file. Check format or genome. No joins on first batch.")
                elif total is None:
                    total = df
                else:
                    total = total.extend(df)

                read_batches += 1
                print(
                    f"\tRead {read_batches}/{read_approx} batch | Total size - {round(total.estimated_size('mb'), 1)}Mb RAM", end="\r")
            batches = bismark.next_batches(cpu)
        print("DONE")

        return cls(total)

    def save_plot_rds(self, path, compress: bool = False):
        """
        Saves plot data in a rds DataFrame with columns:

        +----------+---------+
        | fragment | density |
        +==========+=========+
        | Int      | Float   |
        +----------+---------+
        """
        write_rds(path, self.plot_data.to_pandas(),
                  compress="gzip" if compress else None)

    def filter(self, context: str = None, strand: str = None, chr: str = None):
        """
        :param context: Methylation context (CG, CHG, CHH) to filter (only one).
        :param strand: Strand to filter (+ or -).
        :param chr: Chromosome name to filter.
        :return: Filtered :class:`Bismark`.
        """
        context_filter = self.bismark["context"] == context if context is not None else True
        strand_filter = self.bismark["strand"] == strand if strand is not None else True
        chr_filter = self.bismark["chr"] == chr if chr is not None else True

        if context_filter is None and strand_filter is None and chr_filter is None:
            return self
        else:
            return self.__class__(self.bismark.filter(context_filter & strand_filter & chr_filter))

    def draw(
            self,
            fig_axes: tuple = None,
            smooth: int = 10,
            label: str = None,
            linewidth: float = 1.0,
            linestyle: str = '-',
    ) -> Figure:

        if fig_axes is None:
            fig, axes = plt.subplots()
        else:
            fig, axes = fig_axes

        ticks_data = self.plot_data.group_by("chr", maintain_order = True).agg(pl.min("fragment"))

        x_lines = ticks_data["fragment"].to_numpy()
        x_lines = np.append(x_lines, self.plot_data["fragment"].max())

        x_ticks = (x_lines[1:] + x_lines[:-1]) // 2

        # get middle ticks

        x_labels = ticks_data["chr"].to_list()

        data = self.plot_data["density"].to_numpy()

        polyorder = 3
        window = smooth if smooth > polyorder else polyorder + 1

        if smooth:
            data = savgol_filter(data, window, 3, mode='nearest')

        x = np.arange(len(data))
        data = data * 100  # convert to percents
        axes.plot(x, data, label=label,
                  linestyle=linestyle, linewidth=linewidth)

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)

        axes.legend()
        axes.set_ylabel('Methylation density, %')
        axes.set_xlabel('Position')

        for tick in x_lines:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.1)

        fig.set_size_inches(12, 5)

        return fig
