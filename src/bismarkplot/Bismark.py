import polars as pl
from multiprocessing import cpu_count
from .LinePlot import LinePlot
from .HeatMap import HeatMap
from .BarPlot import BarPlot
from .BoxPlot import BoxPlot
from .read_bismark import read_bismark_batches


class Bismark:
    """
    Base class for Bismark data
    """
    def __init__(
            self,
            file: str,
            genome: pl.DataFrame,
            flank_windows: int = 500,
            gene_windows: int = 2000,
            batch_size: int = 10 ** 6,
            cpu: int = cpu_count()
    ):
        """
        :param cpu: How many cores to use. Uses every physical core by default
        :param file: path to bismark genomeWide report
        :param genome: polars.Dataframe with gene ranges
        :param flank_windows: Number of windows flank regions to split
        :param gene_windows: Number of windows gene regions to split
        :param batch_size: Number of rows to read by one CPU core
        """
        if flank_windows < 0:
            flank_windows = 0
        if gene_windows < 0:
            gene_windows = 0

        self.genome  = genome
        self.bismark = read_bismark_batches(file, genome, flank_windows, gene_windows, batch_size, cpu)

        self.__flank_windows = flank_windows
        self.__gene_windows  = gene_windows

        self.__line_plot = None
        self.__heat_map  = None
        self.__bar_plot  = None
        self.__box_plot  = None

    def line_plot(self) -> LinePlot:
        """
        Getter for line plot. Calculates data once after first call.
        """
        if self.__line_plot is None:
            self.__line_plot = LinePlot(self.bismark)
        return self.__line_plot

    def heat_map(self) -> HeatMap:
        """
        Getter for heat map. Calculates data once after first call.
        """
        if self.__heat_map is None:
            self.__heat_map = HeatMap(self.bismark)
        return self.__heat_map

    def box_plot(self) -> BoxPlot:
        """
        Getter for box plot. Calculates data once after first call.
        """
        if self.__box_plot is None:
            self.__box_plot = BoxPlot(self.bismark)
        return self.__box_plot

    def bar_plot(self) -> BarPlot:
        """
        Getter for bar plot. Calculates data once after first call.
        """
        if self.__bar_plot is None:
            self.__bar_plot = BarPlot(self.bismark)
        return self.__bar_plot













