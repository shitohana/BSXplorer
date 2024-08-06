from .MetageneClasses import Metagene, MetageneFiles
from .Plots import LinePlot, HeatMap, PCA, BoxPlot
from .Binom import BinomialData, RegionStat
from .GenomeClass import Genome, Enrichment, EnrichmentResult, align_regions, RegAlignResult
from .ChrLevelsClass import ChrLevels
from .UniversalReader_classes import UniversalReader, UniversalWriter, UniversalReplicatesReader, UniversalBatch
from .BamReader import BAMReader, PivotRegion
from .SeqMapper import SequenceFile
from . import Config

Config._enable_string_cache()
Config.set_polars_threads(1)
