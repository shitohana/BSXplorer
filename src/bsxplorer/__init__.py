from .MetageneClasses import Metagene, MetageneFiles
from .Plots import LinePlot, HeatMap, PCA
from .Binom import BinomialData, RegionStat
from .GenomeClass import Genome
from .ChrLevelsClass import ChrLevels
from .UniversalReader_classes import UniversalReader, UniversalWriter, UniversalReplicatesReader
from . import Config

Config._enable_string_cache()
Config.set_polars_threads(1)
