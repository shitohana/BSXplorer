from .MetageneClasses import Metagene, MetageneFiles
from .Plots import LinePlot, HeatMap, PCA, BoxPlot
from .Binom import BinomialData, RegionStat
from .GenomeClass import (
    Genome,
    Enrichment,
    EnrichmentResult,
    align_regions,
    RegAlignResult,
)
from .ChrLevelsClass import ChrLevels
from .UniversalReader_classes import (
    UniversalReader,
    UniversalWriter,
    UniversalReplicatesReader,
    UniversalBatch,
)
from .SeqMapper import SequenceFile
from . import Config

__all__ = [
    "Metagene",
    "MetageneFiles",
    "LinePlot",
    "HeatMap",
    "PCA",
    "BoxPlot",
    "BinomialData",
    "RegionStat",
    "Genome",
    "Enrichment",
    "EnrichmentResult",
    "align_regions",
    "RegAlignResult",
    "ChrLevels",
    "UniversalReader",
    "UniversalWriter",
    "UniversalReplicatesReader",
    "UniversalBatch",
    "SequenceFile",
    "Config",
]


Config._enable_string_cache()
Config.set_polars_threads(1)
