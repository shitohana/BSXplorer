from . import Config
from .bam import BAMReader, PivotRegion
from .binom import BinomialData, RegionStat
from .chr_levels import ChrLevels
from .genome import (
    Enrichment,
    EnrichmentResult,
    Genome,
    RegAlignResult,
    align_regions,
)
from .metagene import Metagene, MetageneFiles
from .plots import PCA, BoxPlot, HeatMap, LinePlot
from .sequence import SequenceFile
from .universal_reader import (
    UniversalBatch,
    UniversalReader,
    UniversalReplicatesReader,
    UniversalWriter,
)

Config._enable_string_cache()
Config.set_polars_threads(1)
