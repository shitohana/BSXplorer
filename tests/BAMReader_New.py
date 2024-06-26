import time

from src.bsxplorer import Genome, Config
from src.bsxplorer.BamReader import BAMReader
from src.bsxplorer.SeqMapper import Sequence

Config.set_polars_threads(8)
genome = Genome.from_gff("~/Documents/CX_reports/flax/Linum/genomic.gff").gene_body(0)
sequence = Sequence.from_preprocessed("/Users/shitohana/Desktop/PycharmProjects/BSXplorer/tests/Linum_sequence.parquet")
reader = BAMReader(
    "/Users/shitohana/Documents/CX_reports/flax/Linum/MAtF3-1.sorted.bam",
    index_filename="/Users/shitohana/Documents/CX_reports/flax/Linum/MAtF3-1.sorted.bam.bai",
    sequence=sequence,
    # regions=genome,
    context="CG",
    threads=8,
    batch_num=10000,
    readahead=5,
    qc=True
)

start_time = time.time()
for _ in reader.report_iter():
    pass

reader.plot_qc().savefig("test.pdf")
print("\n\n", time.time() - start_time)
