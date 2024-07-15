from __future__ import annotations

import gzip
import itertools
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from numba import njit


@njit
def convert_trinuc(trinuc, reverse=False):
    """Get trinucleotide context from raw trinucleotide

    Parameters
    ----------
    trinuc
        trinucleotide sequence
    reverse
        is trinucleotide from reversed sequence

    Returns
    -------
    unknown
        trinucleotide context
    """
    if reverse:
        if   trinuc[1] == "C": return "CG"
        elif trinuc[0] == "C": return "CHG"
        else:                  return "CHH"
    else:
        if   trinuc[1] == "G": return "CG"
        elif trinuc[2] == "G": return "CHG"
        else:                  return "CHH"

@njit
def convert_reverse(trinuc):
    out = ""
    for nuc in trinuc:
        if nuc == "C": out += "G"
        if nuc == "G": out += "C"
        if nuc == "A": out += "T"
        if nuc == "T": out += "A"
        if nuc == "U": out += "R"
        if nuc == "R": out += "U"
        if nuc == "Y": out += "K"
        if nuc == "K": out += "Y"
        if nuc == "S": out += "S"
        if nuc == "W": out += "W"
        if nuc == "B": out += "N"
        if nuc == "D": out += "N"
        if nuc == "H": out += "N"
        if nuc == "V": out += "N"
        if nuc == "N": out += "N"
    return out

@njit
def get_trinuc(record_seq: str, reverse=False):
    """Parse sequence and extract trinucleotide contexts and positions

    Parameters
    ----------
    record_seq
        sequence
    reverse
        does sequence need to be reversed

    Returns
    -------
    unknown
        tuple(positions, contexts)
    """
    positions = []
    trinucs = []
    contexts = []

    record_seq = record_seq.upper()

    nuc = "G" if reverse else "C"
    up_shift = 1 if reverse else 3
    down_shift = -2 if reverse else 0

    for position in range(2 if reverse else 0, len(record_seq) if reverse else len(record_seq) - 2):
        if record_seq[position] == nuc:
            positions.append(position + 1)
            trinuc = record_seq[position + down_shift:position + up_shift]
            if reverse:
                trinucs.append(convert_reverse(trinuc)[::-1])
            else:
                trinucs.append(trinuc)
            contexts.append(convert_trinuc(trinuc, reverse))

    return positions, trinucs, contexts


def possible_trinucs():
    nuc_symbols = ["A", "C", "G", "T", "U", "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N"]
    possible_trinuc = ["C" + "".join(product) for product in itertools.product(nuc_symbols, repeat=2)]
    return possible_trinuc


def init_tempfile(temp_dir, name, delete, suffix=".bedGraph.parquet") -> Path:
    """Init temporary cytosine file

    Parameters
    ----------
    temp_dir
        directory where file will be created
    name
        filename
    delete
        does file need to be deleted after script completion

    Returns
    -------
    unknown
        temporary file
    """
    # temp cytosine file
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, delete=delete)

    # change name if not None
    if name is not None:
        new_path = Path(temp_dir) / (Path(name).stem + suffix)

        os.rename(temp_file.name, new_path)
        temp_file.name = new_path

    return Path(temp_file.name)


class SequenceFile:
    """
    Class for working with fasta genome sequence file.

    Parameters
    ----------
    file
        Path to FASTA genome sequence file.
    """
    def __init__(self, file: str | Path):
        self.file = Path(file).expanduser().absolute()

    @property
    def _handle(self):
        if self.file.suffix == ".gz":
            return gzip.open(self.file, 'rt')
        else:
            return open(self.file.absolute())

    @property
    def record_ids(self):
        """

        Returns
        -------
        list
            List, containing chromosomes ids.
        """
        ids = [record.id for record in SeqIO.parse(self._handle, format="fasta")]
        self._handle.seek(0)
        return ids

    cytosine_file_schema = pa.schema([
            ("chr", pa.dictionary(pa.int8(), pa.utf8())),
            ("position", pa.int32()),
            ("strand", pa.bool_()),
            ("context", pa.dictionary(pa.int16(), pa.utf8())),
            ("trinuc", pa.dictionary(pa.int16(), pa.utf8()))
        ])

    def close(self):
        """
        Close FASTA file.
        """
        self._handle.close()

    @staticmethod
    def _unify_dictionaries(table: pa.Table, dummy_table: pa.Table):
        return pa.concat_tables([dummy_table, table]).unify_dictionaries()[len(dummy_table):]

    @property
    def _dummy_table(self) -> pa.Table:
        contexts = ["CG", "CHG", "CHH"]
        chrom_ids, trinucs, contexts = list(zip(*itertools.zip_longest(possible_trinucs(), self.record_ids, contexts)))
        max_length = len(chrom_ids)

        schema_table = pa.Table.from_arrays(
            arrays=[
                [chrom if chrom is not None else chrom_ids[0] for chrom in chrom_ids],
                list(itertools.repeat(-1, max_length)),
                list(itertools.repeat(True, max_length)),
                [context if context is not None else contexts[0] for context in contexts],
                [trinuc if trinuc is not None else trinucs[0] for trinuc in trinucs]
            ],
            schema=self.cytosine_file_schema
        )

        return schema_table

    def preprocess_cytosines(self, output_file):
        """
        Write cytosine chromosome, position, strand, trinucleotide
        sequence and context into parquet file for further use in BSXplorer.

        Parameters
        ----------
        output_file
            Path, where cytosine file will be written.
        """
        with pq.ParquetWriter(output_file, self.cytosine_file_schema) as writer:
            for chr_record in SeqIO.index(self._handle, "fasta"):
                sequence_region = SequenceRegion(chr_record)
                table = sequence_region.parse_cytosines()
                unified = self._unify_dictionaries(table, self._dummy_table)
                writer.write(unified)

                print(f"Read chromosome: {chr_record.id}\t+", end="\r")

        self._handle.seek(0)



class SequenceRegion:
    def __init__(self, record: SeqIO.SeqRecord):
        self.record = record
        self.sequence = str(record.seq)

    def parse_cytosines(self):
        positions, trinucs, contexts = zip(*[get_trinuc(self.sequence), get_trinuc(self.sequence, reverse=True)])
        length = len(positions)

        arrow_table = pa.Table.from_arrays(
            arrays=[
                list(itertools.repeat(self.record.id, length * 2)),
                positions,
                list(itertools.repeat(True, length)) + list(itertools.repeat(True, length)),
                contexts,
                trinucs
            ],
            schema=SequenceFile.cytosine_file_schema
        ).sort_by("position")

        return arrow_table


class CytosinesFileCM:
    def __init__(self, path: str | Path, temp_dir: str = Path.cwd(), save: bool = False):
        self.path = Path(path).expanduser().absolute()
        self.save = save
        self.temp_dir = temp_dir

        self.is_cytosine = self.check_pq(path)

    @staticmethod
    def check_pq(path):
        try:
            pq.read_metadata(path)
            return True
        except Exception:
            return False

    def __enter__(self):
        if self.is_cytosine:
            self.cytosine_path = self.path
        else:
            self.temp_file = tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=not self.save)
            self.cytosine_path = self.temp_dir / Path(self.path.name + ".parquet")
            Path(self.temp_file.name).rename(self.cytosine_path)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_cytosine:
            self.temp_file.close()
