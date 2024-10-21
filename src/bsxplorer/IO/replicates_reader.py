from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import polars as pl

from ..misc.utils import ReportBar
from .batches import UniversalBatch
from .single_reader import UniversalReader


class UniversalReplicatesReader:
    """
    Class for reading from replicates methylation reports. The reader sums up the
    methylation counts.

    Parameters
    ----------
    readers
        List of initialized instances of :class:`UniversalReader`.

    Examples
    --------
    >>> reader1 = UniversalReader(
    ...     file="path/to/file1.txt",
    ...     report_type="bismark",
    ...     use_threads=True,
    ... )
    >>> reader2 = UniversalReader(
    ...     file="path/to/file2.txt",
    ...     report_type="bismark",
    ...     use_threads=True,
    ... )
    >>>
    >>> for batch in UniversalReplicatesReader([reader1, reader2]):
    >>>     do_something(batch)
    """

    def __init__(
        self,
        readers: list[UniversalReader],
    ):
        self.readers = readers
        self.haste_limit = 1e9
        self.bar = None

        if any(
            map(
                lambda reader: reader.report_type.name.lower() in ["bedgraph"],
                self.readers,
            )
        ):
            warnings.warn(
                "Merging bedGraph may lead to incorrect results. Please, "
                "use other report types.",
                stacklevel=1,
            )

    def __iter__(self):
        self.bar = ReportBar(max=self.full_size)
        self.bar.start()

        self._seen_chroms = []
        self._unfinished = None

        self._readers_data = {
            idx: dict(
                iterator=iter(reader),
                read_rows=0,
                haste=0,
                finished=False,
                name=reader.file,
                chr=None,
                pos=None,
            )
            for reader, idx in zip(self.readers, range(len(self.readers)))
        }

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self.readers:
            reader.__exit__(exc_type, exc_val, exc_tb)
        if self.bar is not None:
            self.bar.goto(self.bar.max)
            self.bar.finish()

    def __next__(self):
        # Key is readers' index
        reading_from = [
            key
            for key, value in self._readers_data.items()
            if (value["haste"] < self.haste_limit and not value["finished"])
        ]

        # Read batches from not hasting readers
        batches_data = {}
        for key in reading_from:
            # If batch is present, drop "density" col and set "chr" and "position"
            # sorted for further sorted merge
            try:
                batch = next(self._readers_data[key]["iterator"])
                batches_data[key] = (
                    batch.data.set_sorted(["chr", "position"])
                    .drop("density")
                    .with_columns(pl.lit([np.uint8(key)]).alias("group_idx"))
                )
                # Upd metadata
                self._readers_data[key]["read_rows"] += len(batch)
                self._readers_data[key] |= dict(
                    chr=batch.data[-1]["chr"].to_list()[0],
                    pos=batch.data[-1]["position"].to_list()[0],
                )
            # Else mark reader as finished
            except StopIteration:
                self._readers_data[key]["finished"] = True

        if not batches_data and len(self._unfinished) == 0:
            raise StopIteration
        elif batches_data:
            self.haste_limit = sum(len(batch) for batch in batches_data.values()) // 3

        # We assume that input file is sorted in some way
        # So we gather chromosomes order in the input files to understand which is last
        if self._unfinished is not None:
            batches_data[-1] = self._unfinished

        pack_seen_chroms = []
        for key in batches_data:
            batch_chrs = batches_data[key]["chr"].unique(maintain_order=True).to_list()
            [
                pack_seen_chroms.append(chrom)
                for chrom in batch_chrs
                if chrom not in pack_seen_chroms
            ]
        [
            self._seen_chroms.append(chrom)
            for chrom in pack_seen_chroms
            if chrom not in self._seen_chroms
        ]

        # Merge read batches and unfinished data with each other and then group
        merged = []
        for chrom in pack_seen_chroms:
            # Retrieve first batch (order doesn't matter) with which we will merge
            chr_merged = batches_data[list(batches_data.keys())[0]].filter(chr=chrom)
            # Get keys of other batches
            other = [key for key in batches_data if key != list(batches_data.keys())[0]]
            for key in other:
                chr_merged = chr_merged.merge_sorted(
                    batches_data[key].filter(chr=chrom), key="position"
                )
            merged.append(chr_merged)
        merged = pl.concat(merged).set_sorted("chr", "position")

        # Group merged rows by chromosome and position and check indexes that
        # have merged
        grouped = (
            merged.lazy()
            .group_by(["chr", "position"], maintain_order=True)
            .agg(
                [
                    pl.first("strand", "context", "trinuc"),
                    pl.sum("count_m", "count_total"),
                    pl.col("group_idx").explode(),
                ]
            )
            .with_columns(pl.col("group_idx").list.len().alias("group_count"))
        )

        # Finished rows are those which have grouped among all readers
        min_chr_idx = min(
            self._seen_chroms.index(reader_data["chr"])
            for reader_data in self._readers_data.values()
        )
        min_position = min(
            reader_data["pos"]
            for reader_data in self._readers_data.values()
            if reader_data["chr"] == self._seen_chroms[min_chr_idx]
        )

        # Finished if all readers have grouped or there is no chance to group because
        # position is already skipped
        marked = (
            grouped.with_columns(
                [
                    pl.col("chr")
                    .replace(self._seen_chroms, list(range(len(self._seen_chroms))))
                    .cast(pl.Int8)
                    .alias("chr_idx"),
                    pl.lit(min_position).alias("min_pos"),
                ]
            )
            .with_columns(
                pl.when(
                    (pl.col("group_count") == len(self._readers_data))
                    | (
                        (pl.col("chr_idx") < min_chr_idx)
                        | (pl.col("chr_idx") == min_chr_idx)
                        & (pl.col("position") < pl.col("min_pos"))
                    )
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("finished")
            )
            .collect()
        )

        self._unfinished = marked.filter(finished=False).select(merged.columns)

        hasting_stats = defaultdict(int)
        if len(self._unfinished) > 0:
            group_idx_stats = self._unfinished.select(
                pl.col("group_idx").list.to_struct()
            ).unnest("group_idx")
            for col in group_idx_stats.columns:
                hasting_stats[group_idx_stats[col].drop_nulls().item(0)] += (
                    group_idx_stats[col].count()
                )

        for key in self._readers_data:
            self._readers_data[key]["haste"] = hasting_stats[key]

        # Update bar
        if reading_from:
            self.bar.next(sum(self.readers[idx].batch_size for idx in reading_from))

        out = marked.filter(finished=True)
        return UniversalBatch(self._convert_to_full(out), out.to_arrow())

    @staticmethod
    def _convert_to_full(df: pl.DataFrame):
        return df.with_columns(
            (pl.col("count_m") / pl.col("count_total")).alias("density")
        ).select(UniversalBatch.schema.polars.names())

    @property
    def full_size(self):
        """

        Returns
        -------
        int
            Total size of readers' files in bytes.
        """
        return sum(map(lambda reader: reader.file_size, self.readers))
