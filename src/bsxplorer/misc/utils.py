from __future__ import annotations

import datetime
import re
from collections import OrderedDict
from fractions import Fraction
from typing import Literal

import numpy as np
import polars as pl
from progress.bar import Bar
from scipy import stats


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


MetageneSchema = dotdict(
    dict(
        chr=pl.Categorical,
        strand=pl.Categorical,
        position=pl.UInt64,
        gene=pl.Categorical,
        context=pl.Categorical,
        id=pl.Categorical,
        fragment=pl.UInt32,
        sum=pl.Float32,
        count=pl.UInt32,
    )
)


MetageneJoinedSchema = dotdict(
    dict(
        chr=pl.Categorical,
        strand=pl.Categorical,
        gene=pl.Categorical,
        start=pl.UInt64,
        id=pl.Categorical,
        context=pl.Categorical,
        fragment=pl.UInt32,
        sum=pl.Float32,
        count=pl.UInt32,
    )
)


class ReportBar(Bar):
    suffix = "%(progress2mb)d/%(max2mb)d Mb [%(elapsed_fmt)s | ETA: %(eta_fmt)s]"
    fill = "@"

    @property
    def progress2mb(self):
        return int(self.index) / (1024**2)

    @property
    def max2mb(self):
        return int(self.max) / (1024**2)

    @property
    def elapsed_fmt(self):
        return str(datetime.timedelta(seconds=self.elapsed))

    @property
    def eta_fmt(self):
        return str(datetime.timedelta(seconds=self.eta))


CONTEXTS = ["CG", "CHG", "CHH"]
ContextsType = Literal["CG", "CHG", "CHH"]
STRANDS = ["+", "-"]
StrandsType = Literal["+", "-"]
ReportTypes = Literal["bismark", "cgmap", "binom", "bedgraph", "coverage"]
AvailableSumfunc = Literal["wmean", "mean", "min", "max", "median", "1pgeom"]
AvailableBAM = Literal["bismark"]
AvailablePlotStats = Literal["mean", "wmean"]

UniversalBatchSchema = OrderedDict(
    chr=pl.Utf8,
    strand=pl.Utf8,
    position=pl.UInt64,
    context=pl.Utf8,
    trinuc=pl.Utf8,
    count_m=pl.UInt32,
    count_total=pl.UInt32,
    density=pl.Float64,
)


def fraction(n, limit):
    f = Fraction(n).limit_denominator(limit)
    return f.numerator, f.denominator


fraction_v = np.vectorize(fraction)


def interval_chr(sum_density: list[int], sum_counts: list[int], alpha=0.95):
    """
    Evaluate confidence interval for point

    :param sum_density: Sums of methylated counts in fragment
    :param sum_counts: Sums of all read cytosines in fragment
    :param alpha: Probability for confidence band
    """
    with np.errstate(invalid="ignore"):
        sum_density, sum_counts = np.array(sum_density), np.array(sum_counts)
        average = sum_density.sum() / len(sum_counts)

        variance = np.average((sum_density - average) ** 2)

        n = sum(sum_counts) - 1

        i = stats.t.interval(alpha, df=n, loc=average, scale=np.sqrt(variance / n))

        return {"lower": i[0], "upper": i[1]}
