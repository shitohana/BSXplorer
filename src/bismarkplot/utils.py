import re
from os.path import getsize

import numpy as np
from matplotlib.axes import Axes
from scipy import stats


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


def prepare_labels(major_labels: list, minor_labels: list):
    labels = dict(
        up_mid="Upstream",
        body_start="TSS",
        body_mid="Body",
        body_end="c",
        down_mid="Downstream"
    )

    if major_labels is not None and len(major_labels) == 2:
        labels["body_start"], labels["body_end"] = major_labels
    elif major_labels is not None:
        print("Length of major tick labels != 2. Using default.")
    else:
        labels["body_start"], labels["body_end"] = [""] * 2

    if minor_labels is not None and len(major_labels) != 3:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = minor_labels
    elif minor_labels is not None:
        print("Length of minor tick labels != 3. Using default.")
    else:
        labels["up_mid"], labels["body_mid"], labels["down_mid"] = [""] * 3

    return labels


def approx_batch_num(path, batch_size, check_lines=1000):
    size = getsize(path)

    length = 0
    with open(path, "rb") as file:
        for _ in range(check_lines):
            length += len(file.readline())

    return round(np.ceil(size / (length / check_lines * batch_size)))


def hm_flank_lines(axes: Axes, upstream_windows: int, gene_windows: int, downstream_windows: int):
    """
    Add flank lines to the given axis (for line plot)
    """
    x_ticks = []
    x_labels = []
    if upstream_windows > 0:
        x_ticks.append(upstream_windows - .5)
        x_labels.append('TSS')
    if downstream_windows > 0:
        x_ticks.append(gene_windows + downstream_windows - .5)
        x_labels.append('TES')

    if x_ticks and x_labels:
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labels)
        for tick in x_ticks:
            axes.axvline(x=tick, linestyle='--', color='k', alpha=.3)


def interval(sum_density: list[int], sum_counts: list[int], alpha=.95):
    """
    Evaluate confidence interval for point

    :param sum_density: Sums of methylated counts in fragment
    :param sum_counts: Sums of all read cytosines in fragment
    :param alpha: Probability for confidence band
    """
    sum_density, sum_counts = np.array(sum_density), np.array(sum_counts)
    average = sum_density.sum() / sum_counts.sum()

    normalized = np.divide(sum_density, sum_counts)

    variance = np.average((normalized - average) ** 2, weights=sum_counts)

    n = sum(sum_counts) - 1

    i = stats.t.interval(alpha, df=n, loc=average, scale=np.sqrt(variance / n))

    return {"lower": i[0], "upper": i[1]}