import re
from os.path import getsize

import numpy as np
from matplotlib.axes import Axes


def remove_extension(path):
    re.sub("\.[^./]+$", "", path)


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
