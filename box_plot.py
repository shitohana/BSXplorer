import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import numpy as np
import polars as pl


class BoxPlot:
    def __init__(self, bismark: pl.DataFrame):
        self.bismark = (
            bismark.lazy()
            .groupby(['chr', 'start', 'context'])
            .agg(
                (pl.sum('sum') / pl.sum('count')).alias('density')
            )
            .groupby(['context'])
            .agg(pl.col('density'))
        ).collect()


def boxplot_data(bismark: pl.DataFrame):
    bismark = (
        bismark.lazy()
        .groupby(['chr', 'start', 'context'])
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
        .groupby(['context'])
        .agg(pl.col('density'))
    ).collect()

    return bismark


def draw_boxplot(data: list, out_dir='', labels: list = None):
    plt.clf()
    contexts = {
        'CG': {'data': []}, 'CHG': {'data': []}, 'CHH': {'data': []}
    }
    for i in range(len(data)):
        label = str(i)
        if labels is not None:
            label = labels[i]
        for context in contexts.keys():
            contexts[context]['data'].append(data[i].filter(pl.col('context') == context)['density'].to_list()[0])
            contexts[context]['label'] = label

    count = 1
    for context in contexts:
        x_pos = []
        for _ in contexts[context]['data']:
            x_pos.append(count)
            count += 1
        contexts[context]['x_pos'] = x_pos
        count += 1

    bplots = []
    for context in contexts:
        bplots.append(
            plt.boxplot(
                contexts[context]['data'], positions = contexts[context]['x_pos'], widths = 0.6, showfliers=False, patch_artist=True,
            )
        )

    colors = mpl_colors.TABLEAU_COLORS
    for bplot in bplots:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot['medians']:
            median.set_color('black')

    x_ticks = [np.mean(contexts[context]['x_pos']) for context in contexts.keys()]
    plt.xticks(x_ticks, list(contexts.keys()))

    if labels is not None:
        lines = []
        for (_, color) in zip(contexts['CG']['data'], colors):
            line, = plt.plot([1, 1], color)
            lines.append(line)
        plt.legend(lines, labels)
        [line.set_visible(False) for line in lines]
    plt.gcf().set_size_inches(7, 5)
    plt.ylabel('Methylation density')
    plt.savefig(f'{out_dir}/bar.png', dpi=300)