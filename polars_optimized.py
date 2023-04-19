import multiprocessing
import time

import pandas as pd
from scipy import signal
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors as mpl_colors


def read_genome(file: str, columns: list = None, flank_length: int = 2000, min_length: int = 4000):
    if columns is None:
        columns = [0, 2, 3, 4, 6]

    genes = pl.read_csv(
        file,
        comment_char='#',
        columns=columns,
        has_header=False,
        separator='\t',
        new_columns=['chr', 'type', 'start', 'end', 'strand'],
        dtypes={'start': pl.Int32, 'end': pl.Int32}
    ).filter(pl.col('type') == 'gene').drop('type')

    if flank_length is not None:
        genes = genes.filter(pl.col('start') > 2000)
    if min_length is not None:
        genes = genes.filter(pl.col('end') - pl.col('start') > 4000)

    genes = (
        genes.lazy()
        .groupby(['chr', 'strand'], maintain_order=True).agg([
            pl.col('start'),
            pl.col('end'),
            # upstream shift
            (pl.col('start').shift(-1) - pl.col('end'))
            .shift(1)
            .fill_null(flank_length)
            .alias('upstream'),
            # downstream shift
            (pl.col('start').shift(-1) - pl.col('end'))
            .fill_null(flank_length)
            .alias('downstream')
        ])
        .explode(['start', 'end', 'upstream', 'downstream'])
        .with_columns([
            (pl.col('start') - pl.when(
                pl.col('upstream') >= flank_length
            ).then(
                flank_length
            ).otherwise(
                (pl.col('upstream') - pl.col('upstream') % 2) // 2
            )).alias('upstream'),
            (pl.col('end') + pl.when(
                pl.col('downstream') >= flank_length
            ).then(
                flank_length
            ).otherwise(
                (pl.col('downstream') - pl.col('downstream') % 2) // 2
            )).alias('downstream')
        ])
    ).collect()
    return genes


def read_bismark_batches(
        file: str, genome: pl.DataFrame, flank_windows=500, gene_windows=2000, batch_size: int = 10 ** 6
) -> pl.DataFrame:
    cpu_count = multiprocessing.cpu_count()
    with pl.StringCache():
        total = None
        genome = genome.with_columns(
            [
                pl.col('strand').cast(pl.Categorical),
                pl.col('chr').cast(pl.Categorical)
            ]
        )
        bismark = pl.read_csv_batched(
            file,
            separator='\t', has_header=False,
            new_columns=['chr', 'position', 'strand', 'count_m', 'count_um', 'context'],
            columns=[0, 1, 2, 3, 4, 5],
            batch_size=batch_size
        )
        batches = bismark.next_batches(cpu_count)
        while batches:
            for df in batches:
                df = (
                    df.lazy()
                    .filter((pl.col('count_m') + pl.col('count_um') != 0))
                    # calculate density for each cytosine
                    .with_columns([
                        pl.col('position').cast(pl.Int32),
                        pl.col('count_m').cast(pl.Int8),
                        pl.col('count_um').cast(pl.Int8),
                        pl.col('chr').cast(pl.Categorical),
                        pl.col('strand').cast(pl.Categorical),
                        pl.col('context').cast(pl.Categorical),
                        ((pl.col('count_m')) / (pl.col('count_m') + pl.col('count_um'))).alias('density')
                    ])
                    # delete redundant columns
                    .drop(['count_m', 'count_um'])
                    # join on nearest start for every row
                    .join_asof(
                        genome.lazy(), left_on='position', right_on='upstream', by=['chr', 'strand']
                    )
                    # limit by end of gene
                    .filter(pl.col('position') <= pl.col('downstream'))
                    .with_columns(
                        # upstream
                        pl.when(
                            pl.col('position') < pl.col('start')
                        ).then(
                            (((pl.col('position') - pl.col('upstream')) /
                              (pl.col('start') - pl.col('upstream'))
                              ) * flank_windows).floor()
                        )
                        # gene body
                        .when(
                            (pl.col('start') <= pl.col('position')) & (pl.col('position') <= pl.col('end'))
                        ).then(
                            (((pl.col('position') - pl.col('start'))
                              / (pl.col('end') - pl.col('start') + 1e-10)
                              ) * gene_windows).floor() + flank_windows
                        )
                        # downstream
                        .when(
                            (pl.col('position') > pl.col('end'))
                        ).then(
                            (((pl.col('position') - pl.col('end'))
                              / (pl.col('downstream') - pl.col('end') + 1e-10)
                              ) * flank_windows).floor() + flank_windows + gene_windows
                        )
                        .cast(pl.Int32).alias('fragment')
                    )
                    .groupby(
                        by=['chr', 'strand', 'start', 'context', 'fragment']
                    )
                    .agg([
                        pl.sum('density').alias('sum'),
                        pl.count('density').alias('count')
                    ])
                    .drop_nulls(subset=['sum'])
                ).collect()
                if total is None:
                    total = df
                else:
                    total = total.extend(df)
            batches = bismark.next_batches(cpu_count)
    return total


def line_plot_data(bismark: pl.DataFrame):
    density = (
        bismark.lazy()
        .groupby(
            ['fragment', 'context', 'strand']
        )
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
    ).collect()

    fragments = density.max()['fragment'].to_list()[0] + 1

    density = (
        density.lazy()
        .groupby(['context', 'strand'])
        .agg(
            pl.arange(0, fragments, dtype=density.schema['fragment'])
            .alias('fragment')
        )
        .explode('fragment')
        .join(
            density.lazy(),
            on=['context', 'strand', 'fragment'],
            how='left'
        )
        .sort('fragment')
        .interpolate()
    ).collect()

    return density


def line_plot_filter(density: pl.DataFrame, context: str = 'CG', strand: str = '+', smooth: float = None):
    density = density.filter(
        (pl.col('context') == context) & (pl.col('strand') == strand)
    )

    density = density['density'].to_list()

    if smooth is not None:
        density = signal.savgol_filter(density, int(len(density) * smooth), 3, mode='nearest')

    if strand == '-':
        density = np.flip(density)

    return density


def heat_map_data(bismark: pl.DataFrame, rescale: float = None, sort_type: int = 'count'):
    if sort_type == 'count':
        sort_expr = pl.col('sum').ceil().sum() / pl.col('count').sum()
    else:
        sort_expr = pl.col('sum').sum() / pl.col('count').sum()

    if rescale is not None:
        bismark = bismark.with_columns(
            (pl.col('fragment') * rescale).floor().cast(pl.Int32).alias('fragment')
        )

    density = (
        # sort
        bismark.lazy()
        .groupby(['chr', 'start', 'context', 'strand'])
        .agg(
            pl.col('fragment'), pl.col('sum'), pl.col('count'),
            sort_expr.alias('sort')
        )
    )

    return density.collect()


def heat_map_filter(density: pl.DataFrame, context: str = 'CG', strand: str = '+', resolution: int = 300):
    density = density.filter(
        (pl.col('context') == context) & (pl.col('strand') == strand)
    )

    density = (
        density.lazy()
        .sort('sort', descending=True)
        # add row count
        .with_row_count(name='row')
        # round row count
        .with_columns(
            (pl.col('row') / (pl.col('row').max() + 1) * resolution).floor().alias('row').cast(pl.Int16)
        )
        .explode(['fragment', 'sum', 'count'])
        # calc sum count for row|fragment
        .groupby(['row', 'fragment', 'context', 'strand'])
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
    )

    # join with template
    fragments = density.max().collect()['fragment'].to_list()[0] + 1

    density = (
        # template
        density
        .groupby(['context', 'strand'])
        .agg(
            pl.arange(0, resolution, dtype=density.schema['row'])
            .alias('row')
        )
        .explode('row')
        .groupby(['context', 'strand', 'row'], maintain_order=True)
        .agg(
            pl.arange(0, fragments, dtype=density.schema['fragment'])
            .alias('fragment')
        )
        .explode('fragment')
        # join with orig
        .join(
            density,
            on=['context', 'strand', 'row', 'fragment'],
            how='left'
        )
        .fill_null(0)
        .sort(['row', 'fragment'])
    ).collect()

    density = np.array(
        density.groupby('row', maintain_order=True).agg(pl.col('density'))['density'].to_list(), dtype=np.float32
    )

    if strand == '-':
        density = np.fliplr(density)

    return density


def hist_data(bismark: pl.DataFrame):
    bismark = (
        bismark.lazy()
        .groupby(['chr', 'start', 'context'])
        .agg(
            (pl.sum('sum') / pl.sum('count')).alias('density')
        )
        .groupby(['context'])
        .agg(
            pl.mean('density').alias('density')
        )
    ).collect()

    return bismark


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


def read_multiple_bismark(files: list[str], genome_path: str,
                          flank_windows=500, gene_windows=2000, flank_length=2000,
                          batch_size: int = 10 ** 6,
                          heatmap=False, boxplot=False, hist=False,
                          sort_type: int = 'count',
                          rescale: float = None
                          ):
    genome = read_genome(genome_path, flank_length=flank_length)

    output = {
        'line_plot': [],
        'heatmap': [],
        'boxplot': [],
        'hist': []
    }

    for file in files:
        start = time.time()
        print(f'File: {file}')
        bismark = read_bismark_batches(file, genome, gene_windows=gene_windows, flank_windows=flank_windows,
                                       batch_size=batch_size)
        print(time.time() - start)
        start = time.time()
        output['line_plot'].append(line_plot_data(bismark))
        if heatmap:
            output['heatmap'].append(heat_map_data(bismark, rescale, sort_type))
        if boxplot:
            output['boxplot'].append(boxplot_data(bismark))
        if hist:
            output['hist'].append(hist_data(bismark))
        print(time.time() - start)
        bismark.clear()

    return output


def draw_line_plot(data: list, flank_windows: int = 0, labels: list = None, title: str = '', out_dir: str = ''):
    plt.clf()
    x = list(range(len(data[0])))
    for i in range(len(data)):
        label = None
        if labels:
            label = labels[i]
        plt.plot(x, data[i], lw=1, label=label)
    if flank_windows:
        x_ticks = [flank_windows - 1, len(data[0]) - flank_windows]
        x_labels = ['TSS', 'TES']
        plt.xticks(x_ticks, x_labels)
        for tick in x_ticks:
            plt.axvline(x=tick, linestyle='--', color='k', alpha=.3)
    if title:
        plt.title(title, fontstyle='italic')
    plt.ylabel('Methylation density')
    plt.xlabel('Position')
    plt.gcf().set_size_inches(7, 5)
    plt.legend(loc='best')
    plt.savefig(f'{out_dir}/{title}.png', dpi=300)


def draw_line_plots(data: list[pl.DataFrame], flank_windows: int = 0, labels: list = None, out_dir: str = '', smooth=.1):
    if labels is not None:
        if len(labels) != len(data):
            raise ValueError('Not enough labels')

    for context in ['CG', 'CHH', 'CHG']:
        for strand in ['+', '-']:
            line_data = [line_plot_filter(density, context, strand, smooth) for density in data]

            title = f'{context}{strand}'
            draw_line_plot(line_data, flank_windows, labels, title, out_dir)


def draw_heatmaps(data: list[pl.DataFrame], flank_windows: int = 0, labels: list = None, out_dir: str = '', resolution=300):
    if labels is not None:
        if len(labels) != len(data):
            raise ValueError('Not enough labels')

    for context in ['CG', 'CHH', 'CHG']:
        for strand in ['+', '-']:
            hm_data = [heat_map_filter(density, context, strand, resolution) for density in data]

            title = f'{context}{strand}'
            draw_heatmap(hm_data, flank_windows, labels, title, out_dir)


def draw_heatmap(data: list, flank_windows=0, labels: list = None, title='', out_dir=''):
    plt.clf()

    if len(data) > 3:
        subplots_y = 2
    else:
        subplots_y = 1
    subplots_x = len(data) // subplots_y
    if len(data) > 3:
        fig, axes = plt.subplots(subplots_y, subplots_x)
    else:
        fig, axes = plt.subplots(subplots_y, subplots_x)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    vmin = np.min(np.array(data))
    vmax = np.max(np.array(data))

    for i in range(subplots_y):
        for j in range(subplots_x):
            if subplots_y > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            im_data = data[i * subplots_x + j]
            image = ax.imshow(im_data, interpolation="nearest", aspect='auto', cmap=colormaps['cividis'],
                              vmin=vmin, vmax=vmax)
            if labels is not None:
                ax.set(title=labels[i * subplots_x + j])
            x_ticks = [flank_windows - 1, len(data[0][0]) - flank_windows]
            x_labels = ['TSS', 'TES']
            ax.set(
                xticks=x_ticks,
                xticklabels=x_labels,
                yticks=[],
                ylabel='Methylation density',
                xlabel='Position'
            )
            [ax.axvline(tick, linestyle='--', color='w', alpha=.3) for tick in x_ticks]
            plt.colorbar(image, ax=ax)

    plt.title(title, fontstyle='italic')
    fig.set_size_inches(6 * subplots_x, 5 * subplots_y)
    plt.savefig(f'{out_dir}/hm_{title}.png', dpi=300)


def draw_hist(data: list, out_dir='', labels: list = None):
    plt.clf()
    contexts = ['CG', 'CHG', 'CHH']
    df = pd.DataFrame({'context': ['CG', 'CHG', 'CHH']})
    for i in range(len(data)):
        label = str(i)
        if labels is not None:
            label = labels[i]

        df[label] = data[i].with_columns(pl.col('context').cast(pl.Utf8)).sort('context')['density'].to_list()

    df.plot(x='context', kind='bar', stacked=False, edgecolor='k', linewidth=1)
    plt.ylabel('Methylation density')
    plt.gcf().set_size_inches(7, 5)
    plt.savefig(f'{out_dir}/hist.png', dpi=300, bbox_inches='tight')


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


if __name__ == '__main__':
    data = read_multiple_bismark(
        files=[
            f'/home/kistain/Documents/BSSEQ/MAt{name}_pe.deduplicated.CX_report.txt' for name in [
                'F3-1', 'F3-2'
            ]
        ],
        genome_path='/home/kistain/Documents/BSSEQ/flax_NCBI_gene_v2.0_202008.gff3',
        heatmap=True,
        hist=True,
        boxplot=True,
        rescale=.1,
        gene_windows=2000,
        flank_windows=500,
        flank_length=2000,
        batch_size=10 ** 6
    )
    draw_line_plots(data['line_plot'], flank_windows=500, labels=['F3-1', 'F3-2'], out_dir='test', smooth=.05)
    draw_heatmaps(data['heatmap'], flank_windows=50, labels=['F3-1', 'F3-2'], out_dir='test', resolution=300)
    draw_boxplot(data['boxplot'], labels=['F3-1', 'F3-2'], out_dir='test')
    draw_hist(data['hist'], labels=['F3-1', 'F3-2'], out_dir='test')
    print(1)
