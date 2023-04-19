import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


def bar_data(bismark: pl.DataFrame):
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


def draw_bar(data: list, out_dir='', labels: list = None):
    plt.clf()
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
