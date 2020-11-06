from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hits.visualize

import ddr.visualize
import ddr.visualize.gene_significance

def guides(guides_df, guide_to_color=None, gene_to_color=None, gene_sets=None):
    data = guides_df.copy()
    data['color'] = guide_to_color

    guides_to_label = data.query('color != "silver"').index

    x_column = 'log2_fold_change'
    y_column = '-log10_p_relevant'

    common_kwargs = dict(x=x_column, y=y_column, c='color', linewidths=(0,), s=20, clip_on=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(data=data.loc[data.index.difference(guides_to_label)], alpha=0.5, **common_kwargs)
    if len(guides_to_label) > 0:
        ax.scatter(data=data.loc[guides_to_label], alpha=0.95, zorder=10, **common_kwargs)

    hits.visualize.label_scatter_plot(ax, x_column, y_column, 'guide',
                                      data=data.loc[guides_to_label],
                                      text_kwargs={'size': 8},
                                      initial_distance=5,
                                      color='color',
                                      avoid=False,
                                      vector='above',
                                    )

    ax.axvline(0, color='black', alpha=0.5)

    ax.set_xlabel('average log2 fold-change from non-targeting')
    ax.set_ylabel('-log10 guide p-value')

    for i, (name, gene_set) in enumerate(gene_sets.items()):
        ax.annotate(name,
                    xy=(1, 0.75),
                    xycoords='axes fraction',
                    xytext=(10, -20 * i),
                    textcoords='offset points',
                    color=gene_to_color[sorted(gene_set)[0]],
                    size=12,
                   )
        
    ax.set_ylim(0)
        
    return fig

def genes(pool, outcomes, gene_sets, interesting_genes,
          label_top_n=None,
          label_all_above=None,
          x_lims=None,
          y_lims=None,
          denominator_outcomes=None,
          label_members_of_sets=True,
          label_set_names=True,
          figsize=(10, 8),
          palette=None,
          ax=None,
          color_labels=True,
          min_x=None,
         ):
    guides_df, nt_fraction, genes_df = ddr.visualize.gene_significance.get_outcome_statistics(pool, outcomes, denominator_outcomes=denominator_outcomes)

    genes_df = genes_df.drop('negative_control', axis=0)

    gene_to_color, guide_to_color = ddr.visualize.make_guide_to_color(pool.variable_guide_library.guide_to_gene, interesting_genes, gene_sets, palette=palette)

    if label_top_n is not None:
        top_genes = defaultdict(set)

        for direction in ['up', 'down']:
            if direction == 'up':
                ascending = False
            else:
                ascending = True

            #genes = genes_df['log2 fold change'][direction].sort_values(ascending=ascending).index.values[:label_top_n]
            #top_genes[direction].update(genes)

            genes = genes_df['-log10 p'][direction].sort_values(ascending=False).index.values[:label_top_n]
            top_genes[direction].update(genes)

        for gene in top_genes['up']:
            if gene not in gene_to_color or gene_to_color[gene] == 'silver':
                gene_to_color[gene] = 'tab:red'

        for gene in top_genes['down']:
            if gene not in gene_to_color or gene_to_color[gene] == 'silver':
                gene_to_color[gene] = 'tab:blue'

        for gene in genes_df.index:
            if gene not in gene_to_color:
                gene_to_color[gene] = 'silver'

    data = genes_df.xs('relevant', axis=1, level=1).copy()

    data['color'] = pd.Series(gene_to_color)

    genes_to_label = data.query('color != "silver"').index

    if label_all_above is not None:
        high_p_genes = genes_df[genes_df['-log10 p']['relevant'] > label_all_above].index
        genes_to_label = set(genes_to_label) | set(high_p_genes)

    x_column = 'log2 fold change'
    y_column = '-log10 p'

    if min_x is not None:
        data[x_column] = np.maximum(data[x_column], min_x)

    common_kwargs = dict(x=x_column, y=y_column, c='color', linewidths=(0,), s=30, clip_on=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(data=data.loc[data.index.difference(genes_to_label)], alpha=0.5, **common_kwargs)
    if len(genes_to_label) > 0:
        ax.scatter(data=data.loc[genes_to_label], alpha=0.95, zorder=10, **common_kwargs)

    if not label_members_of_sets:
        genes_to_label = [g for g in genes_to_label if not any(g in s for s in gene_sets.values())]

    if x_lims is not None:
        ax.set_xlim(*x_lims)

    if y_lims is not None:
        ax.set_ylim(*y_lims)

    hits.visualize.label_scatter_plot(ax, x_column, y_column, 'gene',
                                      data=data.loc[genes_to_label],
                                      text_kwargs={'size': 10},
                                      initial_distance=5,
                                      distance_increment=10,
                                      color='color' if color_labels else 'black',
                                      avoid=True,
                                      vector='above',
                                     )

    ax.axvline(0, color='black', alpha=0.5)

    ax.set_xlabel('log2 fold-change from non-targeting\n(average of top 2 guides)', size=12)
    ax.set_ylabel('-log10 gene p-value', size=12)

    if label_set_names:
        for i, (name, gene_set) in enumerate(gene_sets.items()):
            ax.annotate(name,
                        xy=(0, 0.6),
                        xycoords='axes fraction',
                        xytext=(5, -20 * i),
                        textcoords='offset points',
                        color=gene_to_color[sorted(gene_set)[0]],
                        size=12,
                        ha='left',
                    )

    return fig, genes_df
