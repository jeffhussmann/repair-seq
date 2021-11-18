from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import bokeh.palettes

import hits.visualize
import hits.utilities
import repair_seq.visualize.heatmap
import repair_seq.visualize.outcome_diagrams
import repair_seq.visualize.gene_significance
import repair_seq.visualize.volcano

def plot_correlations(pool, guide, num_outcomes, label=True, extra_genes=None, n_highest=5, legend=True):

    gene = pool.variable_guide_library.guide_to_gene[guide]

    outcomes = pool.most_frequent_outcomes('none')[:num_outcomes]
    log2_fcs = pool.log2_fold_changes('perfect', 'none')['none'].loc[outcomes].drop(columns=['all_non_targeting'])
    correlations = log2_fcs.corr().loc[guide].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8 * len(correlations) / 1513, 6))

    df = pd.DataFrame({'correlation': correlations})
    df['gene'] = pool.variable_guide_library.guides_df['gene'].loc[df.index]
    df['x'] = np.arange(len(df))
    df['best_promoter'] = pool.variable_guide_library.guides_df['best_promoter']

    ax.set_xlim(-int(0.02 * len(df)), int(1.02 * len(df)))
    ax.set_ylim(-1.01, 1.01)

    ax.scatter('x', 'correlation', c='C0', s=20, data=df.query('gene == @gene'), clip_on=False, label=f'{gene} guides')
    ax.scatter('x', 'correlation', c='C1', s=15, data=df.query('gene == "negative_control"'), clip_on=False, label='non-targeting guides')
    ax.scatter('x', 'correlation', c='grey', s=1, data=df.query('gene != "negative_control" and gene != @gene'), clip_on=False, label='other guides')
    
    if legend:
        ax.legend()

    for y in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
        ax.axhline(y, color='black', alpha=0.5 if y == 0 else 0.1)

    ax.set_xlabel('guide rank', size=12)
    ax.set_ylabel(f'correlation of repair profile with {guide}', size=12)
    
    if label:
        if extra_genes is None or gene not in extra_genes:
            to_label = df.query('gene == @gene')
            to_label.index.name = 'short_name'

            hits.visualize.label_scatter_plot(ax, 'x', 'correlation', 'short_name', to_label,
                                              avoid=True,
                                              initial_distance=10,
                                              vector='sideways',
                                              color='C0',
                                             )
        
        if extra_genes is not None:
            other_guides = df.loc[df['gene'].isin(list(extra_genes)) & df['best_promoter']]
            to_label = other_guides
        else:
            other_guides = df.query('gene != "negative_control" and gene != @gene')
            to_label = pd.concat([other_guides.iloc[:n_highest], other_guides.iloc[-n_highest:]])
         
        to_label = to_label.sort_values('correlation', ascending=False)
        to_label.index.name = 'short_name'
            
        hits.visualize.label_scatter_plot(ax, 'x', 'correlation', 'short_name', to_label, avoid=True, initial_distance=10, vector='sideways')
            
    for side in ['top', 'bottom', 'right']:
        ax.spines[side].set_visible(False)
    
    return fig, df

def fraction_removed(pool, genes, fraction_y_lims=(5e-4, 1), fold_change_y_lims=(-4, 2), fixed_guide='none'):
    fig, (fraction_ax, fc_ax) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw=dict(hspace=0.05))

    guide_sets = [
        ('negative_control', 'individual non-targeting guide', dict(color='black', alpha=0.1)),
    ]

    for gene_i, gene in enumerate(genes):
        guide_sets.append((gene, f'{gene} guides', dict(color=repair_seq.visualize.good_colors[gene_i], alpha=0.8, linewidth=1.5)))

    kwargs = dict(color='black', alpha=0.9, linewidth=2, label='all non-targeting guides')
    fraction_ax.plot(pool.fraction_removed[fixed_guide, 'all_non_targeting'], '-', **kwargs)
    fc_ax.plot(pool.fraction_removed_log2_fold_changes[fixed_guide, 'all_non_targeting'], '-', **kwargs)    

    for gene, label, kwargs in guide_sets:
        guides = pool.variable_guide_library.gene_guides(gene, only_best_promoter=True)

        for i, guide in enumerate(guides):
            if i == 0:
                label_to_use = label
            else:
                label_to_use = ''
            
            ys = pool.fraction_removed[fixed_guide, guide].replace(to_replace=0, value=np.nan)
            fraction_ax.plot(ys, '-', label=label_to_use, **kwargs)
            fc_ax.plot(pool.fraction_removed_log2_fold_changes[fixed_guide, guide], '-', label=label_to_use, **kwargs)

    fraction_ax.legend()
    for ax in [fraction_ax, fc_ax]:
        ax.set_xlim(pool.fraction_removed.index[0], pool.fraction_removed.index[-1])

    fraction_ax.set_title(pool.group)
    fraction_ax.set_ylim(*fraction_y_lims)
    fraction_ax.set_yscale('log')
    fraction_ax.set_ylabel('fraction of outcomes with position deleted', size=12)

    fc_ax.set_ylim(*fold_change_y_lims)
    fc_ax.set_ylabel('log2 fold change from non-targeting', size=12)
    
    if len(pool.target_info.cut_afters) == 1:
        fc_ax.set_xlabel('distance from cut site (nts)', size=12)
    else:
        for name, position in pool.target_info.cut_afters.items():
            for ax in [fraction_ax, fc_ax]:
                ax.axvline(position - pool.target_info.cut_after, color='black', alpha=0.3)

        fc_ax.set_xlabel('distance from primary cut site (nts)', size=12)

    return fig

def compare_fraction_removed(numerator_pool,
                             denominator_pool,
                             fraction_y_lims=(5e-4, 1),
                             fold_change_y_lims=(-4, 2),
                             normalize_to_non_HDR_edited=False,
                            ):
    fig, (fraction_ax, fc_ax) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw=dict(hspace=0.05))

    pools = {'numerator': numerator_pool, 'denominator': denominator_pool}
    fractions = {}

    for k in ['numerator', 'denominator']:
        pool = pools[k]

        fraction_removed = pool.fraction_removed['all_non_targeting', 'all_non_targeting']

        if normalize_to_non_HDR_edited:
            nt_fracs = pool.non_targeting_fractions('perfect', 'none')
            normalization = 1 - nt_fracs.loc['wild type'].sum() - nt_fracs.loc['donor'].sum()
            fraction_removed = fraction_removed / normalization

        fractions[k] = fraction_removed

        fraction_ax.plot(fraction_removed, '-', linewidth=2, label=pool.short_name)

    fc_ax.plot(np.log2(fractions['numerator'] / fractions['denominator']), '-', linewidth=2, color='C1')

    fraction_ax.set_ylim(*fraction_y_lims)
    fraction_ax.set_yscale('log')
    if normalize_to_non_HDR_edited:
        title = 'fraction of edited, non-HDR\noutcomes with position deleted'
    else:
        title = 'fraction of outcomes with position deleted'
    fraction_ax.set_ylabel(title, size=12)

    fc_ax.set_ylim(*fold_change_y_lims)
    fc_ax.set_ylabel(f'log2 ratio', size=12)
    
    fc_ax.set_xlabel('distance from primary cut site (nts)', size=12)
    fc_ax.axhline(0, linewidth=2, color='C0')

    fraction_ax.legend()
    for ax in [fraction_ax, fc_ax]:
        ax.set_xlim(numerator_pool.fraction_removed.index[0], numerator_pool.fraction_removed.index[-1])

    return fig

def make_color_column(guide_to_gene, genes, full_gene_list=None, default_color='silver'):
    if full_gene_list is None:
        full_gene_list = genes

    guide_to_color = pd.Series(default_color, index=guide_to_gene.index)

    gene_to_color = {}

    for i, gene in enumerate(full_gene_list):
        if gene in genes:
            if i + 1 < 10:
                color = f'C{i + 1}'
            else:
                color = bokeh.palettes.Category20b[20][::4][(i + 1) % 10]

            gene_to_color[gene] = color

            guide_to_color[guide_to_gene == gene] = color

    guide_to_color[guide_to_gene == 'negative_control'] = 'C0'
        
    guide_to_color = guide_to_color.apply(matplotlib.colors.to_hex)
        
    return guide_to_color, gene_to_color

def scatter_and_pc(results,
                   x_column,
                   y_column,
                   genes_to_label,
                   full_gene_list=None,
                   lims=(-4, 2),
                   avoid_overlapping_labels=False,
                   pn_to_name=None,
                   column_order=None,
                   legend=True,
                   draw_labels=True,
                   guide_to_color=None,
                   gene_to_color=None,
                   guides_to_highlight=None,
                   manual_labels=None,
                   resolution_to_plot='genes',
                   scatter_genes_to_label=None,
                   **kwargs,
                  ):

    guide_to_gene = results['guides_df']['gene']

    if manual_labels is None:
        manual_labels = []

    if resolution_to_plot == 'guides':
        data = results['guides_df'].xs('log2_fold_change', axis=1, level=1).copy()

        best_promoter = pool.variable_guide_library.guides_df['best_promoter']
        if guides_to_highlight is None:
            guides_to_highlight = guide_to_gene[guide_to_gene.isin(genes_to_label) & best_promoter].index
    else:
        data = results['genes_df']

    if column_order is None:
        column_order = data.columns.values

    data = data[column_order]
        
    if gene_to_color is None and guide_to_color is None:
        guide_to_color, gene_to_color = make_color_column(guide_to_gene, genes_to_label, full_gene_list=full_gene_list)

    pc_width = kwargs.get('pc_width_per_column', 0.75) * len(column_order)
    fig, pc_ax = plt.subplots(figsize=(pc_width, kwargs.get('pc_height', 2)))
    
    if resolution_to_plot == 'genes':
        parallel_coordinates_genes(data,
                                   pc_ax,
                                   gene_to_color,
                                   genes_to_label,
                                   lims=lims,
                                   text_labels=['right'],
                                   pn_to_name=pn_to_name,
                                   legend=legend,
                                  )
    else:
        parallel_coordinates(data, pc_ax,
                             guides_to_label,
                             guide_to_gene,
                             guide_to_color,
                             lims=lims,
                             text_labels=['right'],
                             pn_to_name=pn_to_name,
                             #legend=legend,
                            )

    pc_ax_p = pc_ax.get_position()
    fig_width, fig_height = fig.get_size_inches()
    scatter_height = pc_ax_p.height
    scatter_width = fig_height / fig_width * scatter_height
    scatter_ax = fig.add_axes((pc_ax_p.x1 + 0.5 * scatter_width,
                               pc_ax_p.y0,
                               scatter_width,
                               scatter_height,
                              ),
                              sharey=pc_ax,
                             ) 

    if resolution_to_plot == 'genes':
        if scatter_genes_to_label is None:
            scatter_genes_to_label = genes_to_label

        scatter_genes(data,
                      scatter_ax,
                      x_column,
                      y_column,
                      scatter_genes_to_label,
                      gene_to_color,
                      avoid_overlapping_labels=avoid_overlapping_labels,
                      lims=lims,
                      pn_to_name=pn_to_name,
                      draw_labels=draw_labels,
                     )
    else:
        scatter(data, scatter_ax,
                x_column,
                y_column,
                guides_to_highlight,
                guide_to_color,
                avoid_overlapping_labels=avoid_overlapping_labels,
                lims=lims,
                pn_to_name=pn_to_name,
                draw_labels=draw_labels,
            )

    for label, color, xy in manual_labels:
        scatter_ax.annotate(label,
                            xy=xy,
                            xycoords='data',
                            color=color,
                            ha='center',
                            annotation_clip=False,
                           )

    fig_transform = fig.transFigure
    inverse_figure = fig_transform.inverted()
    pc_transform = pc_ax.get_xaxis_transform() + inverse_figure
    scatter_transform = scatter_ax.transAxes + inverse_figure

    def draw_path(points, transforms):
        xs, ys = np.array([t.transform_point(p) for t, p in zip(transforms, points)]).T
        pc_ax.plot(xs, ys, linewidth=0.5, transform=fig_transform, clip_on=False, color='black')

    bracket_offset = 0.03
    bracket_height = 0.025

    def draw_bracket(x):
        draw_path([(x - 0.1, -bracket_offset + bracket_height),
                   (x - 0.1, -bracket_offset),
                   (x + 0.1, -bracket_offset),
                   (x + 0.1, -bracket_offset + bracket_height),
                  ],
                  [pc_transform, pc_transform, pc_transform, pc_transform],
                 )

    y_pc_x = data.columns.get_loc(y_column)
    draw_bracket(y_pc_x)
    draw_path([(y_pc_x, -bracket_offset),
               (y_pc_x, -0.1),
               (-0.22, -0.1),
               (-0.22, 0.5),
               (-0.22 + 0.025, 0.5),
              ],
              [pc_transform, pc_transform, scatter_transform, scatter_transform, scatter_transform],
             )

    x_pc_x = data.columns.get_loc(x_column)
    draw_bracket(x_pc_x)
    draw_path([(x_pc_x, -bracket_offset),
               (x_pc_x, -0.2),
               (0.5, -0.2),
               (0.5, -0.2),
               (0.5, -0.2 + 0.025)],
              [pc_transform, pc_transform, scatter_transform, scatter_transform, scatter_transform],
             )

    return fig, pc_ax, scatter_ax

def scatter(data, ax, x_column, y_column,
            guides_to_highlight,
            guide_to_color, 
            lims=(-4, 2),
            avoid_overlapping_labels=True,
            pn_to_name=None,
            draw_labels=True,
           ):

    data = data.copy()

    data['color'] = guide_to_color

    common_kwargs = dict(x=x_column,
                         y=y_column,
                         c='color',
                         linewidths=(0,),
                        )

    to_plot = data[[x_column, y_column, 'color']].dropna()
    
    ax.scatter(data=to_plot.loc[to_plot.index.difference(guides_to_highlight)], s=15, alpha=0.5, **common_kwargs)
    ax.scatter(data=to_plot.loc[guides_to_highlight], s=35, alpha=0.95, zorder=10, **common_kwargs)

    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', alpha=0.3)
    
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal') 
    
    hits.visualize.draw_diagonal(ax, alpha=0.3)
    
    if pn_to_name is not None:
        x_label = pn_to_name[x_column.rsplit('_', 1)[0]]
        y_label = pn_to_name[y_column.rsplit('_', 1)[0]]
    else:
        x_label = x_column
        y_label = y_column

    ax.set_xlabel(x_label + '\nlog$_2$ fold change', size=12)
    ax.set_ylabel(y_label + '\nlog$_2$ fold change', size=12)
    
    if draw_labels:
        ax.annotate('non-targeting guides',
                    xy=(0, 0),
                    xycoords='data',
                    xytext=(20, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    color='C0',
                    size=10,
                )

        hits.visualize.label_scatter_plot(ax, x_column, y_column, 'guide',
                                        data=to_plot.loc[guides_to_highlight],
                                        text_kwargs={'size': 10},
                                        initial_distance=20,
                                        color='color',
                                        avoid=avoid_overlapping_labels,
                                        avoid_existing=avoid_overlapping_labels,
                                        )

def scatter_genes(data, ax, x_column, y_column,
                  genes_to_highlight,
                  gene_to_color, 
                  lims=(-4, 2),
                  avoid_overlapping_labels=True,
                  pn_to_name=None,
                  draw_labels=True,
                 ):

    data = data.copy()
    data.index.name = 'gene'

    data['color'] = [gene_to_color.get(gene, 'grey') for gene in data.index]

    data.loc[data.index.str.startswith('non-targeting'), 'color'] = 'black'

    common_kwargs = dict(x=x_column,
                         y=y_column,
                         c='color',
                         linewidths=(0,),
                        )

    to_plot = data[[x_column, y_column, 'color']].dropna()
    
    ax.scatter(data=to_plot.loc[to_plot.index.difference(genes_to_highlight)], s=10, alpha=0.5, **common_kwargs)
    ax.scatter(data=to_plot.loc[genes_to_highlight], alpha=0.95, s=15, zorder=10, **common_kwargs)

    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', alpha=0.3)
    
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal') 

    ticks = np.arange(int(np.ceil(lims[0])), int(np.floor(lims[1])) + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=6)
    
    hits.visualize.draw_diagonal(ax, alpha=0.3)
    
    if pn_to_name is not None:
        x_label = pn_to_name[x_column.rsplit('_', 1)[0]]
        y_label = pn_to_name[y_column.rsplit('_', 1)[0]]
    else:
        x_label = x_column
        y_label = y_column

    ax.set_xlabel(x_label + '\nlog$_2$ fold change', size=6)
    ax.set_ylabel(y_label + '\nlog$_2$ fold change', size=6)
    
    if draw_labels:
        ax.annotate('non-targeting\nguide sets',
                    xy=(0, 0),
                    xycoords='data',
                    xytext=(-20, 10),
                    textcoords='offset points',
                    ha='right',
                    va='bottom',
                    color='black',
                    size=6,
                )

        hits.visualize.label_scatter_plot(ax, x_column, y_column, 'gene',
                                        data=to_plot.loc[genes_to_highlight],
                                        text_kwargs={'size': 6},
                                        initial_distance=5,
                                        color='color',
                                        avoid=avoid_overlapping_labels,
                                        avoid_existing=avoid_overlapping_labels,
                                        )

def parallel_coordinates(data, ax,
                         guides_to_label=None,
                         guide_to_gene=None,
                         guide_to_color=None,
                         lims=(-4, 2),
                         text_labels=None,
                         pn_to_name=None,
                         legend=((-0.35, 1), 'upper right'),
                         y_label='log2 fold-change from all non-targeting',
                         xs=None,
                         markersize=8,
                         draw_non_targeting=True,
                         linewidth=2.5,
                         draw_all_guides=False,
                         alpha=0.9,
                         guide_to_kwargs=None,
                         **kwargs,
                        ):

    if guides_to_label is None:
        guides_to_label = []

    if guide_to_gene is None:
        guide_to_gene = pd.Series()

    if guide_to_color is None:
        guide_to_color = defaultdict(lambda: 'grey')

    if xs is None:
        xs = np.arange(len(data.columns))

    if text_labels is None:
        text_labels = []

    if pn_to_name is None:
        pn_to_name = {n.rsplit('_', 1)[0]: n.rsplit('_', 1)[0] for n in data.columns}

    genes_to_label = guide_to_gene[guides_to_label].unique()
    #for gene_i, gene in enumerate(genes_to_label, 1):
    #    guides = guides_to_label[guide_to_gene[guides_to_label] == gene]
    #    for guide_i, guide in enumerate(guides):
    #        guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
    #                                      marker='.',
    #                                      markersize=markersize,
    #                                      alpha=0.8,
    #                                      linewidth=2.5,
    #                                      label=f'{gene} guides' if guide_i == 0 else '',
    #                                      clip_on=False,
    #                                     )


    if guide_to_kwargs is None:
        guide_to_kwargs = {}

        if draw_non_targeting:
            for guide_i, guide in enumerate(guide_to_gene[guide_to_gene == 'negative_control'].index):     
                guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
                                            alpha=kwargs.get('nt_guides_alpha', 0.8),
                                            linewidth=1,
                                            label='individual\nnon-targeting\nguides' if guide_i == 0 else '',
                                            )

        for guide_i, guide in enumerate(guides_to_label):
            guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
                                        marker='.',
                                        markersize=markersize,
                                        alpha=alpha,
                                        linewidth=linewidth,
                                        clip_on=False,
                                        zorder=10,
                                        )

        if draw_all_guides:
            for guide in data.index:
                if guide not in guide_to_kwargs:
                    guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
                                                alpha=kwargs.get('other_guides_alpha', 0.2),
                                                linewidth=1,
                                                clip_on=False,
                                                )


    for guide, row in data.iterrows():
        plot_kwargs = guide_to_kwargs.get(guide)
        if kwargs is None:
            continue
        
        ax.plot(xs, row.values, clip_on=False, **plot_kwargs)

        if guide_to_gene[guide] != 'negative_control':
            if 'right' in text_labels:
                ax.annotate(guide,
                            xy=(1, row.values[-1]),
                            xycoords=('axes fraction', 'data'),
                            xytext=(5, 0),
                            textcoords='offset points',
                            color=plot_kwargs['color'],
                            ha='left',
                            va='center',
                            size=markersize,
                )

    if legend is not None:
        ax.legend(bbox_to_anchor=legend[0],
                  loc=legend[1],
                 )
        
    ax.set_ylabel(y_label, size=kwargs.get('axis_label_size', 12))

    ax.set_xticks(xs)
    labels = []
    for pn in data.columns:
        try:
            label = pn_to_name[pn]
        except KeyError:
            pn, outcome = pn.rsplit('_', 1)
            label = f'{pn_to_name[pn]} {outcome}'
        labels.append(label)
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.tick_params(labelbottom=False, labeltop=True)

    ax.set_ylim(*lims)

    all_axs = [ax]
    for x in xs[1:]:
        other_ax = ax.twinx()

        other_ax.spines['left'].set_position(('data', x))
        other_ax.yaxis.tick_left()
        other_ax.set_ylim(ax.get_ylim())
        other_ax.set_yticklabels([])
        
        all_axs.append(other_ax)

    for ax in all_axs:
        for side in ['right', 'top', 'bottom']:
            ax.spines[side].set_visible(False)

        ax.spines['left'].set_alpha(0.5)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=2, color=hits.visualize.apply_alpha('black', 0.35))

    #x_extent = xs[-1] - xs[0]
    #ax.set_xlim(xs[0] - x_extent * 0.05, xs[1] + x_extent * 0.05)

    return all_axs
        
def parallel_coordinates_genes(data,
                               ax,
                               gene_to_color,
                               genes_to_label,
                               lims=(-4, 2),
                               text_labels=None,
                               pn_to_name=None,
                               legend=True,
                               gene_aliases=None,
                              ):
    gene_to_kwargs = {}
    if gene_aliases is None:
        gene_aliases = {}

    if pn_to_name is None:
        pn_to_name = {n.rsplit('_', 1)[0]: n.rsplit('_', 1)[0] for n in data.columns}

    if text_labels is None:
        text_labels = []

    non_targeting_sets = [g for g in data.index if g.startswith('non-targeting_set')]
    for non_targeting_set in non_targeting_sets:     
        gene_to_kwargs[non_targeting_set] = dict(color='black',
                                                 alpha=0.5,
                                                 linewidth=1,
                                                )

    for gene in genes_to_label:
        gene_to_kwargs[gene] = dict(color=gene_to_color.get(gene, 'tab:blue'),
                                    marker='.',
                                    markersize=6,
                                    alpha=0.8,
                                    linewidth=1.5,
                                    label=gene,
                                    clip_on=False,
                                   )

    for gene, row in data.iterrows():
        kwargs = gene_to_kwargs.get(gene)
        if kwargs is None:
            continue
        
        ax.plot(row.values, **kwargs)

        if gene.startswith('non-targeting_set'):
            continue

        common_kwargs = dict(
            text=gene_aliases.get(gene, gene),
            color=kwargs['color'],
            textcoords='offset points',
            va='center',
            size=6,
            xycoords=('axes fraction', 'data'),
        )
        specific_kwargs_list = []
        if 'right' in text_labels:
            specific_kwargs_list.append(dict(
                xy=(1, row.values[-1]),
                xytext=(5, 0),
                ha='left',
            ))

        if 'left' in text_labels:
            specific_kwargs_list.append(dict(
                xy=(0, row.values[0]),
                xytext=(-5, 0),
                ha='right',
            ))

        for specific_kwargs in specific_kwargs_list:
            ax.annotate(**common_kwargs, **specific_kwargs)
        
    if legend:
        ax.legend(bbox_to_anchor=(-0.35, 1),
                loc='upper right',
                )
        
    ax.set_ylabel('Log$_2$ fold change from non-targeting',
                  size=6,
                  labelpad=16 if 'left' in text_labels else None,
                 )

    num_columns = len(data.columns)
    ax.set_xlim(0, num_columns - 1)
    ax.set_xticks(np.arange(num_columns))
    labels = []
    for n in data.columns:
        pn, outcome = n.rsplit('_', 1)
        label = f'{pn_to_name[pn]} {outcome}'
        labels.append(label)

    ax.set_xticklabels(labels, rotation=45, ha='left', size=6)
    ax.tick_params(labelbottom=False, labeltop=True, labelsize=6)

    ax.set_ylim(*lims)

    y_lines = np.arange(int(np.floor(lims[0])), int(np.ceil(lims[1])) + 1)
    for y in y_lines:
        ax.axhline(y, color='black', alpha=0.2, linewidth=0.5, clip_on=False)

    all_axs = [ax]
    for x in range(1, len(data.columns)):
        other_ax = ax.twinx()

        other_ax.spines['left'].set_position(('data', x))
        other_ax.yaxis.tick_left()
        other_ax.set_ylim(ax.get_ylim())
        other_ax.set_yticklabels([])
        
        all_axs.append(other_ax)

    for ax in all_axs:
        for side in ['right', 'top', 'bottom']:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)

def annotate_with_donors_and_sgRNAs(ax, data, pools, pn_to_name, show_text_labels=True):
    ax.set_autoscale_on(False)
    ax.set_xticklabels([])

    sgRNA_to_target_name = {
        'sgRNA-5': '1',
        'sgRNA-3': '2',
        'sgRNA-2': '3',
        'sgRNA-7': '4',
        'SpCas9 target 1': '1',
        'SpCas9 target 2': '2',
        'SpCas9 target 3': '3',
        'SpCas9 target 4': '4',
    }

    library_name_to_alias = {
        'DDR_library': '1.5k',
        'DDR_sublibrary': '366',
        'AX227': '1.5k',
        'AC001': '366',
    }

    bottom_y = -0.25
        
    label_y = bottom_y + 0.17#1.24
    donor_y = bottom_y + 0.14#1.21
    sgRNA_y = bottom_y + 0.07#1.14
    library_y = bottom_y#1.07

    donor_half_width = 0.3

    arrow_width = donor_half_width * 0.2
    arrow_height = 0.01

    top_strand_y = donor_y + 0.007
    bottom_strand_y = donor_y - 0.007

    at_least_one_donor = False
        
    for x, pn in enumerate([n.rsplit('_', 1)[0] for n in data.columns]):
        pool = pools[pn]
        ti = pool.target_info

        if ti.donor is not None:
            at_least_one_donor = True
            # Draw donor.
            _, _, is_reverse_complement = ti.best_donor_target_alignment
                
            if 'sODN' in pn or ti.donor in ['oBA701', 'oJAH158', 'oBA701-PCR']:
                donor_color = bokeh.palettes.Dark2[8][1]
            elif 'all-SNVs' in pn or ti.donor in ['oJAH159', 'oJAH160']:
                donor_color = bokeh.palettes.Set2[8][0]
            elif 'first-half' in pn:
                donor_color = bokeh.palettes.Set2[8][1]
            elif 'second-half' in pn:
                donor_color = bokeh.palettes.Set2[8][2]
            else:
                donor_color = 'black'

            if 'PAGE' in pn:
                ax.annotate('PAGE',
                            xy=(x, donor_y),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, 3),
                            textcoords='offset points',
                            color=donor_color,
                            size=5,
                            ha='center',
                           )
            
            double_stranded = 'dsODN' in pn or 'PCR' in pn
            
            common_kwargs = dict(
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                color=donor_color,
            )
            
            # Draw reverse orientation strands.
            if is_reverse_complement or double_stranded:
                xs = [x - donor_half_width + arrow_width, x - donor_half_width, x + donor_half_width]
                ys = [bottom_strand_y - arrow_height, bottom_strand_y, bottom_strand_y]
                ax.plot(xs, ys, **common_kwargs)
            
            # Draw forward orientation strands.
            if (not is_reverse_complement) or double_stranded:
                xs = [x - donor_half_width, x + donor_half_width, x + donor_half_width - arrow_width]
                ys = [top_strand_y, top_strand_y, top_strand_y + arrow_height]
                ax.plot(xs, ys, **common_kwargs)
            
            # Draw base pairing ticks for double stranded donors.
            if double_stranded:
                bp_xs = np.linspace(x - 0.9 * donor_half_width, x + 0.9 * donor_half_width, endpoint=True, num=10)
                for bp_x in bp_xs:
                    ax.plot([bp_x, bp_x], [top_strand_y, bottom_strand_y], alpha=0.5, solid_capstyle='butt', **common_kwargs)
                
        if show_text_labels:
            ax.annotate(pn_to_name[pn],
                        xy=(x, label_y),
                        xycoords=('data', 'axes fraction'),
                        ha='left',
                        va='bottom',
                        rotation=45,
                        size=6,
                    )
        
        ax.annotate(sgRNA_to_target_name[ti.sgRNA],
                    xy=(x, sgRNA_y),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color=ti.PAM_color,
                    size=6,
                   )

        library_alias = library_name_to_alias[pool.variable_guide_library.name]
        ax.annotate(library_alias,
                    xy=(x, library_y),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color='black',
                    size=6,
                   )
            
    common_kwargs = dict(
        xycoords='axes fraction',
        xytext=(-10, 0),
        textcoords='offset points',
        ha='right',
        va='center',
        size=6,
    )

    if at_least_one_donor:
        ax.annotate('donor:',
                    xy=(0, donor_y),
                    **common_kwargs,
                   )

    ax.annotate('Cas9 target site:',
                xy=(0, sgRNA_y),
                **common_kwargs,
               )

    ax.annotate('CRISPRi library:',
                xy=(0, library_y),
                **common_kwargs,
               )

def deletion_heatmap(pool, buffer, groups, grids, guide=None, log2_fold_change=False, min_MH=2, max_offset=30):
    def extract_grid(grids, key):
        return grids.loc[key].unstack().reindex(index=np.arange(buffer, -(max_offset + 1), -1), columns=np.arange(-buffer + 1, max_offset)).fillna(0)
    
    nt_grid = 100 * extract_grid(grids, 'all_non_targeting')
    
    if guide is None:
        grid = nt_grid
        cmap = hits.visualize.greens
        #cmap = plt.get_cmap('Greens')
        v_min = 0
        v_max = grid.max().max()
        colorbar_label = f'percentage of outcomes'
        title = f'{pool.short_name}\nall non-targeting'
   
    else:
        guide_grid = 100 * extract_grid(grids, guide)
        grid = guide_grid - nt_grid
        
        if log2_fold_change:
            grid = np.log2(grid / nt_grid)
            v_min = -2
            v_max = 2
        
        else:
            largest = grid.abs().max().max()
            v_min = -largest
            v_max = largest
        
        cmap = plt.get_cmap('bwr')
        
        colorbar_label = f'change in\npercentage of outcomes'
        title = f'{pool.short_name}\n{guide}'
        
    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(12, 12))

    im = ax.imshow(grid,
                   cmap=cmap,
                   vmin=v_min,
                   vmax=v_max,
                  )

    left_offset_to_y = lambda left_offset: buffer - left_offset
    right_offset_to_x = lambda right_offset: right_offset + buffer - 1

    for group in groups:
        if len(group) > min_MH:
            first, *middle, last = group
            ys = [left_offset_to_y(first[0]), left_offset_to_y(last[0])]
            xs = [right_offset_to_x(first[1]), right_offset_to_x(last[1])]
            ax.plot(xs, ys, alpha=0.1 * len(group), linewidth=1 + 0.1 * len(group), color='black')

    intervals = [
        ((pool.target_info.PAM_slice.start, pool.target_info.PAM_slice.stop - 1), 'green'),
        ((pool.target_info.sgRNA_feature.start, pool.target_info.sgRNA_feature.end), 'blue'),
    ]
    
    for (start, end), color in intervals:
        absolute_start = right_offset_to_x(start - pool.target_info.cut_after - 0.5)
        absolute_end = right_offset_to_x(end - pool.target_info.cut_after + 0.5)
        
        if absolute_start < 0 and absolute_end < 0:
            continue

        absolute_start = max(absolute_start, -0.5)
        absolute_end = max(absolute_end, -0.5)
            
        x0, x1 = absolute_start, absolute_end
        y0, y1 = 1.03, 1

        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]
            
        patch = plt.Polygon(path,
                            fill=True,
                            closed=True,
                            alpha=0.3,
                            color=color,
                            linewidth=0,
                            clip_on=False,
                            transform=ax.get_xaxis_transform(),
                           )
        ax.add_patch(patch)
        
    for (start, end), color in intervals:
        absolute_start = left_offset_to_y(start - pool.target_info.cut_after - 0.5)
        absolute_end = left_offset_to_y(end - pool.target_info.cut_after + 0.5)
        
        if absolute_start < 0 and absolute_end < 0:
            continue

        absolute_start = max(absolute_start, -0.5)
        absolute_end = max(absolute_end, -0.5)
            
        x0, x1 = -0.03, 0
        y0, y1 = absolute_start, absolute_end

        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]
            
        patch = plt.Polygon(path,
                            fill=True,
                            closed=True,
                            alpha=0.3,
                            color=color,
                            linewidth=0,
                            clip_on=False,
                            transform=ax.get_yaxis_transform(),
                           )
        ax.add_patch(patch)
        
    for right_offset in np.arange(-buffer, max_offset):
        b = pool.target_info.target_sequence[pool.target_info.cut_after + right_offset]
        ax.annotate(b, 
                    xy=(right_offset_to_x(right_offset), 1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    annotation_clip=True,
                   )
    
    for left_offset in np.arange(-max_offset, buffer + 1):
        b = pool.target_info.target_sequence[pool.target_info.cut_after + left_offset]
        ax.annotate(b,
                    xy=(0, left_offset_to_y(left_offset)),
                    xycoords=('axes fraction', 'data'),
                    xytext=(-5, 0),
                    textcoords='offset points',
                    va='center',
                    ha='right',
                    rotation=90,
                    annotation_clip=True,
                   )

    cut_offsets = [c - pool.target_info.cut_after for c in pool.target_info.cut_afters.values()]

    for cut_offset in cut_offsets:
        ax.axvline(right_offset_to_x(cut_offset + 0.5), linestyle='--', color='black')
        ax.axhline(left_offset_to_y(cut_offset + 0.5), linestyle='--', color='black')

    ax.set_xlim(-0.5, max_offset + 0.5)
    ax.set_ylim(max_offset + 0.5, -0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    for side in ax.spines:
        ax.spines[side].set_visible(False)
        
    ax.set_xlabel('first base remaining after deletion on right', labelpad=30, size=14)
    ax.xaxis.set_label_position('top')
    
    ax.set_ylabel('last base remaining before deletion on left', labelpad=30, size=14)
    
    cax = ax.inset_axes((1.15, 0.25, 0.03, 0.5))
    colorbar = fig.colorbar(im, cax=cax)
    cax.set_title(colorbar_label)

    fig.suptitle(title, y=1.2, size=20)
    
    ax_p = ax.get_position()
    
    marginal_right_ax = fig.add_axes([ax_p.x0, ax_p.y1 + ax_p.height * 0.1, ax_p.width, ax_p.height * 0.2], sharex=ax)
    
    ys = grid.sum(axis=0).values
    xs = np.arange(len(ys))

    if guide is not None:
        marginal_right_ax.plot(xs, guide_grid.sum(axis=0).values, '.-', color='C1')
    marginal_right_ax.plot(xs, nt_grid.sum(axis=0).values, '.-', color='black')
    
    for side in marginal_right_ax.spines:
        marginal_right_ax.spines[side].set_visible(False)
        
    marginal_right_ax.axhline(0, color='black', clip_on=False, alpha=0.5)
        
    marginal_left_ax = fig.add_axes([ax_p.x0 - ax_p.width * 0.3, ax_p.y0, ax_p.width * 0.2, ax_p.height], sharey=ax)
    marginal_left_ax.axhline(left_offset_to_y(0.5), linestyle='--', color='black')
    marginal_left_ax.patch.set_alpha(0)
    marginal_right_ax.patch.set_alpha(0)
    
    xs = grid.sum(axis=1).values
    ys = np.arange(len(xs))

    if guide is not None:
        marginal_left_ax.plot(guide_grid.sum(axis=1).values, ys, '.-', color='C1')
    marginal_left_ax.plot(nt_grid.sum(axis=1).values, ys, '.-', color='black')
    marginal_left_ax.invert_xaxis()
    
    for side in marginal_left_ax.spines:
        marginal_left_ax.spines[side].set_visible(False)
        
    marginal_left_ax.axvline(0, color='black', clip_on=False, alpha=0.5)

    marginal_right_ax.axvline(right_offset_to_x(0.5), linestyle='--', color='black')
    
    marginal_left_ax.xaxis.tick_top()
    
    #t = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    #ax.plot([0, 1], [left_offset_to_y(0.5), left_offset_to_y(0.5)], linestyle='--', color='black', transform=t, clip_on=False)
    
    #t = matplotlib.transforms.blended_transform_factory(ax.transData, fig.transFigure)
    #ax.plot([right_offset_to_x(0.5), right_offset_to_x(0.5)], [0, 1], linestyle='--', color='black', transform=t, clip_on=False)
    
    return fig

def sorted_by_significance(pool, outcomes,
                           top_n_guides=None,
                           show_non_top=False,
                           n_up=20,
                           n_down=20,
                           y_lims=None,
                           quantity_to_plot='log2_fold_change',
                          ):

    df, nt_frac, gene_df = repair_seq.visualize.gene_significance.get_outcome_statistics(pool, outcomes)
                          
    if quantity_to_plot == 'log2_fold_change':
        y_key = 'log2_fold_change'
        bottom_key = 'log2_fold_change_interval_bottom'
        top_key = 'log2_fold_change_interval_top'

        y_label = 'log2 fold-change from non-targeting'

        hline_y = 0

    else:
        if quantity_to_plot == 'percentage':
            nt_percentage = nt_frac * 100
            df['percentage'] = df['frequency'] * 100
            df['percentage_interval_bottom'] = df['interval_bottom'] * 100
            df['percentage_interval_top'] = df['interval_top'] * 100

            y_key = 'percentage'
            bottom_key = 'percentage_interval_bottom'
            top_key = 'percentage_interval_top'
            y_label = 'percentage of outcomes'

            hline_y = nt_percentage

        else:
            y_key = 'frequency'
            bottom_key = 'interval_bottom'
            top_key = 'interval_top'

            hline_y = nt_frac

    targeting_ps = gene_df['p'].drop(index='negative_control')

    down_genes = targeting_ps.query('down < up')['down'].sort_values().index
    up_genes = targeting_ps.query('up < down')['up'].sort_values(ascending=False).index

    gene_order = np.concatenate([down_genes, ['negative_control'], up_genes])

    guide_order = []
    xs = []

    gap = 20
    x = 0

    gene_to_color = {}

    guides_in_top_n = set(pool.variable_guide_library.non_targeting_guides)

    for gene_i, gene in enumerate(gene_order):
        guides = pool.variable_guide_library.gene_guides(gene)
        if gene == 'negative_control' or top_n_guides is None:
            top_guides = guides
        else:
            ordered_guides = df.reindex(guides)[y_key].sort_values().index
            if gene in up_genes:
                top_guides = ordered_guides[-top_n_guides:]
            else:
                top_guides = ordered_guides[:top_n_guides]

        guides_in_top_n.update(top_guides)

        if show_non_top:
            guides_to_plot = guides
        else:
            guides_to_plot = [g for g in guides if g in top_guides]
                
        for guide in guides_to_plot:
            if guide in df.index:
                guide_order.append(guide)
                xs.append(x)
                x += 1

        if gene_i < n_down or gene_i > len(gene_order) - n_up - 1 - 1:
            x += gap

        if gene == 'negative_control':
            color = 'grey'
        else:
            color = f'C{gene_i % 10}'

        gene_to_color[gene] = color


    df_ordered = df.reindex(guide_order)
    df_ordered['x'] = xs

    fig, ax = plt.subplots(figsize=(15, 6))

    colors = df_ordered['gene'].map(gene_to_color)

    genes_to_label = np.concatenate([gene_order[:n_down], ['negative_control'], gene_order[-n_up:]])

    guides_to_label = set()

    for gene in genes_to_label:
        gene_rows = df_ordered.query('gene == @gene')

        guides_to_label.update(gene_rows.index)

        if gene in down_genes or gene == 'negative_control':
            y = gene_rows[bottom_key].min()
            va = 'top'
            y_offset = -5
        else:
            y = gene_rows[top_key].max()
            va = 'bottom'
            y_offset = 5

        x = gene_rows['x'].mean()

        ax.annotate(gene,
                    xy=(x, y),
                    xytext=(0, y_offset),
                    textcoords='offset points',
                    size=8,
                    color=gene_to_color[gene],
                    va=va,
                    ha='center',
                   )

    guides_to_label = guides_to_label & guides_in_top_n

    point_colors = matplotlib.colors.to_rgba_array(colors)
    point_alphas = [0.95 if guide in guides_to_label else 0.25 for guide in df_ordered.index]
    point_colors[:, 3] = point_alphas

    line_colors = matplotlib.colors.to_rgba_array(colors)
    line_alphas = [0.3 if guide in guides_to_label else 0.15 for guide in df_ordered.index]
    line_colors[:, 3] = line_alphas

    ax.scatter(x='x', y=y_key, c=point_colors, data=df_ordered, s=15, linewidths=(0,))
    ax.set_xlim(-10, xs[-1] +  10)

    ax.axhline(hline_y, color='black', alpha=0.5)

    for (_, row), line_color in zip(df_ordered.iterrows(), line_colors):
        x = row['x']
        ax.plot([x, x], [row[bottom_key], row[top_key]], color=line_color)
        
    for side in ax.spines:
        ax.spines[side].set_visible(False)

    if quantity_to_plot == 'log2_fold_change':
        if y_lims is None:
            y_min = int(np.floor(min(df_ordered[bottom_key])))
            y_max = int(np.ceil(max(df_ordered[top_key])))
        else:
            y_min, y_max = y_lims
        ax.set_yticks(np.arange(y_min, y_max + 1))

        for y in np.arange(y_min, y_max + 1):
            ax.axhline(y, color='black', alpha=0.1, clip_on=False)

    else:
        if y_lims is None:
            y_lims = (0, None)

        y_min, y_max = y_lims
        ax.grid(axis='y', color='black', alpha=0.1, clip_on=False)

    ax.set_ylabel(y_label)
            
    ax.set_ylim(y_min, y_max)

    ax.set_xticks([])

    ax.annotate('most significant genes down',
                xy=(0, 0),
                xycoords='axes fraction',
                xytext=(0, -10),
                textcoords='offset points',
                ha='left',
                va='top',
    )

    ax.annotate('most significant genes up',
                xy=(1, 0),
                xycoords='axes fraction',
                xytext=(0, -10),
                textcoords='offset points',
                ha='right',
                va='top',
    )

    ax.annotate('guides grouped by gene,\nsorted by gene-level significance',
                xy=(0.5, 0),
                xycoords='axes fraction',
                xytext=(0, -10),
                textcoords='offset points',
                ha='center',
                va='top',
    )
    
    return fig

def parts_lists(pool, outcomes, gene_sets, interesting_genes, label_all_above=None, label_top_n=None, x_lims=None):
    guides_df, nt_fraction, genes_df = repair_seq.visualize.gene_significance.get_outcome_statistics(pool, outcomes)

    gene_to_color, guide_to_color = make_guide_to_color(pool.variable_guide_library.guide_to_gene, interesting_genes, gene_sets)

    figs = {}

    #figs['alphabetical'], *rest = repair_seq.visualize.gene_significance.gene_significance_simple(pool, outcomes, quantity_to_plot='log2_fold_change', max_num_to_label=25, figsize=(15, 6))

    #figs['sorted_fc'] = sorted_by_significance(pool, outcomes, top_n_guides=2, n_up=15, n_down=15, quantity_to_plot='log2_fold_change')
    #
    #figs['sorted_percentage'] = sorted_by_significance(pool, outcomes, top_n_guides=2, n_up=15, n_down=15, quantity_to_plot='percentage')
    #
    #figs['volcano_guides'] = volcano_guides(guides_df, guide_to_color, gene_to_color, gene_sets)
    
    figs['volcano_genes'] = repair_seq.visualize.volcano.genes(genes_df, gene_to_color, gene_sets, label_all_above=label_all_above, x_lims=x_lims, label_top_n=label_top_n)
    #figs['volcano_genes'].axes[0].set_title(pool.short_name)

    return figs
