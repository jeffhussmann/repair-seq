import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import bokeh.palettes

import hits.visualize
import hits.utilities
import ddr.visualize.heatmap
import ddr.visualize.outcome_diagrams

def plot_correlations(pool, guide, num_outcomes, label=True, extra_genes=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    gene = pool.variable_guide_library.guide_to_gene[guide]
    gene_guides = pool.variable_guide_library.gene_guides(gene)

    outcomes = pool.most_frequent_outcomes('none')[:num_outcomes]
    log2_fcs = pool.log2_fold_changes('perfect', 'none')['none'].loc[outcomes].drop(columns=['all_non_targeting'])
    correlations = log2_fcs.corr().loc[guide].sort_values(ascending=False)

    df = pd.DataFrame({'correlation': correlations})
    df['gene'] = pool.variable_guide_library.guides_df['gene'].loc[df.index]
    df['x'] = np.arange(len(df))
    df['best_promoter'] = pool.variable_guide_library.guides_df['best_promoter']

    ax.set_xlim(-int(0.02 * len(df)), int(1.02 * len(df)))
    ax.set_ylim(-1.01, 1.01)

    ax.scatter('x', 'correlation', c='C0', s=20, data=df.query('gene == @gene'), clip_on=False, label=f'{gene} guides')
    ax.scatter('x', 'correlation', c='C1', s=15, data=df.query('gene == "negative_control"'), clip_on=False, label='non-targeting guides')
    ax.scatter('x', 'correlation', c='grey', s=1, data=df.query('gene != "negative_control" and gene != @gene'), clip_on=False, label='other guides')
    
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
        
#         for guide, row in df.query('gene == @gene').iterrows():
#             ax.annotate(guide,
#                         xy=(row['x'], row['correlation']),
#                         xytext=(5, 0),
#                         textcoords='offset points',
#                         ha='left',
#                         va='center',
#                         color='C1',
#                         size=6,
#                        )
        
        if extra_genes is not None:
            other_guides = df.loc[df['gene'].isin(list(extra_genes)) & df['best_promoter']]
            to_label = other_guides
        else:
            other_guides = df.query('gene != "negative_control" and gene != @gene')
            to_label = pd.concat([other_guides.iloc[:5], other_guides.iloc[-5:]])
         
        to_label = to_label.sort_values('correlation', ascending=False)
        to_label.index.name = 'short_name'
            
        hits.visualize.label_scatter_plot(ax, 'x', 'correlation', 'short_name', to_label, avoid=True, initial_distance=10, vector='sideways')
#         for guide, row in to_label.iterrows():
#             ax.annotate(guide,
#                         xy=(row['x'], row['correlation']),
#                         xytext=(5, 0),
#                         textcoords='offset points',
#                         ha='left',
#                         va='center',
#                         color='grey',
#                         size=8,
#                        )
            
    for side in ['top', 'bottom', 'right']:
        ax.spines[side].set_visible(False)
    
    return fig, df

def correlation_heatmap(pool, num_outcomes=40, guides=100, upside_down=False, layout_kwargs=None, fixed_guide='none'):
    if layout_kwargs is None:
        layout_kwargs = {}

    cmap = plt.get_cmap('PuOr_r')
    cmap.set_bad('white')

    clustered_guide_order, clustered_outcome_order, correlations = ddr.visualize.heatmap.cluster(pool,
                                                                             num_outcomes,
                                                                             guides,
                                                                             method='average',
                                                                             fixed_guide=fixed_guide,
                                                                            )
    if isinstance(guides, int):
        guide_order = clustered_guide_order
    else:
        guide_order = guides

    if upside_down:
        mask = np.triu(np.ones((len(correlations), len(correlations))), k=1)
    else:
        mask = np.tri(len(correlations), k=-1)

    masked = np.ma.array(correlations, mask=mask)

    size = 16 * len(guide_order) / 100
    fig, correlation_ax = plt.subplots(figsize=(size, size))

    correlation_im = correlation_ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)

    correlation_ax.set_xticks([])
    correlation_ax.set_yticks([])

    for spine in correlation_ax.spines.values():
        spine.set_visible(False)

    transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + correlation_ax.transData

    correlation_im.set_transform(transform)

    diag_length = np.sqrt(2 * len(correlations)**2)

    if upside_down:
        correlation_ax.set_ylim(diag_length / 2 + 1, 0)
    else:
        correlation_ax.set_ylim(0, -diag_length / 2 - 1)

    correlation_ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    for i, guide in enumerate(guide_order):
        x = 2**0.5 * i
        correlation_ax.annotate(guide,
                    xy=(x, 0),
                    xytext=(0, 3 if upside_down else -3),
                    textcoords='offset points',
                    rotation=90,
                    ha='center',
                    va='bottom' if upside_down else 'top',
                    size=6,
                )

    if upside_down:
        title_kwargs = dict(
            xy=(0.5, 0),
            xytext=(0, -15),
            va='top',
        )
    else:
        title_kwargs = dict(
            xy=(0.5, 1),
            xytext=(0, 15),
            va='bottom',
        )

    correlation_ax.annotate(pool.group,
            xycoords='axes fraction',
            textcoords='offset points',
            ha='center',
            **title_kwargs,
    )

    ax_p = correlation_ax.get_position()

    # Draw correlation colorbar.
    cbar_ax = fig.add_axes((ax_p.x1 - ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.4, ax_p.width * 0.02, ax_p.height * 0.4))
    cbar = fig.colorbar(correlation_im, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar_ax.annotate('correlation\nbetween\noutcome\nredistribution\nprofiles',
                     xy=(1, 0.5),
                     xycoords='axes fraction',
                     xytext=(60, 0),
                     textcoords='offset points',
                     ha='center',
                     va='center',
                    )

    cbar.outline.set_alpha(0.1)

    width = ax_p.width
    height = width * len(clustered_outcome_order) / len(guide_order)
    # Make gap equal to height for 5 outcome rows.
    gap = height / len(clustered_outcome_order) * 5

    if upside_down:
        y0 = ax_p.y1 + gap
    else:
        y0 = ax_p.y0 - height - gap

    fc_ax = fig.add_axes((ax_p.x0, y0, width, height))

    fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[clustered_outcome_order, guide_order]

    heatmap_kwargs = dict(cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2)
    heatmap_im = fc_ax.imshow(fcs, **heatmap_kwargs)

    fc_ax.axis('off')

    fc_ax_p = fc_ax.get_position()

    diagram_width = fc_ax_p.width * 0.3
    diagram_gap = diagram_width * 0.02

    diagram_ax = fig.add_axes((fc_ax_p.x0 - diagram_width - diagram_gap, fc_ax_p.y0, diagram_width, fc_ax_p.height), sharey=fc_ax)
    _ = ddr.visualize.outcome_diagrams.plot(clustered_outcome_order[::-1],
                              pool.target_info,
                              ax=diagram_ax,
                              **layout_kwargs,
                             )

    diagram_ax.set_title('distance from cut site (nts)', size=10, pad=15)

    frequency_width = fc_ax_p.width * 0.15
    frequency_gap = frequency_width * 0.1

    frequency_ax = fig.add_axes((fc_ax_p.x1 + frequency_gap, fc_ax_p.y0, frequency_width, fc_ax_p.height), sharey=fc_ax)

    frequencies = pool.non_targeting_fractions('perfect', 'none').loc[clustered_outcome_order]

    frequency_ax.plot(np.log10(frequencies), np.arange(len(frequencies)), '.', markeredgewidth=0, markersize=10, alpha=0.9)

    x_lims = np.log10(np.array([1e-3, 2e-1]))

    for exponent in [3, 2, 1]:
        xs = np.log10(np.arange(1, 10) * 10**-exponent)        
        for x in xs:
            if x_lims[0] <= x <= x_lims[1]:
                frequency_ax.axvline(x, color='black', alpha=0.07, clip_on=False)

    x_ticks = [x for x in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1] if x_lims[0] <= np.log10(x) <= x_lims[1]]

    frequency_ax.set_xticks(np.log10(x_ticks))
    frequency_ax.set_xticklabels([f'{100 * x:g}' for x in x_ticks])

    for side in ['left', 'right', 'bottom']:
        frequency_ax.spines[side].set_visible(False)

    frequency_ax.xaxis.tick_top()
    frequency_ax.set_xlim(*x_lims)

    frequency_ax.set_title('frequency of outcome\nin cells containing\nnon-targeting guides\n(percentage)', size=10, pad=15)

    ddr.visualize.heatmap.add_fold_change_colorbar(fig, heatmap_im, -0.05, 0.4, 0.15, 0.02)

    return fig, guide_order, clustered_outcome_order

def draw_ssODN_configurations(pools=None, tis=None):
    if pools is None:
        common_ti = tis[0]
    else:
        common_ti = pools[0].target_info
        tis = [pool.target_info for pool in pools]

    rect_height = 0.25

    fig, ax = plt.subplots(figsize=(25, 4))

    def draw_rect(x0, x1, y0, y1, alpha, color='black', fill=True):
        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]

        patch = plt.Polygon(path,
                            fill=fill,
                            closed=True,
                            alpha=alpha,
                            color=color,
                            linewidth=0 if fill else 1.5,
                            clip_on=False,
                           )
        ax.add_patch(patch)

    def mark_5_and_3(y, x_start, x_end, color):
        if y > 0:
            left_string = '5\''
            right_string = '3\''
        else:
            right_string = '5\''
            left_string = '3\''

        ax.annotate(left_string, (x_start, y),
                    xytext=(-5, 0),
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    color=color,
                   )

        ax.annotate(right_string, (x_end, y),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    color=color,
                   )

    kwargs = dict(ha='center', va='center', fontfamily='monospace',)

    offset_at_cut = 0

    ys = {
        'target_+': 0.15,
        'target_-': -0.15,
        'donor_+': 0.75,
        'donor_-': -0.75,
    }

    colors = bokeh.palettes.Set2[8]

    donor_x_min = -1
    donor_x_max = 1


    for ti, color in zip(tis, colors):
        _, offset, is_reverse_complement = ti.best_donor_target_alignment

        # For ss donors, ti.donor_sequence is the actual stranded sequence
        # supplied. There is no need to complement anything, since it is drawn
        # on the bottom with the - strand if it aligned to the reverse complement
        # of the target strandedness.

        if is_reverse_complement:
            aligned_donor_seq = ti.donor_sequence[::-1]
            donor_y = ys['donor_-']
        else:
            aligned_donor_seq = ti.donor_sequence
            donor_y = ys['donor_+']

        donor_cut_after = ti.cut_after - offset

        donor_before_cut = aligned_donor_seq[:donor_cut_after + 1]
        donor_after_cut = aligned_donor_seq[donor_cut_after + 1:]

        for b, x in zip(donor_before_cut[::-1], np.arange(len(donor_before_cut))):
            final_x = -x - offset_at_cut - 0.5

            donor_x_min = min(donor_x_min, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        for b, x in zip(donor_after_cut, np.arange(len(donor_after_cut))):
            final_x = x + offset_at_cut + 0.5

            donor_x_max = max(donor_x_max, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        if is_reverse_complement:
            y = ys['donor_-']
        else:
            y = ys['donor_+']

        draw_rect(-len(donor_before_cut),
                  len(donor_after_cut),
                  donor_y + rect_height / 2,
                  donor_y - rect_height / 2,
                  alpha=0.5,
                  fill=True,
                  color=color,
                 )

        mark_5_and_3(donor_y, -len(donor_before_cut), len(donor_after_cut), color)

        for name, info in common_ti.donor_SNVs['target'].items():
            x = info['position'] - ti.cut_after

            if x >= 0:
                x = -0.5 + offset_at_cut + x
            else:
                x = -0.5 - offset_at_cut + x

            draw_rect(x - 0.5, x + 0.5, donor_y - rect_height / 2, donor_y + rect_height / 2, 0.2)

    # Draw resected target.

    resect_before = int(np.abs(np.floor(donor_x_min))) + 1
    resect_after = int(np.abs(np.ceil(donor_x_max))) + 1

    x_min = -resect_before - 5
    x_max = resect_after + 5

    before_cut = common_ti.target_sequence[:common_ti.cut_after + 1][x_min:]
    after_cut = common_ti.target_sequence[common_ti.cut_after + 1:][:x_max]

    for b, x in zip(before_cut[::-1], np.arange(len(before_cut))):
        final_x = -x - offset_at_cut - 0.5

        ax.annotate(b,
                    (final_x, ys['target_+']),
                    **kwargs,
                   )

        if x < resect_before:
            alpha = 0.3
        else:
            alpha = 1

        ax.annotate(hits.utilities.complement(b),
                    (final_x, ys['target_-']),
                    alpha=alpha,
                    **kwargs,
                   )

    for b, x in zip(after_cut, np.arange(len(after_cut))):
        final_x = x + offset_at_cut + 0.5

        if x < resect_after:
            alpha = 0.3
        else:
            alpha = 1

        ax.annotate(b,
                    (final_x, ys['target_+']),
                    alpha=alpha,
                    **kwargs,
                   )

        ax.annotate(hits.utilities.complement(b),
                    (final_x, ys['target_-']),
                    **kwargs,
                   )

    alpha = 0.1
    draw_rect(offset_at_cut + resect_after, x_max, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(0, x_max, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

    draw_rect(0, x_min, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(-offset_at_cut - resect_before, x_min, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-2, 2)

    ax.plot([0, 0], [ys['target_-'] - rect_height, ys['target_+'] + rect_height], color='black', linestyle='--', alpha=0.5)

    mark_5_and_3(ys['target_+'], x_min, x_max, 'black')
    mark_5_and_3(ys['target_-'], x_min, x_max, 'black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    for name, info in common_ti.donor_SNVs['target'].items():
        x = info['position'] - ti.cut_after

        if x >= 0:
            x = -0.5 + offset_at_cut + x
        else:
            x = -0.5 - offset_at_cut + x

        for y in [ys['target_+'], ys['target_-']]:
            draw_rect(x - 0.5, x + 0.5, y - rect_height / 2, y + rect_height / 2, 0.2)

    for name, PAM_slice in common_ti.PAM_slices.items():
        sgRNA = common_ti.sgRNA_features[name]
        if sgRNA.strand == '+':
            y_start = ys['target_+'] - rect_height / 4
            y_end = y_start - rect_height / 4
        else:
            y_start = ys['target_-'] + rect_height / 4
            y_end = y_start + rect_height / 4

        x_start = PAM_slice.start - common_ti.cut_after - 1 # offset empirically determined, confused by it
        x_end = PAM_slice.stop - common_ti.cut_after - 1

        ax.plot([x_start, x_start, x_end, x_end], [y_start, y_end, y_end, y_start])
            
    return fig

def conversion_tracts(pool, genes, fc_ylims=None):
    f = draw_ssODN_configurations([pool])

    ax = f.axes[0]

    xs = pool.SNV_name_to_position - pool.target_info.cut_after

    # overall incorporation frequency plot

    ax_p = ax.get_position()
    ax = f.add_axes([ax_p.x0, ax_p.y0 - 0.5 * ax_p.height, ax_p.width, 0.5 * ax_p.height], sharex=ax)

    ys = pool.conversion_fractions['all_non_targeting']
    ax.plot(xs, ys, 'o-', color='black', linewidth=2)

    ax.axvline(0, linestyle='--', color='black', alpha=0.5)
    
    ax.set_ylim(0, max(ys) * 1.1)

    ax.set_ylabel('overall\nconversion\nfrequency')

    # log2 fold changes plot

    ax_p = ax.get_position()
    ax = f.add_axes([ax_p.x0, ax_p.y0 - 2.2 * ax_p.height, ax_p.width, 2 * ax_p.height], sharex=ax)

    guide_sets = [
        ('negative_control', 'non-targeting', dict(color='black', alpha=0.2)),
    ]

    for gene_i, gene in enumerate(genes):
        guide_sets.append((gene, None, dict(color=ddr.visualize.heatmap.good_colors[gene_i], alpha=0.8, linewidth=1.5)))

    max_y = 1
    min_y = -2
    
    for gene, label, kwargs in guide_sets:
        for i, guide in enumerate(pool.variable_guide_library.gene_guides(gene)):
            ys = pool.conversion_log2_fold_changes[guide]

            max_y = np.ceil(max(max_y, max(ys)))
            min_y = np.floor(min(min_y, min(ys)))

            label_to_use = None

            if i == 0:
                if label is None:
                    label_to_use = gene
                else:
                    label_to_use = label
            else:
                label_to_use = ''

            ax.plot(xs, ys, '.-', label=label_to_use, **kwargs)

    ax.legend()

    if fc_ylims is None:
        fc_ylims = (max(-6, min_y), min(5, max_y))
    
    ax.set_ylim(*fc_ylims)
    #ax.set_ylim(-4, 1)

    ax.axhline(0, color='black', alpha=0.2)
    ax.axvline(0, linestyle='--', color='black', alpha=0.5)

    ax.set_ylabel('log2 fold-change in conversion from non-targeting')

    return f

def fraction_removed(pool, genes, fraction_y_lims=(5e-4, 1), fold_change_y_lims=(-4, 2)):
    fig, (fraction_ax, fc_ax) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw=dict(hspace=0.05))

    guide_sets = [
        ('negative_control', 'individual non-targeting guide', dict(color='black', alpha=0.1)),
    ]

    for gene_i, gene in enumerate(genes):
        guide_sets.append((gene, f'{gene} guides', dict(color=ddr.visualize.heatmap.good_colors[gene_i], alpha=0.8, linewidth=1.5)))

    kwargs = dict(color='black', alpha=0.9, linewidth=2, label='all non-targeting guides')
    fraction_ax.plot(pool.fraction_removed['all_non_targeting'], '-', **kwargs)
    fc_ax.plot(pool.fraction_removed_log2_fold_changes['all_non_targeting'], '-', **kwargs)    

    for gene, label, kwargs in guide_sets:
        guides = pool.variable_guide_library.gene_guides(gene, only_best_promoter=True)

        for i, guide in enumerate(guides):
            if i == 0:
                label_to_use = label
            else:
                label_to_use = ''
            
            ys = pool.fraction_removed[guide].replace(to_replace=0, value=np.nan)
            fraction_ax.plot(ys, '-', label=label_to_use, **kwargs)
            fc_ax.plot(pool.fraction_removed_log2_fold_changes[guide], '-', label=label_to_use, **kwargs)

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

def make_color_column(guide_to_gene, genes, full_gene_list=None, default_color='silver'):
    if full_gene_list is None:
        full_gene_list = genes

    guide_to_color = pd.Series(default_color, index=guide_to_gene.index)

    for i, gene in enumerate(full_gene_list):
        if gene in genes:
            if i + 1 < 10:
                color = f'C{i + 1}'
            else:
                color = bokeh.palettes.Category20b[20][::4][(i + 1) % 10]

            guide_to_color[guide_to_gene == gene] = color

    guide_to_color[guide_to_gene == 'negative_control'] = 'C0'
        
    guide_to_color = guide_to_color.apply(matplotlib.colors.to_hex)
        
    return guide_to_color

def scatter_and_pc(pool,
                   results,
                   x_column,
                   y_column,
                   genes_to_label,
                   full_gene_list=None,
                   lims=(-4, 2),
                   avoid_overlapping_labels=False,
                   pn_to_name=None,
                  ):
    data = results['full_df'].xs('log2_fold_change', axis=1, level=1).copy()
    best_promoter = pool.variable_guide_library.guides_df['best_promoter']
    guide_to_gene = results['full_df']['gene']
        
    guides_to_label = guide_to_gene[guide_to_gene.isin(genes_to_label) & best_promoter].index
    #guides_to_label = guide_to_gene[guide_to_gene.isin(genes_to_label)].index

    guide_to_color = make_color_column(guide_to_gene, genes_to_label, full_gene_list=full_gene_list)

    fig, pc_ax = plt.subplots(figsize=(0.75 * len(data.columns), 6))
    
    parallel_coordinates(data, pc_ax,
                         guides_to_label,
                         guide_to_gene,
                         guide_to_color,
                         lims=lims,
                         text_labels=['right'],
                         pn_to_name=pn_to_name,
                        )

    pc_ax_p = pc_ax.get_position()
    fig_width, fig_height = fig.get_size_inches()
    scatter_height = pc_ax_p.height
    scatter_width = fig_height / fig_width * scatter_height
    scatter_ax = fig.add_axes((pc_ax_p.x1 + pc_ax_p.width * 0.5, pc_ax_p.y0, scatter_width, scatter_height), sharey=pc_ax) 

    scatter(data, scatter_ax,
            x_column,
            y_column,
            guides_to_label,
            guide_to_color,
            avoid_overlapping_labels=avoid_overlapping_labels,
            lims=lims,
            pn_to_name=pn_to_name,
           )

    fig_transform = pc_ax.figure.transFigure
    inverse_figure = fig_transform.inverted()
    pc_transform = pc_ax.get_xaxis_transform() + inverse_figure
    scatter_transform = scatter_ax.transAxes + inverse_figure

    def draw_path(points, transforms):
        xs, ys = np.array([t.transform_point(p) for t, p in zip(transforms, points)]).T
        pc_ax.plot(xs, ys, transform=fig_transform, clip_on=False, color='black')

    def draw_bracket(x):
        draw_path([(x - 0.1, -bracket_offset + bracket_height),
                   (x - 0.1, -bracket_offset),
                   (x + 0.1, -bracket_offset),
                   (x + 0.1, -bracket_offset + bracket_height),
                  ],
                  [pc_transform, pc_transform, pc_transform, pc_transform],
                 )

    bracket_offset = 0.03
    bracket_height = 0.025
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
            guides_to_label,
            guide_to_color, 
            lims=(-4, 2),
            avoid_overlapping_labels=True,
            pn_to_name=None,
           ):

    data = data.copy()

    data['color'] = guide_to_color

    common_kwargs = dict(x=x_column, y=y_column, c='color', linewidths=(0,))
    
    ax.scatter(data=data.loc[data.index.difference(guides_to_label)], s=15, alpha=0.5, **common_kwargs)
    ax.scatter(data=data.loc[guides_to_label], s=25, alpha=0.95, zorder=10, **common_kwargs)

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

    ax.set_xlabel(x_label + '\nlog2 fold change', size=12)
    ax.set_ylabel(y_label + '\nlog2 fold change', size=12)
    
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
                                      data=data.loc[guides_to_label],
                                      text_kwargs={'size': 10},
                                      initial_distance=20,
                                      color='color',
                                      avoid=avoid_overlapping_labels,
                                      avoid_existing=avoid_overlapping_labels,
                                     )
        

def parallel_coordinates(data, ax,
                         guides_to_label,
                         guide_to_gene,
                         guide_to_color,
                         lims=(-4, 2),
                         text_labels=None,
                         pn_to_name=None,
                        ):
    guide_to_kwargs = {}

    if text_labels is None:
        text_labels = []

    if pn_to_name is None:
        pn_to_name = {n.rsplit('_', 1)[0]: n.rsplit('_', 1)[0] for n in data.columns}

    genes_to_label = guide_to_gene[guides_to_label].unique()
    for gene_i, gene in enumerate(genes_to_label, 1):
        guides = guides_to_label[guide_to_gene[guides_to_label] == gene]
        for guide_i, guide in enumerate(guides):
            guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
                                          marker='.',
                                          markersize=8,
                                          alpha=0.8,
                                          linewidth=2.5,
                                          label=f'{gene} guides' if guide_i == 0 else '',
                                          clip_on=False,
                                         )

    for guide_i, guide in enumerate(guide_to_gene[guide_to_gene == 'negative_control'].index):     
        guide_to_kwargs[guide] = dict(color=guide_to_color[guide],
                                      alpha=0.4,
                                      linewidth=1,
                                      label='individual\nnon-targeting\nguides' if guide_i == 0 else '',
                                     )

    for guide, row in data.iterrows():
        kwargs = guide_to_kwargs.get(guide)
        if kwargs is None:
            kwargs = dict(color='black', alpha=0.04)
            continue
        
        ax.plot(row.values, **kwargs)

        if guide_to_gene[guide] != 'negative_control':
            if 'right' in text_labels:
                ax.annotate(guide,
                            xy=(1, row.values[-1]),
                            xycoords=('axes fraction', 'data'),
                            xytext=(5, 0),
                            textcoords='offset points',
                            color=kwargs['color'],
                            ha='left',
                            va='center',
                            size=6,
                )

        
    ax.legend(bbox_to_anchor=(-0.2, 1), loc='upper right')
        
    ax.set_ylabel('log2 fold-change from all non-targeting', size=12)

    ax.set_xticks(np.arange(len(data.columns)))
    labels = []
    for n in data.columns:
        pn, outcome = n.rsplit('_', 1)
        label = f'{pn_to_name[pn]} {outcome}'
        labels.append(label)
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.tick_params(labelbottom=False, labeltop=True)

    ax.set_ylim(*lims)

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

def annotate_with_donors_and_sgRNAs(ax, data, pools, pn_to_name):
    ax.set_autoscale_on(False)
    ax.set_xticklabels([])

    sgRNA_colors = bokeh.palettes.Colorblind8[3::2]

    sgRNAs = [
        'sgRNA-5',
        'sgRNA-3',
    ]

    sgRNA_to_color = dict(zip(sgRNAs, sgRNA_colors))
        
    label_y = 1.20
    donor_y = 1.14
    sgRNA_y = 1.07

    donor_half_width = 0.3

    arrow_width = donor_half_width * 0.2
    arrow_height = 0.01

    top_strand_y = donor_y + 0.005
    bottom_strand_y = donor_y - 0.005
        
    for x, pn in enumerate([n.rsplit('_', 1)[0] for n in data.columns]):
        ti = pools[pn].target_info

        _, _, is_reverse_complement = ti.best_donor_target_alignment
            
        if 'sODN' in pn:
            donor_color = 'black'
        else:
            donor_color = 'red'
        
        double_stranded = 'dsODN' in pn
        
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
                
        ax.annotate(pn_to_name[pn],
                    xy=(x, label_y),
                    xycoords=('data', 'axes fraction'),
                    ha='left',
                    va='bottom',
                    rotation=45,
                )
        
        ax.annotate(pools[pn].target_info.sgRNA,
                    xy=(x, sgRNA_y),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='center',
                    color=sgRNA_to_color[pools[pn].target_info.sgRNA],
                )
            
    common_kwargs = dict(
        xycoords='axes fraction',
        xytext=(-30, 0),
        textcoords='offset points',
        ha='right',
        va='center',
    )

    ax.annotate('donor:',
                xy=(0, donor_y),
                **common_kwargs,
            )

    ax.annotate('cutting sgRNA:',
                xy=(0, sgRNA_y),
                **common_kwargs,
            )