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

    clusterd_guide_order, clustered_outcome_order, correlations = ddr.visualize.heatmap.cluster(pool,
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

    im = correlation_ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)

    correlation_ax.set_xticks([])
    correlation_ax.set_yticks([])

    for spine in correlation_ax.spines.values():
        spine.set_visible(False)

    transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + correlation_ax.transData

    im.set_transform(transform)

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

    ax_p = correlation_ax.get_position()

    # Draw correlation colorbar.
    cbar_ax = fig.add_axes((ax_p.x1 - ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.4, ax_p.width * 0.01, ax_p.height * 0.3))
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar_ax.annotate('correlation\nbetween\noutcome\nredistribution\nprofiles',
                     xy=(1, 0.5),
                     xycoords='axes fraction',
                     xytext=(60, 0),
                     textcoords='offset points',
                     ha='center',
                     va='center',
                    )

    width = ax_p.width
    height = width * len(clustered_outcome_order) / len(guide_order)
    # Make gap equal to height for 5 outcome rows.
    gap = height / len(clustered_outcome_order) * 5
    fc_ax = fig.add_axes((ax_p.x0, ax_p.y0 - height - gap, width, height))

    fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[clustered_outcome_order, guide_order]

    heatmap_kwargs = dict(cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2)
    fc_ax.imshow(fcs, **heatmap_kwargs)

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

    return fig, guide_order, clustered_outcome_order


def draw_ssODN_configurations(pools, pool_names):
    common_ti = pools[pool_names[0]].target_info

    x_min = -120
    x_max = 120

    rect_height = 0.25

    fig, ax = plt.subplots(figsize=(30, 4))

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
    length_to_resect = 110

    ys = {
        'target_+': 0.15,
        'target_-': -0.15,
        'donor_+': 0.75,
        'donor_-': -0.75,
    }

    colors = bokeh.palettes.Set2[8]

    before_cut = common_ti.target_sequence[:common_ti.cut_after + 1][-120:]
    after_cut = common_ti.target_sequence[common_ti.cut_after + 1:][:120]

    for b, x in zip(before_cut[::-1], np.arange(len(before_cut))):
        final_x = -x - offset_at_cut - 0.5

        ax.annotate(b,
                    (final_x, ys['target_+']),
                    **kwargs,
                   )

        if x < length_to_resect:
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

        if x < length_to_resect:
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

    for pool_name, color in zip(pool_names, colors):
        ti = pools[pool_name].target_info
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

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        for b, x in zip(donor_after_cut, np.arange(len(donor_after_cut))):
            final_x = x + offset_at_cut + 0.5

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

    alpha = 0.1
    draw_rect(offset_at_cut + length_to_resect, x_max, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(0, x_max, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

    draw_rect(0, x_min, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(-offset_at_cut - length_to_resect, x_min, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

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