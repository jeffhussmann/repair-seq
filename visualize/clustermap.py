import copy
from collections import defaultdict, Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import seaborn as sns

import hits.visualize
import ddr.cluster
import ddr.visualize.outcome_diagrams

def clustermap(pool,
               outcomes=40,
               guides=100,
               diagram_ax_rectangle=None,
               fig=None,
               upside_down=False,
               diagram_kwargs=None,
               fixed_guide='none',
               cluster='both',
               gene_to_color=None,
               min_UMIs=None,
               draw_colorbars=True,
               num_guide_clusters=10,
               num_outcome_clusters=10,
               draw_outcome_dendrogram=True,
               draw_guide_dendrogram=True,
               guide_cluster_label_offsets=None,
               use_hdbscan=True,
              ):

    if diagram_kwargs is None:
        diagram_kwargs = {}

    if gene_to_color is None:
        gene_to_color = {}

    if guide_cluster_label_offsets is None:
        guide_cluster_label_offsets = {}

    correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
    correlation_cmap.set_bad('white')

    fc_cmap = plt.get_cmap('RdBu_r')

    cluster_results = ddr.cluster.hierarchcical(pool,
                                          outcomes,
                                          guides,
                                          method='average',
                                          fixed_guide=fixed_guide,
                                          min_UMIs=min_UMIs,
                                          num_outcome_clusters=num_outcome_clusters,
                                          num_guide_clusters=num_guide_clusters,
                                         )
    
    if use_hdbscan:
        hdbscan_results = ddr.cluster.hdbscan(pool, outcomes, guides, cluster_selection_epsilon=0.3)
        cluster_results.update(hdbscan_results)

    if cluster == 'both' or cluster == 'guides':
        final_guide_order = cluster_results['clustered_guide_order']
    else:
        final_guide_order = sorted(cluster_results['clustered_guide_order'])

    if cluster == 'both' or cluster == 'outcomes':
        final_outcome_order = cluster_results['clustered_outcome_order']
    else:
        final_outcome_order = outcomes

    num_guides = len(final_guide_order)
    num_outcomes = len(final_outcome_order)

    window_start, window_end = diagram_kwargs.get('window', (-30, 30))
    window_size = window_end - window_start + 1

    inches_per_guide = 14 / 100

    fc_ax_width_inches = inches_per_guide * num_guides
    fc_ax_height_inches = inches_per_guide * num_outcomes

    # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
    diagram_width_inches = inches_per_guide * window_size  * 0.75
    diagram_height_inches = fc_ax_height_inches

    axs = {}

    if diagram_ax_rectangle is None:
        fig, diagram_ax = plt.subplots(figsize=(diagram_width_inches, diagram_height_inches))
        diagram_ax_rectangle = (0, 0, 1, 1)
    else:
        diagram_ax = fig.add_axes(diagram_ax_rectangle)
    
    axs['diagrams'] = diagram_ax

    fig_width_inches, fig_height_inches = fig.get_size_inches()

    height_per_guide = inches_per_guide / fig_height_inches
    width_per_guide = inches_per_guide / fig_width_inches

    # Draw outcome diagrams.

    ddr.visualize.outcome_diagrams.plot(final_outcome_order[::-1],
                                        pool.target_info,
                                        ax=diagram_ax,
                                        **diagram_kwargs,
                                       )

    diagram_ax.set_title('distance from cut site (nts)', size=10, pad=15)
    diagram_ax.set_position(diagram_ax_rectangle)

    diagram_ax_p = diagram_ax.get_position()

    # Draw fold-change heatmap.

    fc_x0 = diagram_ax_p.x1 + diagram_ax_p.width * 0.02
    fc_y0 = diagram_ax_p.y0
    fc_width = fc_ax_width_inches / fig_width_inches
    fc_height = fc_ax_height_inches / fig_height_inches

    fc_rectangle = (fc_x0, fc_y0, fc_width, fc_height)

    fc_ax = fig.add_axes(fc_rectangle, sharey=diagram_ax)
    axs['fold_changes'] = fc_ax

    fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[final_outcome_order, final_guide_order]

    heatmap_kwargs = dict(cmap=fc_cmap, vmin=-2, vmax=2, interpolation='none')
    heatmap_im = fc_ax.imshow(fcs, **heatmap_kwargs)

    fc_ax.axis('off')

    fc_ax_p = fc_ax.get_position()

    # Draw outcome correlations.

    corrs = cluster_results['outcome_correlations']
    
    gap = 0.5 * width_per_guide
    outcome_corr_x0 = fc_ax_p.x1 + gap
    outcome_corr_y0 = fc_ax_p.y0
    outcome_corr_height = fc_ax_p.height
    outcome_corr_width = outcome_corr_height * fig_height_inches / 2 / fig_width_inches

    outcome_corr_rectangle = (outcome_corr_x0, outcome_corr_y0, outcome_corr_width, outcome_corr_height)

    outcome_corr_ax = fig.add_axes(outcome_corr_rectangle)
    axs['outcome_correlations'] = outcome_corr_ax

    for side in ['left', 'top', 'bottom']:
        outcome_corr_ax.spines[side].set_visible(False)

    outcome_corr_ax.spines['right'].set_color('white')

    mask = np.tri(len(corrs), k=-1)
    masked = np.ma.array(corrs)

    outcome_corr_im = outcome_corr_ax.imshow(masked, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

    transform = matplotlib.transforms.Affine2D().rotate_deg(45) + outcome_corr_ax.transData
    outcome_corr_im.set_transform(transform)

    outcome_corr_ax.set_xticks([])
    outcome_corr_ax.set_yticks([])

    diagonal_length = np.sqrt(2 * len(corrs)**2)

    outcome_corr_ax.set_xlim(0, diagonal_length / 2)
    outcome_corr_ax.set_ylim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diagonal_length)

    # Draw guide correlations.

    corrs = cluster_results['guide_correlations']

    guide_corr_x0 = fc_ax_p.x0

    guide_corr_width = fc_ax_p.width
    guide_corr_height = guide_corr_width * fig_width_inches / 2 / fig_height_inches

    gap = height_per_guide * 0.5
    if upside_down:
        guide_corr_y0 = fc_ax_p.y0 - gap - guide_corr_height
    else:
        guide_corr_y0 = fc_ax_p.y1 + gap

    guide_corr_rectangle = (guide_corr_x0, guide_corr_y0, guide_corr_width, guide_corr_height)
    guide_corr_ax = fig.add_axes(guide_corr_rectangle)
    axs['guide_correlations'] = guide_corr_ax

    if upside_down:
        mask = np.triu(np.ones((len(corrs), len(corrs))), k=1)
    else:
        mask = np.tri(len(corrs), k=-1)

    masked = np.ma.array(corrs, mask=mask)

    guide_corr_im = guide_corr_ax.imshow(masked, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

    guide_corr_ax.set_xticks([])
    guide_corr_ax.set_yticks([])

    plt.setp(guide_corr_ax.spines.values(), visible=False)

    transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + guide_corr_ax.transData

    guide_corr_im.set_transform(transform)

    diag_length = np.sqrt(2 * len(corrs)**2)

    if upside_down:
        guide_corr_ax.set_ylim(diag_length / 2, 0)
    else:
        guide_corr_ax.set_ylim(0, -diag_length / 2)

    guide_corr_ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    for i, guide in enumerate(final_guide_order):
        x = i
        gene = pool.variable_guide_library.guide_to_gene[guide]
        color = gene_to_color.get(gene, 'black')
        fc_ax.annotate(guide,
                       xy=(x, 1 if upside_down else 0),
                       xycoords=('data', 'axes fraction'),
                       xytext=(0, 3 if upside_down else -3),
                       textcoords='offset points',
                       rotation=90,
                       ha='center',
                       va='bottom' if upside_down else 'top',
                       size=7 if color == 'black' else 8,
                       color=color,
                       weight='normal' if color == 'black' else 'bold',
                      )

    if draw_guide_dendrogram:
        gap_before_clusters = height_per_guide * 6
        guide_clusters_height = height_per_guide * 1
        guide_clusters_width = fc_ax_p.width

        guide_clusters_x0 = fc_ax_p.x0
        guide_clusters_y0 = fc_ax_p.y0 - gap_before_clusters - guide_clusters_height

        guide_clusters_rectangle = (guide_clusters_x0, guide_clusters_y0, guide_clusters_width, guide_clusters_height)
        guide_clusters_ax = fig.add_axes(guide_clusters_rectangle)
        axs['guide_clusters'] = guide_clusters_ax

        num_clusters = len(set(cluster_results['guide_cluster_assignments']))

        palette = sns.husl_palette(num_clusters)

        id_to_color = {i: palette[i] for i in range(num_clusters)}

        cluster_blocks = ddr.cluster.get_cluster_blocks(cluster_results['guide_cluster_assignments'])
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y = 1
                x_start = block_start + 0.1
                x_end = block_end + 0.9
                guide_clusters_ax.plot([x_start, x_end],
                                       [y, y],
                                       color=id_to_color[cluster_id],
                                       linewidth=6,
                                       solid_capstyle='butt',
                                       clip_on=False,
                                      )

        guide_clusters_ax.set_ylim(0.5, 1.5)
        guide_clusters_ax.set_xlim(0, num_guides)
        guide_clusters_ax.axis('off')

        gene_text_size = 14

        for cluster_id in cluster_blocks:
            if cluster_id == -1:
                continue

            blocks = cluster_blocks[cluster_id]
            for block in blocks:
                if block[1] - block[0] == 0:
                    continue

                genes = cluster_results['cluster_genes'][cluster_id]
                gene_and_counts = sorted(genes.most_common(), key=lambda gene_and_count: (-gene_and_count[1], gene_and_count[0]))
                for i, (gene, count) in enumerate(gene_and_counts):
                    count_string = f' x{count}' if count > 1 else '' 
                    guide_clusters_ax.annotate(f'{gene}{count_string}',
                                xy=(np.mean(blocks[0]) + 0.5, 0),
                                xycoords=('data', 'axes fraction'),
                                xytext=(0, -(gene_text_size + 1) * (i + 1 + guide_cluster_label_offsets.get(cluster_id, 0))),
                                textcoords='offset points',
                                ha='center',
                                va='top',
                                color=id_to_color[cluster_id],
                                size=gene_text_size,
                            )

        #guide_clusters_p = guide_clusters_ax.get_position()

        #guide_dendro_height = height_per_guide * 10
        #guide_dendro_width = fc_ax_p.width

        #gap_after_clusters = height_per_guide * 0.5

        #guide_dendro_x0 = fc_ax_p.x0
        #guide_dendro_y0 = guide_clusters_p.y0 - gap_after_clusters - guide_dendro_height

        #guide_dendro_rectangle = (guide_dendro_x0, guide_dendro_y0, guide_dendro_width, guide_dendro_height)
        #guide_dendro_ax = fig.add_axes(guide_dendro_rectangle)
        #axs['guide_dendro'] = guide_dendro_ax

        #scipy.cluster.hierarchy.dendrogram(cluster_results['guide_linkage'],
        #                                   ax=guide_dendro_ax,
        #                                   color_threshold=-1,
        #                                   above_threshold_color='black',
        #                                   orientation='bottom',
        #                                  ) 

        #guide_dendro_ax.axis('off')

    if draw_outcome_dendrogram:
        gap_before_clusters = width_per_guide * 0.5
        outcome_clusters_height = fc_ax_p.height
        outcome_clusters_width = width_per_guide * 1

        outcome_clusters_x0 = diagram_ax_p.x0 - gap_before_clusters - outcome_clusters_width
        outcome_clusters_y0 = diagram_ax_p.y0

        outcome_clusters_rectangle = (outcome_clusters_x0, outcome_clusters_y0, outcome_clusters_width, outcome_clusters_height)
        outcome_clusters_ax = fig.add_axes(outcome_clusters_rectangle)
        axs['outcome_clusters'] = outcome_clusters_ax

        num_clusters = len(set(cluster_results['outcome_cluster_assignments']))

        palette = sns.color_palette('colorblind', n_colors=num_clusters)

        id_to_color = {i: palette[i] for i in range(num_clusters)}

        cluster_blocks = ddr.cluster.get_cluster_blocks(cluster_results['outcome_cluster_assignments'])
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y_start = block_start + 0.1
                y_end = block_end + 0.9
                x = 1
                outcome_clusters_ax.plot([x, x],
                                         [y_start, y_end],
                                         color=id_to_color[cluster_id],
                                         linewidth=6,
                                         solid_capstyle='butt',
                                         clip_on=False,
                                        )

        outcome_clusters_ax.set_xlim(0.5, 1.5)
        outcome_clusters_ax.set_ylim(0, num_outcomes)
        outcome_clusters_ax.axis('off')

        outcome_clusters_p = outcome_clusters_ax.get_position()

        outcome_dendro_height = diagram_ax_p.height
        outcome_dendro_width = width_per_guide * 10

        gap_after_clusters = height_per_guide * 0.5

        outcome_dendro_x0 = outcome_clusters_p.x0 - gap_after_clusters - outcome_dendro_width
        outcome_dendro_y0 = diagram_ax_p.y0

        outcome_dendro_rectangle = (outcome_dendro_x0, outcome_dendro_y0, outcome_dendro_width, outcome_dendro_height)
        outcome_dendro_ax = fig.add_axes(outcome_dendro_rectangle)
        axs['outcome_dendro'] = outcome_dendro_ax

        scipy.cluster.hierarchy.dendrogram(cluster_results['outcome_linkage'],
                                           ax=outcome_dendro_ax,
                                           color_threshold=-1,
                                           above_threshold_color='black',
                                           orientation='left',
                                          ) 

        outcome_dendro_ax.axis('off')

    else:
        # Draw non-targeting fractions.

        frequency_width = width_per_guide * 20
        frequency_gap = width_per_guide * 1

        frequency_x0 = diagram_ax_p.x0 - frequency_gap - frequency_width
        frequency_y0 = diagram_ax_p.y0
        frequency_height = diagram_ax_p.height
        
        frequency_rectangle = (frequency_x0, frequency_y0, frequency_width, frequency_height)

        frequency_ax = fig.add_axes(frequency_rectangle, sharey=fc_ax)
        axs['frequencies'] = frequency_ax

        frequencies = pool.non_targeting_fractions('perfect', 'none').loc[final_outcome_order]

        xs = np.log10(frequencies)
        ys = np.arange(len(frequencies))
        frequency_ax.plot(xs, ys, '.', markeredgewidth=0, markersize=10, alpha=0.9, clip_on=False, color='black')

        x_lims = np.log10(np.array([2e-3, 2e-1]))

        for exponent in [3, 2, 1]:
            xs = np.log10(np.arange(1, 10) * 10**-exponent)        
            for x in xs:
                if x_lims[0] <= x <= x_lims[1]:
                    frequency_ax.axvline(x, color='black', alpha=0.07, clip_on=False)

        x_ticks = [x for x in [2e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1] if x_lims[0] <= np.log10(x) <= x_lims[1]]

        frequency_ax.set_xticks(np.log10(x_ticks))
        frequency_ax.set_xticklabels([f'{100 * x:g}' for x in x_ticks])

        for side in ['left', 'right', 'bottom']:
            frequency_ax.spines[side].set_visible(False)

        frequency_ax.xaxis.tick_top()
        frequency_ax.set_xlim(*x_lims)
        frequency_ax.invert_xaxis()

        frequency_ax.set_title('percentage of outcomes\nin cells containing\nnon-targeting guides', size=10, pad=15)

    if draw_colorbars:
        corr_ax_p = outcome_corr_ax.get_position()
        # Draw correlation colorbar.
        width = corr_ax_p.width * 4 / num_outcomes
        height = corr_ax_p.height * 0.4
        cbar_rectangle = (corr_ax_p.x0 + corr_ax_p.width * 0.5 - width * 0.5,
                          corr_ax_p.y1,
                          width,
                          height,
                         )
        cbar_ax = fig.add_axes(cbar_rectangle)
        cbar = fig.colorbar(guide_corr_im, cax=cbar_ax, ticks=[-1, 0, 1])
        cbar_ax.annotate('correlation\nbetween\noutcome\nredistribution\nprofiles',
                         xy=(1, 0.5),
                         xycoords='axes fraction',
                         xytext=(60, 0),
                         textcoords='offset points',
                         ha='center',
                         va='center',
                        )

        cbar.outline.set_alpha(0.1)

        #ddr.visualize.heatmap.add_fold_change_colorbar(fig, heatmap_im, -0.05, 0.4, 0.15, 0.02)

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

        guide_corr_ax.annotate(pool.group,
                xycoords='axes fraction',
                textcoords='offset points',
                ha='center',
                **title_kwargs,
        )

    return fig, axs, cluster_results
