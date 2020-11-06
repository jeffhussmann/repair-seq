import copy

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
               upside_down=False,
               layout_kwargs=None,
               fixed_guide='none',
               cluster='both',
               outcome_diagram='heatmap',
               nt_guide_color='C1',
               gene_to_color=None,
               min_UMIs=None,
               draw_colorbars=True,
              ):

    if layout_kwargs is None:
        layout_kwargs = {}

    if gene_to_color is None:
        gene_to_color = {}

    cmap = copy.copy(plt.get_cmap('PuOr_r'))
    cmap.set_bad('white')

    clustered_guide_order, clustered_outcome_order, correlations = ddr.cluster.cluster(pool,
                                                                            outcomes,
                                                                            guides,
                                                                            method='average',
                                                                            fixed_guide=fixed_guide,
                                                                            min_UMIs=min_UMIs,
                                                                            )

    if cluster == 'both' or cluster == 'guides':
        final_guide_order = clustered_guide_order
    else:
        final_guide_order = sorted(clustered_guide_order)

    if cluster == 'both' or cluster == 'outcomes':
        final_outcome_order = clustered_outcome_order
    else:
        final_outcome_order = outcomes

    if upside_down:
        mask = np.triu(np.ones((len(correlations['guides']), len(correlations['guides']))), k=1)
    else:
        mask = np.tri(len(correlations['guides']), k=-1)

    masked = np.ma.array(correlations['guides'], mask=mask)

    inches_per_guide = 16 / 100
    corr_size_inches = len(final_guide_order) * inches_per_guide
    fig, correlation_ax = plt.subplots(figsize=(corr_size_inches, corr_size_inches))

    correlation_im = correlation_ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, interpolation='none')

    correlation_ax.set_xticks([])
    correlation_ax.set_yticks([])

    for spine in correlation_ax.spines.values():
        spine.set_visible(False)

    transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + correlation_ax.transData

    correlation_im.set_transform(transform)

    diag_length = np.sqrt(2 * len(correlations['guides'])**2)

    if upside_down:
        correlation_ax.set_ylim(diag_length / 2 + 1, 0)
    else:
        correlation_ax.set_ylim(0, -diag_length / 2 - 1)

    correlation_ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    for i, guide in enumerate(final_guide_order):
        x = 2**0.5 * i
        gene = pool.variable_guide_library.guide_to_gene[guide]
        color = gene_to_color.get(gene, 'black')
        correlation_ax.annotate(guide,
                                xy=(x, 0),
                                xytext=(0, 3 if upside_down else -3),
                                textcoords='offset points',
                                rotation=90,
                                ha='center',
                                va='bottom' if upside_down else 'top',
                                size=7 if color == 'black' else 8,
                                color=color,
                                weight='normal' if color == 'black' else 'bold',
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

    if draw_colorbars:
        correlation_ax.annotate(pool.group,
                xycoords='axes fraction',
                textcoords='offset points',
                ha='center',
                **title_kwargs,
        )

    corr_ax_p = correlation_ax.get_position()

    if draw_colorbars:
        # Draw correlation colorbar.
        cbar_ax = fig.add_axes((corr_ax_p.x1 - corr_ax_p.width * 0.1, corr_ax_p.y0 + corr_ax_p.height * 0.4, corr_ax_p.width * 0.02, corr_ax_p.height * 0.4))
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

    heatmap_width = corr_ax_p.width
    heatmap_height = heatmap_width * len(final_outcome_order) / len(final_guide_order)
    # Make gap equal to height for 7 outcome rows.
    gap = heatmap_height / len(final_outcome_order) * 6

    if upside_down:
        y0 = corr_ax_p.y1 + gap
    else:
        y0 = corr_ax_p.y0 - heatmap_height - gap

    fc_ax = fig.add_axes((corr_ax_p.x0, y0, heatmap_width, heatmap_height))

    fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[final_outcome_order, final_guide_order]

    heatmap_kwargs = dict(cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2, interpolation='none')
    heatmap_im = fc_ax.imshow(fcs, **heatmap_kwargs)

    fc_ax.axis('off')

    fc_ax_p = fc_ax.get_position()

    window_start, window_end = layout_kwargs['window']
    window_size = window_end - window_start + 1

    # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
    diagram_width = fc_ax_p.width * window_size / len(final_guide_order) * 0.75
    diagram_gap = diagram_width * 0.02

    diagram_ax = fig.add_axes((fc_ax_p.x0 - diagram_width - diagram_gap, fc_ax_p.y0, diagram_width, fc_ax_p.height), sharey=fc_ax)
    _ = ddr.visualize.outcome_diagrams.plot(final_outcome_order[::-1],
                              pool.target_info,
                              ax=diagram_ax,
                              **layout_kwargs,
                             )

    diagram_ax.set_title('distance from cut site (nts)', size=10, pad=15)

    outcome_corr_ax = fig.add_axes((fc_ax_p.x0, fc_ax_p.y0, fc_ax_p.height, fc_ax_p.height))

    for side in ['left', 'top', 'bottom']:
        outcome_corr_ax.spines[side].set_visible(False)

    outcome_corr_ax.spines['right'].set_color('white')

    if outcome_diagram == 'heatmap' and (cluster == 'both' or cluster == 'outcomes'):
        mask = np.tri(len(correlations['outcomes']), k=-1)
        masked = np.ma.array(correlations['outcomes'], mask=mask)

        correlation_im = outcome_corr_ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, interpolation='none')

        transform = matplotlib.transforms.Affine2D().rotate_deg(45) + outcome_corr_ax.transData
        correlation_im.set_transform(transform)

    outcome_corr_ax.set_xticks([])
    outcome_corr_ax.set_yticks([])

    diagonal_length = np.sqrt(2 * len(correlations['outcomes'])**2)

    outcome_corr_ax.set_xlim(0, diagonal_length / 2)
    outcome_corr_ax.set_ylim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diagonal_length)

    # Can't figure out how to prevent re-positioning of the axes, so simply reset it
    # to the intended position after done.
    outcome_corr_ax.set_position((fc_ax_p.x1 + 0.01 * fc_ax_p.width, fc_ax_p.y0, fc_ax_p.height / 2, fc_ax_p.height))

    # Draw non-targeting fractions.

    frequency_width = fc_ax_p.width * 0.15 * 100 / guides
    frequency_gap = frequency_width * 0.1

    diagram_ax_p = diagram_ax.get_position()

    frequency_ax = fig.add_axes((diagram_ax_p.x0 - frequency_gap - frequency_width, fc_ax_p.y0, frequency_width, fc_ax_p.height), sharey=fc_ax)

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
        ddr.visualize.heatmap.add_fold_change_colorbar(fig, heatmap_im, -0.05, 0.4, 0.15, 0.02)

    return fig, final_guide_order, final_outcome_order

def clustermap_new(pool,
               outcomes=40,
               guides=100,
               diagram_ax_rectangle=None,
               fig=None,
               upside_down=False,
               layout_kwargs=None,
               fixed_guide='none',
               cluster='both',
               outcome_diagram='heatmap',
               nt_guide_color='C1',
               gene_to_color=None,
               min_UMIs=None,
               draw_colorbars=True,
               num_clusters=10,
              ):

    if layout_kwargs is None:
        layout_kwargs = {}

    if gene_to_color is None:
        gene_to_color = {}

    correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
    correlation_cmap.set_bad('white')

    fc_cmap = plt.get_cmap('RdBu_r')

    #clustered_guide_order, clustered_outcome_order, correlations = ddr.cluster.cluster(pool,
    cluster_results = ddr.cluster.cluster(pool,
                                          outcomes,
                                          guides,
                                          method='average',
                                          fixed_guide=fixed_guide,
                                          min_UMIs=min_UMIs,
                                         )

    if cluster == 'both' or cluster == 'guides':
        final_guide_order = cluster_results['guide_order']
    else:
        final_guide_order = sorted(cluster_results['guide_order'])

    if cluster == 'both' or cluster == 'outcomes':
        final_outcome_order = cluster_results['outcome_order']
    else:
        final_outcome_order = outcomes

    num_guides = len(final_guide_order)
    num_outcomes = len(final_outcome_order)

    window_start, window_end = layout_kwargs['window']
    window_size = window_end - window_start + 1

    inches_per_guide = 14 / 100

    fc_ax_width_inches = inches_per_guide * num_guides
    fc_ax_height_inches = inches_per_guide * num_outcomes

    # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
    diagram_width_inches = inches_per_guide * window_size  * 0.75
    diagram_height_inches = fc_ax_height_inches

    if diagram_ax_rectangle is None:
        fig, diagram_ax = plt.subplots(figsize=(diagram_width_inches, diagram_height_inches))
        diagram_ax_rectangle = (0, 0, 1, 1)
    else:
        diagram_ax = fig.add_axes(diagram_ax_rectangle)

    fig_width_inches, fig_height_inches = fig.get_size_inches()

    # Draw outcome diagrams.

    ddr.visualize.outcome_diagrams.plot(final_outcome_order[::-1],
                                        pool.target_info,
                                        ax=diagram_ax,
                                        **layout_kwargs,
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

    fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[final_outcome_order, final_guide_order]

    heatmap_kwargs = dict(cmap=fc_cmap, vmin=-2, vmax=2, interpolation='none')
    heatmap_im = fc_ax.imshow(fcs, **heatmap_kwargs)

    fc_ax.axis('off')

    fc_ax_p = fc_ax.get_position()

    # Draw outcome correlations.

    corrs = cluster_results['outcome_correlations']
    
    gap = inches_per_guide * 0.5 / fig_width_inches
    outcome_corr_x0 = fc_ax_p.x1 + gap
    outcome_corr_y0 = fc_ax_p.y0
    outcome_corr_height = fc_ax_p.height
    outcome_corr_width = outcome_corr_height * fig_height_inches / 2 / fig_width_inches

    outcome_corr_rectangle = (outcome_corr_x0, outcome_corr_y0, outcome_corr_width, outcome_corr_height)

    outcome_corr_ax = fig.add_axes(outcome_corr_rectangle)

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

    gap = inches_per_guide * 0.5 / fig_height_inches
    if upside_down:
        guide_corr_y0 = fc_ax_p.y0 - gap - guide_corr_height
    else:
        guide_corr_y0 = fc_ax_p.y1 + gap

    guide_corr_rectangle = (guide_corr_x0, guide_corr_y0, guide_corr_width, guide_corr_height)
    guide_corr_ax = fig.add_axes(guide_corr_rectangle)

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

    if True:
        guide_dendro_x0 = fc_ax_p.x0

        guide_dendro_width = fc_ax_p.width
        guide_dendro_height = inches_per_guide * 20 / fig_height_inches

        gap = inches_per_guide * 11 / fig_height_inches

        guide_dendro_y0 = fc_ax_p.y0 - gap - guide_dendro_height

        guide_dendro_rectangle = (guide_dendro_x0, guide_dendro_y0, guide_dendro_width, guide_dendro_height)
        guide_dendro_ax = fig.add_axes(guide_dendro_rectangle)

        scipy.cluster.hierarchy.dendrogram(cluster_results['guide_linkage'],
                                           ax=guide_dendro_ax,
                                           color_threshold=-1,
                                           above_threshold_color='black',
                                           orientation='bottom',
                                          ) 

        guide_dendro_ax.axis('off')

        guide_clusters_x0 = fc_ax_p.x0
        guide_clusters_width = fc_ax_p.width
        guide_clusters_height = inches_per_guide * 4 / fig_height_inches

        gap = inches_per_guide * 6 / fig_height_inches

        guide_clusters_y0 = fc_ax_p.y0 - gap - guide_clusters_height

        guide_clusters_rectangle = (guide_clusters_x0, guide_clusters_y0, guide_clusters_width, guide_clusters_height)
        guide_clusters_ax = fig.add_axes(guide_clusters_rectangle)

        indices = [cluster_results['original_guide_order'].get_loc(g) for g in cluster_results['guide_order']]

        cluster_ids = scipy.cluster.hierarchy.fcluster(cluster_results['guide_linkage'], criterion='maxclust', t=num_clusters)

        ordered_cluster_ids = [cluster_ids[i] for i in indices]

        num_clusters = len(set(ordered_cluster_ids))

        palette = sns.husl_palette(num_clusters)

        id_to_color = {i: palette[i - 1] for i in range(1, num_clusters + 1)}

        for cluster_id in range(1, num_clusters + 1):
            for x, v in enumerate(ordered_cluster_ids):
                if v == cluster_id:
                    guide_clusters_ax.plot([x, x + 1], [cluster_id - 1, cluster_id - 1],
                                           color=id_to_color[cluster_id],
                                           linewidth=4,
                                           solid_capstyle='butt',
                                           clip_on=False,
                                          )

        guide_clusters_ax.set_ylim(-0.5, num_clusters - 0.5)
        guide_clusters_ax.axis('off')

    # Draw non-targeting fractions.

    frequency_width = inches_per_guide * 20 / fig_width_inches
    frequency_gap = inches_per_guide * 1 / fig_width_inches

    frequency_x0 = diagram_ax_p.x0 - frequency_gap - frequency_width
    frequency_y0 = diagram_ax_p.y0
    frequency_height = diagram_ax_p.height
    
    frequency_rectangle = (frequency_x0, frequency_y0, frequency_width, frequency_height)

    frequency_ax = fig.add_axes(frequency_rectangle, sharey=fc_ax)

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

    axs = {
        'diagrams': diagram_ax,
        'outcome_correlations': outcome_corr_ax,
        'guide_correlations': guide_corr_ax,
        'frequencies': frequency_ax,
    }

    if draw_colorbars:
        corr_ax_p = outcome_corr_ax.get_position()
        # Draw correlation colorbar.
        cbar_ax = fig.add_axes((corr_ax_p.x1 - corr_ax_p.width * 0.1, corr_ax_p.y0 + corr_ax_p.height * 0.4, corr_ax_p.width * 0.02, corr_ax_p.height * 0.4))
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

        ddr.visualize.heatmap.add_fold_change_colorbar(fig, heatmap_im, -0.05, 0.4, 0.15, 0.02)

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

    return fig, final_guide_order, final_outcome_order, axs