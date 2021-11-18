import itertools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bokeh.palettes
import scipy.cluster.hierarchy

from . import outcome_diagrams
import repair_seq.pooled_screen
import repair_seq.visualize

idx = pd.IndexSlice
ALL_NON_TARGETING = repair_seq.pooled_screen.ALL_NON_TARGETING

def add_fold_change_colorbar(fig, im,
                             x0=None,
                             y0=None,
                             width=None,
                             height=None,
                             cbar_ax=None,
                             baseline_condition_name='non-targeting',
                             label_interpretation=True,
                             text_size=8,
                            ):
    if cbar_ax is None:
        cbar_ax = fig.add_axes((x0, y0, width, height)) 

    v_min, v_max = im.get_clim()
    ticks = np.arange(v_min, v_max + 1)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=ticks)

    tick_labels = [str(int(t) if t.is_integer() else t) for t in ticks]
    tick_labels[0] = r'$\leq$' + tick_labels[0]
    tick_labels[-1] = r'$\geq$' + tick_labels[-1]
    cbar.ax.set_xticklabels(tick_labels, size=text_size)

    if label_interpretation:
        cbar_ax.xaxis.tick_top()

    if label_interpretation:
        cbar_ax.annotate('gene\n activity\npromotes\noutcome',
                        xy=(0, 0),
                        xycoords='axes fraction',
                        xytext=(0, -5),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        size=text_size,
                        )

        cbar_ax.annotate('gene\nactivity\nsuppresses\noutcome',
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(0, -5),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        size=text_size,
                        )

    cbar_ax.annotate(f'log$_2$ fold change\nin frequency from\n{baseline_condition_name}',
                     xy=(0.5, 1),
                     xycoords='axes fraction',
                     xytext=(0, 20 if label_interpretation else 5),
                     textcoords='offset points',
                     va='bottom',
                     ha='center',
                     size=text_size,
                    )

    cbar.outline.set_alpha(0.1)

    return cbar

def genes(pool,
          genes,
          fixed_guide='none',
          heatmap_pools=None,
          pools_to_compare=None,
          only_best_promoter=False,
          guide_status='perfect',
          outcomes=None,
          outcome_group_sizes=None,
          zoom=False,
          just_percentages=False,
          ax_order=None,
          log_2=True,
          layout_kwargs=None,
          v_min=-2, v_max=2,
          frequency_max_multiple=1.05,
          draw_nt_fracs='combined',
          draw_heatmaps=True,
          draw_colorbar=True,
          highlight_targeting_guides=False,
          genes_to_plot=None,
          genes_to_heatmap=None,
          gene_to_colors=None,
          bad_guides=None,
          panel_width_multiple=2.7,
          just_heatmaps=False,
          show_pool_name=True,
          guide_aliases=None,
          gene_to_sort_by=None,
          show_ax_titles=True,
          overall_title=None,
          use_high_frequency=False,
          different_colors_if_one_gene=True,
          log10_x_lims=None,
          clip_on=False,
         ):

    if heatmap_pools is None:
        heatmap_pools = [(pool, fixed_guide)]

    colors = itertools.cycle(bokeh.palettes.Set2[8])
    pool_to_color = {(p.group, fixed_guide): color for (p, fixed_guide), color in zip(heatmap_pools, colors)}

    if guide_aliases is None:
        guide_aliases = {}

    if layout_kwargs is None:
        layout_kwargs = {'draw_all_sequence': False}

    if bad_guides is None:
        bad_guides = []

    if gene_to_colors is None:
        gene_to_colors = {'negative_control': ['C0']*1000}
        real_genes = [g for g in genes if g != 'negative_control']

        if len(real_genes) > 1 or not different_colors_if_one_gene:
            gene_to_colors.update(dict(zip(real_genes, repair_seq.visualize.colors_list)))

        elif len(real_genes) == 1:
            gene = real_genes[0]
            guides = pool.variable_guide_library.gene_guides(gene, only_best_promoter)
            if len(guides) > 6:
                gene_to_colors.update({gene: ['C0']*1000})
            else:
                gene_to_colors.update({gene: repair_seq.visualize.good_colors})
        
    if ax_order is None:
        if just_percentages:
            ax_order = [
                'frequency',
            ]
        elif zoom:
            ax_order = [
                'frequency_zoom',
                'change_zoom',
                'log2_fold_change' if log_2 else 'fold_change',
            ]
        else:
            ax_order = [
                'frequency',
                'log10_frequency',
                'change',
                'log2_fold_change' if log_2 else 'fold_change',
            ]

    if pools_to_compare is not None and 'log2_between_pools' not in ax_order:
        ax_order.append('log2_between_pools')

    if outcomes is None:
        outcome_order, auto_outcome_group_sizes = pool.rational_outcome_order(fixed_guide, by_frequency=True)
    elif isinstance(outcomes, int):
        outcome_order, auto_outcome_group_sizes = pool.rational_outcome_order(fixed_guide, num_outcomes=outcomes, by_frequency=True)
    else:
        outcome_order = outcomes
        auto_outcome_group_sizes = [len(outcome_order)]

    if outcome_group_sizes is None:
        outcome_group_sizes = auto_outcome_group_sizes

    if use_high_frequency:
        def get_log2_fold_changes(p, f=fixed_guide):
            return p.high_frequency_log2_fold_changes[f]
        def get_outcome_fractions(p, f=fixed_guide):
            return p.high_frequency_outcome_fractions[f]
    else:
        def get_log2_fold_changes(p, f=fixed_guide):
            return p.log2_fold_changes(guide_status=guide_status, fixed_guide=f)
        def get_outcome_fractions(p, f=fixed_guide):
            return p.outcome_fractions(guide_status=guide_status)[f]

    if gene_to_sort_by is not None:
        outcome_order = pool.sort_outcomes_by_gene_phenotype(outcome_order, gene_to_sort_by)

    num_outcomes = len(outcome_order)

    ys = np.arange(len(outcome_order))[::-1]
    
    fig, ax_array = plt.subplots(1, len(ax_order),
                                 figsize=(panel_width_multiple * len(ax_order), 48 * num_outcomes / 200),
                                 sharey=True,
                                 gridspec_kw={'wspace': 0.05},
                                )
    if len(ax_order) == 1:
        ax_array = [ax_array]
    axs = dict(zip(ax_order, ax_array))
    
    for ax in axs.values():
        ax.xaxis.tick_top()

    guides = list(pool.variable_guide_library.gene_guides(genes, only_best_promoter))
    guides = [g for g in guides if g not in bad_guides]

    if draw_nt_fracs == 'separate':
        guides.extend(pool.variable_guide_library.non_targeting_guides)

    def dot_and_line(xs, ax, color, label, line_width=1, marker_size=3, marker_alpha=0.6, line_alpha=0.25):
        ax.plot(list(xs), ys, 'o', markeredgewidth=0, markersize=marker_size, color=color, alpha=marker_alpha, label=label, clip_on=clip_on)

        #group_boundaries = np.cumsum([0] + outcome_group_sizes)
        #group_start_and_ends = list(zip(group_boundaries, group_boundaries[1:]))
        #for start, end in group_start_and_ends:
        #    ax.plot(list(xs)[start:end], ys[start:end], '-', linewidth=line_width, color=color, alpha=line_alpha, clip_on=False)

        ax.plot(list(xs), ys, '-', linewidth=line_width, color=color, alpha=line_alpha, clip_on=clip_on)
    
    def guide_to_color(guide, pool):
        gene = pool.variable_guide_library.guide_to_gene[guide]
        i = list(pool.variable_guide_library.gene_guides(gene, only_best_promoter)).index(guide)
        color = gene_to_colors[gene][i]
        
        return color

    def get_nt_fractions(pool, fixed_guide):
        if use_high_frequency:
            return pool.high_frequency_outcome_fractions[fixed_guide, 'all_non_targeting'].reindex(outcome_order, fill_value=0)
        else:
            return pool.non_targeting_fractions(guide_status='all', fixed_guide=fixed_guide).reindex(outcome_order, fill_value=0)

    fractions = get_outcome_fractions(pool).reindex(outcome_order, fill_value=0)
    nt_fracs = get_nt_fractions(pool, fixed_guide)
    absolute_change = fractions.sub(nt_fracs, axis=0)
    log2_fold_changes = get_log2_fold_changes(pool).reindex(outcome_order, fill_value=0)
    
    if draw_nt_fracs == 'combined':
        for heatmap_pool, heatmap_fixed_guide in heatmap_pools:
            outcome_fractions = get_outcome_fractions(heatmap_pool, heatmap_fixed_guide).reindex(outcome_order, fill_value=0)
            nt_guides = heatmap_pool.variable_guide_library.non_targeting_guides
            for key in ('frequency', 'frequency_zoom', 'log10_frequency'):
                if key in axs:
                    xs = get_nt_fractions(heatmap_pool, heatmap_fixed_guide)
                    stds = outcome_fractions[nt_guides].std(axis=1)
                    lowers, uppers = xs - stds, xs + stds

                    if 'log10_frequency' in key:
                        xs = np.log10(np.maximum(xs, 1e-5))
                        lowers = np.log10(np.maximum(lowers, 1e-5))
                        uppers = np.log10(np.maximum(uppers, 1e-5))
                    else:
                        xs = xs * 100
                        lowers = lowers * 100
                        uppers = uppers * 100

                    color = pool_to_color[heatmap_pool.group, heatmap_fixed_guide]

                    # Draw standard deviation bars
                    for y, lower, upper in zip(ys, lowers, uppers): 
                        axs[key].plot([lower, upper], [y, y], color=color, linewidth=1.5, alpha=0.3, clip_on=False)

                    dot_and_line(xs, axs[key], color, 'non-targeting', line_width=1.5, line_alpha=0.9, marker_alpha=1, marker_size=4)

        if 'log2_between_pools' in axs:
            numerator_pool, denominator_pool = pools_to_compare
            numerator_fracs = get_nt_fractions(numerator_pool, fixed_guide)
            denominator_fracs = get_nt_fractions(denominator_pool, fixed_guide)
            xs = np.log2(numerator_fracs / denominator_fracs)
            dot_and_line(xs, axs['log2_between_pools'], 'black', 'between', line_width=1.5, line_alpha=0.9, marker_alpha=1, marker_size=4)
        
    if genes_to_plot is None:
        genes_to_plot = genes

    genes_to_plot = genes_to_plot + ['negative_control']
    
    if genes_to_heatmap is None:
        genes_to_heatmap = genes

    for guide in guides:
        gene = pool.variable_guide_library.guide_to_gene[guide]

        if genes_to_plot is not None:
            if gene not in genes_to_plot:
                continue

        color = guide_to_color(guide, pool)
        label = guide

        for key in ('change', 'change_zoom'):
            if key in axs:
                dot_and_line(absolute_change[guide] * 100, axs[key], color, label)
            
        if gene != 'negative_control':
            if highlight_targeting_guides:
                kwargs = dict(marker_size=5, line_width=2)
            else:
                kwargs = dict()

            if 'log2_fold_change' in axs:
                dot_and_line(log2_fold_changes[guide], axs['log2_fold_change'], color, label, **kwargs)
        
        for key in ('frequency', 'frequency_zoom', 'log10_frequency'):
            if key in axs:
                if highlight_targeting_guides:
                    if pool.variable_guide_library.guide_to_gene[guide] == 'negative_control':
                        kwargs = dict(line_alpha=0.15, marker_size=2)
                    else:
                        kwargs = dict(line_alpha=0.45, marker_size=5, line_width=1.5)
                else:
                    kwargs = dict(line_alpha=0.15)

                xs = fractions[guide]
                if 'log10' in key:
                    UMIs = pool.UMI_counts(guide_status=guide_status).loc[guide]
                    min_frac = 0.5 / UMIs
                    xs = np.maximum(xs, min_frac)
                    xs = np.log10(xs)
                else:
                    xs = xs * 100

                dot_and_line(xs, axs[key], color, label, **kwargs)

    # Apply specific visual styling to log2_between_pools panel.
    if 'log2_between_pools' in axs:
        ax = axs['log2_between_pools']
        ax.set_xlim(-3, 3)
        x_to_alpha = {
            0: 0.5,
        }
        for x in np.arange(-3, 4):
            alpha = x_to_alpha.get(x, 0.1)
            ax.axvline(x, color='black', alpha=alpha, clip_on=False)

        for side in ['left', 'right', 'bottom']:
            ax.spines[side].set_visible(False)

        ax.annotate(f'log2 ({numerator_pool.short_name} /\n {denominator_pool.short_name})',
                    xy=(0.5, 1),
                    xycoords='axes fraction',
                    xytext=(0, 30),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    )

    if log10_x_lims is None:
        log10_x_lims = (np.log10(0.0008), np.log10(0.41))
    
    for key, line_x, title, x_lims in [
        ('change', 0, 'change in percentage', (-8, 8)),
        ('change_zoom', 0, 'change in percentage\n(zoomed)', (-0.75, 0.75)),
        ('fold_change', 1, 'fold change', (0, 5)),
        ('log2_fold_change', 0, 'log2 fold change\nfrom non-targeting', (-3, 3)),
        ('frequency', None, 'percentage of\nrepair outcomes', (0, nt_fracs.max() * 100 * frequency_max_multiple)),
        ('log10_frequency', None, 'percentage of repair outcomes', log10_x_lims),
        ('frequency_zoom', None, 'percentage\n(zoomed)', (0, nt_fracs.max() * 100 * 0.3)),
       ]:

        if key not in axs:
            continue

        if line_x is not None:
            axs[key].axvline(line_x, color='black', alpha=0.3)
       
        if show_ax_titles:
            axs[key].annotate(title,
                              xy=(0.5, 1),
                              xycoords='axes fraction',
                              xytext=(0, 30),
                              textcoords='offset points',
                              ha='center',
                              va='bottom',
                              #size=14,
                             )
                    
        axs[key].xaxis.set_label_coords(0.5, 1.025)

        axs[key].set_xlim(*x_lims)
        
        # Draw lines separating categories.
        if gene_to_sort_by is None:
            for y in np.cumsum(outcome_group_sizes):
                flipped_y = len(outcome_order) - y - 0.5
                axs[key].axhline(flipped_y, color='black', alpha=0.1)
    
        # Apply specific visual styling to log10_frequency panel.
        if key == 'log10_frequency':
            ax = axs['log10_frequency']
            for exponent in [4, 3, 2, 1]:
                xs = np.log10(np.arange(1, 10) * 10**-exponent)        
                for x in xs:
                    if x_lims[0] <= x <= x_lims[1]:
                        ax.axvline(x, color='black', alpha=0.1, clip_on=False)

            x_ticks = [x for x in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1] if x_lims[0] <= np.log10(x) <= x_lims[1]]
            ax.set_xticks(np.log10(x_ticks))
            ax.set_xticklabels([f'{100 * x:g}' for x in x_ticks])

            for side in ['left', 'right', 'bottom']:
                ax.spines[side].set_visible(False)

        if key == 'log2_fold_change':
            ax.grid(axis='x', alpha=0.2)
    
    if draw_heatmaps:
        width, height = fig.get_size_inches()
        
        if just_heatmaps:
            anchor = ax_order[0]
            ax_p = axs[anchor].get_position()
            start_x = ax_p.x0
        else:
            anchor = ax_order[-1]
            ax_p = axs[anchor].get_position()
            start_x = ax_p.x1 + 0.05 * ax_p.width
        
        heatmap_axs = {}
        
        def convert_alias(guide):
            for old, new in guide_aliases.items(): 
                if old in guide:
                    guide = guide.replace(old, new)

            return guide

        first_gene = None

        for gene in genes_to_heatmap:
            for heatmap_i, (heatmap_pool, heatmap_fixed_guide) in enumerate(heatmap_pools):
                gene_guides = heatmap_pool.variable_guide_library.gene_guides(gene, only_best_promoter)
                if bad_guides is not None:
                    gene_guides = [guide for guide in gene_guides if guide not in bad_guides]

                if len(gene_guides) == 0:
                    continue

                if first_gene is None:
                    first_gene = gene
                
                vals = get_log2_fold_changes(heatmap_pool, heatmap_fixed_guide).reindex(outcome_order[::-1], fill_value=0)[gene_guides]
                
                num_rows, num_cols = vals.shape
            
                heatmap_width = ax_p.height * num_cols / num_rows * height / width
                heatmap_ax = fig.add_axes((start_x, ax_p.y0, heatmap_width, ax_p.height), sharey=axs[ax_order[0]])
                        
                im = heatmap_ax.imshow(vals, cmap=plt.get_cmap('RdBu_r'), vmin=v_min, vmax=v_max)

                heatmap_ax.xaxis.tick_top()
                heatmap_ax.set_xticks(np.arange(len(gene_guides)))

                heatmap_ax.set_xticklabels([guide for guide in gene_guides], rotation=90)
                
                # Have to jump through weird hoops because label.set_text() doesn't update in-place.
                labels = []
                for label in heatmap_ax.get_xticklabels():
                    guide = label.get_text()
                    color = guide_to_color(guide, heatmap_pool)
                    label.set_color(color)
                    label.set_text(convert_alias(guide))
                    labels.append(label)

                heatmap_ax.set_xticklabels(labels)

                for spine in heatmap_ax.spines.values():
                    spine.set_visible(False)
                    
                if len(heatmap_pools) > 1:
                    if heatmap_pool == heatmap_pools[-1]:
                        gap_between = 0.4

                        # Draw vertical line separating different genes.
                        if gene != genes_to_heatmap[-1]:
                            line_fig_x = start_x + heatmap_width * (1 + (3 / num_cols) * gap_between * 0.5)
                            heatmap_ax.plot([line_fig_x, line_fig_x], [ax_p.y0 - ax_p.height * 0.05, ax_p.y1 + ax_p.height * 0.05],
                                            clip_on=False,
                                            transform=fig.transFigure,
                                            color='black',
                                            linewidth=2,
                                            alpha=0.5,
                                        )

                    else:
                        gap_between = 0.1

                    # Label each pool with an identifying color stripe.
                    stripe_y = -1 / len(outcome_order)
                    heatmap_ax.plot([0, 1], [stripe_y, stripe_y],
                                    clip_on=False,
                                    transform=heatmap_ax.transAxes,
                                    linewidth=7,
                                    solid_capstyle='butt',
                                    color=pool_to_color[heatmap_pool.group, heatmap_fixed_guide],
                                    )

                    if gene == genes_to_heatmap[0]:
                        # Draw a legend of pool names.
                        heatmap_ax.annotate(heatmap_pool.short_name,
                        #heatmap_ax.annotate(heatmap_fixed_guide,
                                            xy=(0.5, stripe_y),
                                            xycoords='axes fraction',
                                            xytext=(0, -15 * (heatmap_i + 1)),
                                            textcoords='offset points',
                                            ha='center',
                                            va='top',
                                            annotation_clip=False,
                                            color=pool_to_color[heatmap_pool.group, heatmap_fixed_guide],
                                            #arrowprops={'arrowstyle': '-',
                                            #            'alpha': 0.5,
                                            #            'color': pool_to_color[heatmap_pool],
                                            #            'linewidth': 2,
                                            #            },
                                           )

                else:
                    gap_between = 0.1

                start_x += heatmap_width * (1 + (3 / num_cols) * gap_between)

                heatmap_axs[gene] = heatmap_ax

                last_gene = gene

        heatmap_p = heatmap_axs[first_gene].get_position()
        heatmap_x0 = heatmap_p.x0
        heatmap_x1 = heatmap_axs[last_gene].get_position().x1
        heatmap_height = heatmap_p.height

        if draw_colorbar:
            ax_p = axs[ax_order[-1]].get_position()

            bar_x0 = heatmap_x1 + heatmap_p.width

            cbar_offset = -heatmap_p.height / 2
            cbar_height = heatmap_height * 1 / len(outcome_order)
            cbar_width = ax_p.width * 0.75 * 2.7 / panel_width_multiple
            cbar_ax = fig.add_axes((bar_x0, heatmap_p.y1 + cbar_offset, cbar_width, cbar_height)) 

            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

            ticks = np.arange(v_min, v_max + 1)
            cbar.set_ticks(ticks)
            tick_labels = [str(t) for t in ticks]
            tick_labels[0] = r'$\leq$' + tick_labels[0]
            tick_labels[-1] = r'$\geq$' + tick_labels[-1]
            cbar.set_ticklabels(tick_labels)
            cbar_ax.xaxis.tick_top()

            cbar_ax.annotate('gene\nactivity\npromotes\noutcome',
                            xy=(0, 0),
                            xycoords='axes fraction',
                            xytext=(0, -5),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            #size=10,
                            )

            cbar_ax.annotate('gene\nactivity\nsuppresses\noutcome',
                            xy=(1, 0),
                            xycoords='axes fraction',
                            xytext=(0, -5),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            #size=10,
                            )

            cbar_ax.annotate('log2 fold change\nfrom non-targeting',
                            xy=(0.5, 1),
                            xycoords='axes fraction',
                            xytext=(0, 20),
                            textcoords='offset points',
                            va='bottom',
                            ha='center',
                            )
    
    for key in ['frequency_zoom', 'change_zoom', 'fold_change']:
        if key in axs:
            axs[key].set_xticklabels(list(axs[key].get_xticklabels())[:-1] + [''])

    ax_p = axs[ax_order[0]].get_position()

    if 'window' in layout_kwargs:
        window = layout_kwargs['window']
    else:
        window = 70

    if isinstance(window, tuple):
        window_start, window_end = window
    else:
        window_start, window_end = -window, window

    window_size = window_end - window_start

    diagram_width = ax_p.width * window_size / 20 * 2.7 / panel_width_multiple
    diagram_gap = diagram_width * 0.02

    diagram_ax = fig.add_axes((ax_p.x0 - diagram_width - diagram_gap, ax_p.y0, diagram_width, ax_p.height), sharey=axs[ax_order[0]])
    
    outcome_diagrams.plot(outcome_order, pool.target_info,
                          num_outcomes=len(outcome_order),
                          ax=diagram_ax,
                          **layout_kwargs)


    if show_pool_name:
        if overall_title is None:
            overall_title = pool.group
        
        diagram_ax.annotate(overall_title,
                            xy=(0.5, 1),
                            xycoords='axes fraction',
                            xytext=(0, 25),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            size=12,
        )

    if just_heatmaps:
        for ax in ax_array:
            fig.delaxes(ax)

    return fig, outcome_order

def guide_pairs(pool,
                guide_pairs,
                only_best_promoter=False,
                guide_status='perfect',
                outcomes=None,
                outcome_group_sizes=None,
                zoom=False,
                just_percentages=False,
                ax_order=None,
                log_2=True,
                layout_kwargs=None,
                v_min=-2, v_max=2,
                frequency_max_multiple=1.05,
                draw_nt_fracs='combined',
                draw_heatmaps=True,
                draw_colorbar=True,
                highlight_targeting_guides=False,
                genes_to_plot=None,
                genes_to_heatmap=None,
                gene_to_colors=None,
                bad_guides=None,
                panel_width_multiple=2.7,
                just_heatmaps=False,
                show_pool_name=True,
                guide_aliases=None,
                gene_to_sort_by=None,
                show_ax_titles=True,
                log10_x_lims=None,
         ):

    colors = itertools.cycle(bokeh.palettes.Set2[8])

    if guide_aliases is None:
        guide_aliases = {}

    if layout_kwargs is None:
        layout_kwargs = {'draw_all_sequence': False}

    if bad_guides is None:
        bad_guides = []

    if ax_order is None:
        if just_percentages:
            ax_order = [
                'frequency',
            ]
        elif zoom:
            ax_order = [
                'frequency_zoom',
                'change_zoom',
                'log2_fold_change' if log_2 else 'fold_change',
            ]
        else:
            ax_order = [
                'frequency',
                'change',
                'log2_fold_change' if log_2 else 'fold_change',
            ]

    if outcomes is None:
        outcome_order, auto_outcome_group_sizes = pool.rational_outcome_order(None)
    elif isinstance(outcomes, int):
        outcome_order, auto_outcome_group_sizes = pool.rational_outcome_order(None, num_outcomes=outcomes)
    else:
        outcome_order = outcomes
        auto_outcome_group_sizes = [len(outcome_order)]

    if outcome_group_sizes is None:
        outcome_group_sizes = auto_outcome_group_sizes

    num_outcomes = len(outcome_order)

    ys = np.arange(len(outcome_order))[::-1]
    
    fig, ax_array = plt.subplots(1, len(ax_order),
                                 figsize=(panel_width_multiple * len(ax_order), 48 * num_outcomes / 200),
                                 sharey=True,
                                 gridspec_kw={'wspace': 0.05},
                                )
    if len(ax_order) == 1:
        ax_array = [ax_array]
    axs = dict(zip(ax_order, ax_array))
    
    for ax in axs.values():
        ax.xaxis.tick_top()

    def dot_and_line(xs, ax, color, label, line_width=1, marker_size=3, marker_alpha=0.6, line_alpha=0.25):
        ax.plot(list(xs), ys, 'o', markeredgewidth=0, markersize=marker_size, color=color, alpha=marker_alpha, label=label, clip_on=False)
        ax.plot(list(xs), ys, '-', linewidth=line_width, color=color, alpha=line_alpha, clip_on=False)
    
    def get_nt_fractions(pool, fixed_guide):
        return pool.non_targeting_fractions('all', fixed_guide).reindex(outcome_order, fill_value=0)

    fractions = pool.outcome_fractions(guide_status).reindex(outcome_order, fill_value=0)
    nt_fracs = get_nt_fractions(pool, ALL_NON_TARGETING)
    absolute_change = fractions.sub(nt_fracs, axis=0)
    fold_changes = pool.fold_changes(guide_status, ALL_NON_TARGETING).reindex(outcome_order, fill_value=1)
    log2_fold_changes = pool.log2_fold_changes(guide_status, ALL_NON_TARGETING).reindex(outcome_order, fill_value=0)
    
    if draw_nt_fracs == 'combined':
        for key in ('frequency', 'frequency_zoom', 'log10_frequency'):
            if key in axs:
                xs = nt_fracs

                if 'log10_frequency' in key:
                    xs = np.log10(np.maximum(xs, 1e-3))
                else:
                    xs = xs * 100

                dot_and_line(xs, axs[key], 'black', 'non-targeting', line_width=1.5, line_alpha=0.9, marker_alpha=1, marker_size=4)

    for guide_pair, color in guide_pairs:
        if any(guide in bad_guides for guide in guide_pair):
            continue

        label = str(guide_pair)
        kwargs = dict(line_alpha=0.9)

        for key in ('change', 'change_zoom'):
            if key in axs:
                dot_and_line(absolute_change[guide_pair] * 100, axs[key], color, label, **kwargs)
            
        if 'fold_change' in axs:
            dot_and_line(fold_changes[guide_pair], axs['fold_change'], color, label, **kwargs)

        if 'log2_fold_change' in axs:
            dot_and_line(log2_fold_changes[guide_pair], axs['log2_fold_change'], color, label, **kwargs)
        
        for key in ('frequency', 'frequency_zoom', 'log10_frequency'):
            if key in axs:
                xs = fractions[guide_pair]
                if 'log10' in key:
                    xs = np.log10(xs)
                else:
                    xs = xs * 100

                dot_and_line(xs, axs[key], color, label, **kwargs)

    # Apply specific visual styling to log2_between_pools panel.
    if 'log2_between_pools' in axs:
        ax = axs['log2_between_pools']
        ax.set_xlim(-3, 3)
        x_to_alpha = {
            0: 0.5,
        }
        for x in np.arange(-3, 4):
            alpha = x_to_alpha.get(x, 0.1)
            ax.axvline(x, color='black', alpha=alpha, clip_on=False)

        for side in ['left', 'right', 'bottom']:
            ax.spines[side].set_visible(False)

        ax.annotate(f'log2 ({numerator_pool.short_name} /\n {denominator_pool.short_name})',
                    xy=(0.5, 1),
                    xycoords='axes fraction',
                    xytext=(0, 30),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    )

    if log10_x_lims is None:
        log10_x_lims = (np.log10(0.0008), np.log10(0.41))
    
    for key, line_x, title, x_lims in [
        ('change', 0, 'change in percentage', (-8, 8)),
        ('change_zoom', 0, 'change in percentage\n(zoomed)', (-0.75, 0.75)),
        ('fold_change', 1, 'fold change', (0, 5)),
        ('log2_fold_change', 0, 'log2 fold change\nfrom non-targeting', (-3, 3)),
        ('frequency', None, 'percentage of\nrepair outcomes', (0, nt_fracs.max() * 100 * frequency_max_multiple)),
        ('log10_frequency', None, 'percentage of repair outcomes', log10_x_lims),
        ('frequency_zoom', None, 'percentage\n(zoomed)', (0, nt_fracs.max() * 100 * 0.3)),
       ]:

        if key not in axs:
            continue

        if line_x is not None:
            axs[key].axvline(line_x, color='black', alpha=0.3)
       
        if show_ax_titles:
            axs[key].annotate(title,
                              xy=(0.5, 1),
                              xycoords='axes fraction',
                              xytext=(0, 30),
                              textcoords='offset points',
                              ha='center',
                              va='bottom',
                              #size=14,
                             )
                    
        axs[key].xaxis.set_label_coords(0.5, 1.025)

        axs[key].set_xlim(*x_lims)
        
        # Draw lines separating categories.
        if gene_to_sort_by is None:
            for y in np.cumsum(outcome_group_sizes):
                flipped_y = len(outcome_order) - y - 0.5
                axs[key].axhline(flipped_y, color='black', alpha=0.1)
    
        # Apply specific visual styling to log10_frequency panel.
        if key == 'log10_frequency':
            ax = axs['log10_frequency']
            for exponent in [3, 2, 1]:
                xs = np.log10(np.arange(1, 10) * 10**-exponent)        
                for x in xs:
                    if x < x_lims[1]:
                        ax.axvline(x, color='black', alpha=0.1, clip_on=False)

            x_ticks = [x for x in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1] if x_lims[0] <= np.log10(x) <= x_lims[1]]
            ax.set_xticks(np.log10(x_ticks))
            ax.set_xticklabels([f'{100 * x:g}' for x in x_ticks])

            for side in ['left', 'right', 'bottom']:
                ax.spines[side].set_visible(False)

        if key == 'log2_fold_change':
            ax.grid(axis='x', alpha=0.2)
    
    for key in ['frequency_zoom', 'change_zoom', 'fold_change']:
        if key in axs:
            axs[key].set_xticklabels(list(axs[key].get_xticklabels())[:-1] + [''])

    ax_p = axs[ax_order[0]].get_position()

    if 'window' in layout_kwargs:
        window = layout_kwargs['window']
    else:
        window = 70

    if isinstance(window, tuple):
        window_start, window_end = window
    else:
        window_start, window_end = -window, window

    window_size = window_end - window_start

    diagram_width = ax_p.width * window_size / 20 * 2.7 / panel_width_multiple
    diagram_gap = diagram_width * 0.02

    diagram_ax = fig.add_axes((ax_p.x0 - diagram_width - diagram_gap, ax_p.y0, diagram_width, ax_p.height), sharey=axs[ax_order[0]])
    
    outcome_diagrams.plot(outcome_order, pool.target_info,
                          num_outcomes=len(outcome_order),
                          ax=diagram_ax,
                          **layout_kwargs)


    if show_pool_name:
        diagram_ax.annotate(pool.group,
                            xy=(0.5, 1),
                            xycoords='axes fraction',
                            xytext=(0, 25),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            size=12,
        )

        width, height = fig.get_size_inches()
        
        anchor = ax_order[-1]
        ax_p = axs[anchor].get_position()
        start_x = ax_p.x1 + 0.05 * ax_p.width
        
        heatmap_pairs = [(fg, vg) for (fg, vg), color in guide_pairs
                         if fg not in bad_guides and vg not in bad_guides and (fg, vg) != (ALL_NON_TARGETING, ALL_NON_TARGETING)
                        ]
        vals = log2_fold_changes[heatmap_pairs].reindex(outcome_order[::-1], fill_value=0)
        
        num_rows, num_cols = vals.shape
    
        heatmap_width = ax_p.height * num_cols / num_rows * height / width
        heatmap_ax = fig.add_axes((start_x, ax_p.y0, heatmap_width, ax_p.height), sharey=axs[ax_order[0]])
                
        im = heatmap_ax.imshow(vals, cmap=plt.get_cmap('RdBu_r'), vmin=v_min, vmax=v_max, origin='lower')

        heatmap_ax.xaxis.tick_top()
        heatmap_ax.set_xticks(np.arange(len(heatmap_pairs)))

        heatmap_ax.set_xticklabels([f'{fg}+{vg}' for fg, vg in heatmap_pairs], rotation=90)
        
        # Have to jump through weird hoops because label.set_text() doesn't update in-place.
        labels = []
        guide_pair_to_color = dict(guide_pairs)
        for label in heatmap_ax.get_xticklabels():
            fg, vg = label.get_text().split('+')
            color = guide_pair_to_color[fg, vg]
            label.set_color(color)
            labels.append(label)

        heatmap_ax.set_xticklabels(labels)

        for spine in heatmap_ax.spines.values():
            spine.set_visible(False)
        #    
        #if len(heatmap_pools) > 1:
        #    if heatmap_pool == heatmap_pools[-1]:
        #        gap_between = 0.4

        #        # Draw vertical line separating different genes.
        #        if gene != genes_to_heatmap[-1]:
        #            line_fig_x = start_x + heatmap_width * (1 + (3 / num_cols) * gap_between * 0.5)
        #            heatmap_ax.plot([line_fig_x, line_fig_x], [ax_p.y0 - ax_p.height * 0.05, ax_p.y1 + ax_p.height * 0.05],
        #                            clip_on=False,
        #                            transform=fig.transFigure,
        #                            color='black',
        #                            linewidth=2,
        #                            alpha=0.5,
        #                        )

        #    else:
        #        gap_between = 0.1

        #    # Label each pool with an identifying color stripe.
        #    stripe_y = -1 / len(outcome_order)
        #    heatmap_ax.plot([0, 1], [stripe_y, stripe_y],
        #                    clip_on=False,
        #                    transform=heatmap_ax.transAxes,
        #                    linewidth=7,
        #                    solid_capstyle='butt',
        #                    color=pool_to_color[heatmap_pool.group, heatmap_fixed_guide],
        #                    )

        #    if gene == genes_to_heatmap[0]:
        #        # Draw a legend of pool names.
        #        #heatmap_ax.annotate(heatmap_pool.short_name,
        #        heatmap_ax.annotate(heatmap_fixed_guide,
        #                            xy=(0.5, stripe_y),
        #                            xycoords='axes fraction',
        #                            xytext=(0, -15 * (heatmap_i + 1)),
        #                            textcoords='offset points',
        #                            ha='center',
        #                            va='top',
        #                            annotation_clip=False,
        #                            color=pool_to_color[heatmap_pool.group, heatmap_fixed_guide],
        #                            #arrowprops={'arrowstyle': '-',
        #                            #            'alpha': 0.5,
        #                            #            'color': pool_to_color[heatmap_pool],
        #                            #            'linewidth': 2,
        #                            #            },
        #                            )

        #else:
        #    gap_between = 0.1

        #start_x += heatmap_width * (1 + (3 / num_cols) * gap_between)

        #heatmap_axs[gene] = heatmap_ax

        #last_gene = gene

    return fig, outcome_order

def big_heatmap(pool_list,
                genes=None,
                guides=None,
                outcomes_list=None,
                windows=40,
                cluster_guides=False,
                cluster_outcomes=False,
                title=None,
                layout_kwargs=None,
               ):
    if layout_kwargs is None:
        layout_kwargs = dict(draw_all_sequence=False)

    def expand_window(window):
        if isinstance(window, int):
            window_start, window_stop = -window, window
        else:
            window_start, window_stop = window

        window_size = window_stop - window_start

        return window_start, window_stop, window_size

    if isinstance(windows, (int, tuple)):
        windows = itertools.repeat(windows)

    pool = pool_list[0]
    rows = len(outcomes_list[0])

    if guides is None:
        if genes is None:
            genes = pool.variable_guide_library.genes
            guides = pool.variable_guide_library.guides
        else:
            guides = []
            for gene in genes:
                guides.extend(pool.gene_guides(gene))

    max_cols = None
    
    inches_per_col = 0.15
    cols_for_percentage = 6
    width = inches_per_col * cols_for_percentage
    height = width * rows / cols_for_percentage * len(pool_list)

    fig, percentage_axs = plt.subplots(len(pool_list), 1,
                                       figsize=(width, height),
                                       gridspec_kw=dict(left=0, right=1, bottom=0, top=1),
                                      )

    if not isinstance(percentage_axs, np.ndarray):
        percentage_axs = [percentage_axs]

    for pool, p_ax, window, outcome_order in zip(pool_list, percentage_axs, windows, outcomes_list):
        window_start, window_stop, window_size = expand_window(window)
        
        fold_changes = pool.log2_fold_changes('perfect', 'none')['none'].loc[outcome_order, guides]

        if cluster_guides:
            guide_correlations = fold_changes.corr()

            linkage = scipy.cluster.hierarchy.linkage(guide_correlations)
            dendro = scipy.cluster.hierarchy.dendrogram(linkage,
                                                        no_plot=True,
                                                        labels=guide_correlations.index, 
                                                        )

            guide_order = dendro['ivl']
        else:
            guide_order = guides

        if cluster_outcomes:
            outcome_correlations = fold_changes.T.corr()

            linkage = scipy.cluster.hierarchy.linkage(outcome_correlations)
            dendro = scipy.cluster.hierarchy.dendrogram(linkage,
                                                        no_plot=True,
                                                        labels=outcome_correlations.index, 
                                                        )

            outcome_order = dendro['ivl']

        colors = pool.log2_fold_changes('perfect', 'none')['none'].loc[outcome_order, guide_order]
        to_plot = colors.iloc[:, :max_cols]
        rows, cols = to_plot.shape

        nt_fracs = pool.non_targeting_fractions('all', 'none')[outcome_order]
        xs = list(nt_fracs * 100)
        ys = np.arange(len(outcome_order))[::-1]

        p_ax.plot(list(xs), ys, 'o', markersize=3, color='grey', alpha=1)
        p_ax.plot(list(xs), ys, '-', linewidth=1.5, color='grey', alpha=0.9)
        p_ax.set_xlim(0)
        p_ax.xaxis.tick_top()
        p_ax.annotate('percentage',
                            xy=(0.5, 1),
                            xycoords='axes fraction',
                            xytext=(0, 18),
                            textcoords='offset points',
                            size=8,
                            ha='center',
                            va='bottom',
                           )

        p_ax.spines['left'].set_alpha(0.3)
        p_ax.spines['right'].set_alpha(0.3)
        p_ax.tick_params(labelsize=6)
        p_ax.grid(axis='x', alpha=0.3)
        
        p_ax.spines['bottom'].set_visible(False)

        plt.draw()
        fig_width, fig_height = fig.get_size_inches()
        percentage_p = p_ax.get_position()
        p_height_inches = percentage_p.height * fig_height
        heatmap_height_inches = p_height_inches
        heatmap_width_inches = p_height_inches * cols / rows
        heatmap_width = heatmap_width_inches / fig_width
        heatmap_height = heatmap_height_inches / fig_height
        heatmap_gap = percentage_p.width * 1 / cols_for_percentage
        heatmap_ax = fig.add_axes((percentage_p.x1 + heatmap_gap, percentage_p.y0, heatmap_width, heatmap_height), sharey=p_ax)

        im = heatmap_ax.imshow(to_plot[::-1], cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2, origin='lower')

        heatmap_ax.set_yticks([])

        heatmap_ax.set_xticks(np.arange(cols))
        heatmap_ax.set_xticklabels(guide_order, rotation=90, size=6)
        heatmap_ax.xaxis.set_tick_params(labeltop=True)

        #if cluster_guides:
        if False:
            heatmap_ax.xaxis.set_ticks_position('both')
        else:
            heatmap_ax.xaxis.set_ticks_position('top')
            heatmap_ax.xaxis.set_tick_params(labelbottom=False)
            
        for spine in heatmap_ax.spines.values():
            spine.set_visible(False)

        heatmap_p = heatmap_ax.get_position()

        diagram_width = heatmap_p.width * (0.75 * window_size / cols)
        diagram_gap = heatmap_gap

        diagram_ax = fig.add_axes((percentage_p.x0 - diagram_width - diagram_gap, heatmap_p.y0, diagram_width, heatmap_p.height), sharey=heatmap_ax)

        _ = outcome_diagrams.plot(outcome_order, pool.target_info,
                                  num_outcomes=len(outcome_order),
                                  ax=diagram_ax,
                                  window=(window_start, window_stop),
                                  **layout_kwargs)
        
        diagram_ax.annotate(pool.group,
                            xy=(0.5, 1),
                            xycoords='axes fraction',
                            xytext=(0, 28),
                            textcoords='offset points',
                            size=10,
                            ha='center',
                            va='bottom',
                           )

    if title is not None:
        ax = percentage_axs[0]
        ax.annotate(title,
                    xy=(0.5, 1),
                    xycoords='axes fraction',
                    xytext=(0, 70),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=26,
                )

    #cbar_ax = fig.add_axes((heatmap_p.x0, heatmap_p.y1 + heatmap_p.height * (3 / rows), heatmap_p.width * (20 / cols), heatmap_p.height * (1 / rows)))

    #cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

    #cbar.set_ticks([-2, -1, 0, 1, 2])
    #cbar.set_ticklabels(['$\leq$-2', '-1', '0', '1', '$\geq$2'])
    #cbar_ax.xaxis.tick_top()

    #cbar_ax.set_title('log2 fold change from non-targeting guides', y=3) 

    #cbar_ax.annotate('gene activity\npromotes outcome',
    #                    xy=(0, 0),
    #                    xycoords='axes fraction',
    #                    xytext=(0, -5),
    #                    textcoords='offset points',
    #                    ha='center',
    #                    va='top',
    #                    size=8,
    #                )

    #cbar_ax.annotate('gene activity\nsuppresses outcome',
    #                    xy=(1, 0),
    #                    xycoords='axes fraction',
    #                    xytext=(0, -5),
    #                    textcoords='offset points',
    #                    ha='center',
    #                    va='top',
    #                    size=8,
    #                )

    return fig

def doubles_heatmap(pool, outcome, subtract_nt=False):
    if isinstance(outcome, int):
        outcomes = pool.most_frequent_outcomes(None)[:100]
        outcome = outcomes[outcome]

    fcs = np.array([pool.log2_fold_changes('perfect', fg)[fg].loc[outcome].drop('all_non_targeting') for fg in pool.fixed_guides])

    num_rows, num_cols = fcs.shape

    if subtract_nt:
        fcs = fcs - fcs[-1]

    heatmap_kwargs = dict(cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2)
    fig, main_ax = plt.subplots(figsize=(60, 10))
    main_ax.imshow(fcs, **heatmap_kwargs)

    main_ax.xaxis.tick_top()
    main_ax.set_xticks(np.arange(num_cols))
    main_ax.set_xticklabels(pool.variable_guides, rotation=90, size=6)
    #main_ax.set_xticklabels([])

    main_ax.set_yticks([])
    main_ax.set_yticklabels([])

    for g in pool.fixed_guide_library.genes:
        if g == 'negative_control':
            continue
        x_start, x_end = pool.variable_guide_library.gene_indices(g)
        y_start, y_end = pool.fixed_guide_library.gene_indices(g)

        xs = [x_start - 0.5, x_end + 0.5, x_end + 0.5, x_start - 0.5, x_start - 0.5]
        ys = [y_start - 0.5, y_start - 0.5, y_end + 0.5, y_end + 0.5, y_start - 0.5]

        main_ax.plot(xs, ys, color='black', clip_on=False)

    # Get the baseline log2 fold changes for each fixed guides non-targeting variable guides relative to the non-targeting fixed guide.
    fg_fcs = pool.log2_fold_changes('perfect', None).loc[outcome, idx[:, 'all_non_targeting']].drop('all_non_targeting', level=0)
    fg_fcs = np.expand_dims(fg_fcs.values, 1)

    ax_p = main_ax.get_position()
    width = ax_p.width / num_cols
    height = ax_p.height
    left_ax = fig.add_axes((ax_p.x0 - width * 2, ax_p.y0, width, height))

    left_ax.imshow(fg_fcs, **heatmap_kwargs)

    left_ax.xaxis.tick_top()
    left_ax.set_xticks([0])
    left_ax.set_xticklabels(['effect of fixed guide alone'], rotation=90, size=6)

    left_ax.set_yticks(np.arange(num_rows))
    left_ax.set_yticklabels(pool.fixed_guides, size=6)

    for ax in [main_ax, left_ax]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    left_ax.set_ylabel('fixed guide')
    main_ax.set_xlabel('additional effect of variable guide')
    main_ax.xaxis.set_label_position('top')

    diagram_ax = fig.add_axes((ax_p.x0 - 0.12, ax_p.y0 + ax_p.height * 0.4, ax_p.width * 0.1, ax_p.height * 0.15))
    outcome_diagrams.plot([outcome], pool.target_info, draw_wild_type_on_top=True, window=20, ax=diagram_ax)

    return fig, fcs

def arrayed_group(group,
                  outcomes=None,
                  condition_to_sort_by=None,
                  condition_order=None,
                  xlims=None,
                  panels_to_show=['freq', 'log10_freq', 'diff', 'log2_fc', 'heatmaps'],
                  title=None,
                  **diagram_kwargs,
                 ):
    diagram_kwargs.setdefault('window', (-45, 45))

    if title is None:
        title = group.group

    if xlims is None:
        xlims = {}

    xlims.setdefault('freq', (0, 100))
    xlims.setdefault('log10_freq', (np.log10(1e-4), np.log10(50e-2)))

    if outcomes is None:
        outcomes = 150
    
    if isinstance(outcomes, int):
        outcome_order = group.outcomes_by_baseline_frequency[:outcomes]
    else:
        outcome_order = outcomes

    if condition_to_sort_by is not None:
        outcome_order = group.log2_fold_change_condition_means.loc[outcome_order, condition_to_sort_by].sort_values(ascending=False).index.values
        
    grid = repair_seq.visualize.outcome_diagrams.DiagramGrid(outcome_order, group.target_info, title=title, **diagram_kwargs)

    if 'freq' in panels_to_show:
        grid.add_ax('freq', width_multiple=10, gap_multiple=2, title='percentage of reads')

    if 'log10_freq' in panels_to_show:
        grid.add_ax('log10_freq', width_multiple=10, gap_multiple=2, title='percentage of reads\n(log scale)')

    if 'diff' in panels_to_show:
        grid.add_ax('diff', width_multiple=10, gap_multiple=2, title=f'percentage change\nfrom baseline')

    if 'log2_fc' in panels_to_show:
        grid.add_ax('log2_fc', width_multiple=10, gap_multiple=2, title=f'log2 fold-change\nfrom baseline')

    if condition_order is None:
        condition_order = group.conditions

    for i, condition in enumerate(condition_order):
        means = group.outcome_fraction_condition_means[condition]
        stds = group.outcome_fraction_condition_stds[condition]
        
        interval_sources = {
            'lower': means - stds,
            'upper': means + stds,
        }

        color = f'C{i}'
        common_kwargs = dict(marker='o', color=color, markersize=2, interval_alpha=0.5)

        grid.plot_on_ax('freq', means, interval_sources=interval_sources, transform=lambda s: s * 100, **common_kwargs)
        grid.plot_on_ax('log10_freq', means, interval_sources=interval_sources, transform=np.log10, **common_kwargs)
        
        means = group.outcome_fraction_difference_condition_means[condition].reindex(outcome_order)
        stds = group.outcome_fraction_difference_condition_stds[condition].reindex(outcome_order)
        
        interval_sources = {
            'lower': means - stds,
            'upper': means + stds,
        }
        
        grid.plot_on_ax('diff', means, interval_sources=interval_sources, transform=lambda s: s * 100, **common_kwargs)
        
        means = group.log2_fold_change_condition_means[condition].reindex(outcome_order)
        stds = group.log2_fold_change_condition_stds[condition].reindex(outcome_order)
        
        interval_sources = {
            'lower': means - stds,
            'upper': means + stds,
        }
        
        grid.plot_on_ax('log2_fc', means, interval_sources=interval_sources, **common_kwargs)
        
        if 'heatmaps' in panels_to_show:
            if len(group.condition_keys) == 1:
                level = group.condition_keys[0]
            else:
                level = group.condition_keys
            # weird reversing of outcome_order
            vs = group.log2_fold_changes.xs(condition, axis=1, level=level, drop_level=False).reindex(outcome_order[::-1])
            grid.add_heatmap(vs, str(condition), gap_multiple=(2 if i == 0 else 0.5), color=color)
        
    #grid.axs_by_name['freq'].legend(bbox_to_anchor=(0.5, 1), loc='lower center')

    for ax_name in xlims:
        if ax_name in panels_to_show:
            grid.axs_by_name[ax_name].set_xlim(*xlims[ax_name])

    if 'log10_freq' in panels_to_show:
        grid.style_log10_frequency_ax('log10_freq')
    if 'log2_fc' in panels_to_show:
        grid.style_fold_change_ax('log2_fc')

    return grid

def arrayed_group_categories(group,
                             condition_to_sort_by=None,
                             condition_order=None,
                             draw_heatmaps=False,
                             manual_outcome_order=None,
                             panels_to_show=['freq', 'log10_freq', 'diff', 'log2_fc', 'heatmaps'],
                             vmin=-2,
                             vmax=2,
                             freq_xlims=(0, 100),
                             log10_freq_xlims=(0, 100),
                             log2_fc_xlims=None,
                             manual_colors=None,
                             label_aliases=None,
                             grid=None,
                             plot_kwargs=None,
                             inches_per_outcome=0.3,
                            ):
    outcome_order = group.categories_by_baseline_frequency
    if manual_outcome_order is not None:
        outcome_order = manual_outcome_order

    if condition_to_sort_by is not None:
        outcome_order = group.category_log2_fold_change_condition_means.loc[outcome_order, condition_to_sort_by].sort_values().index.values

    if manual_colors is None:
        manual_colors = {}

    if plot_kwargs is None:
        plot_kwargs = {}
    
    if grid is None:
        grid_requires_styling = True
        grid = repair_seq.visualize.outcome_diagrams.DiagramGrid(outcome_order,
                                                        group.target_info,
                                                        title=group.group,
                                                        window=(-10, 10),
                                                        draw_all_sequence=False,
                                                        label_aliases=label_aliases,
                                                        inches_per_outcome=inches_per_outcome,
                                                        label_size=6,
                                                       )

        if 'freq' in panels_to_show:
            grid.add_ax('freq', width_multiple=7, gap_multiple=1, title='percentage of reads\n(linear scale)', title_size=8)
        if 'log10_freq' in panels_to_show:
            grid.add_ax('log10_freq', width_multiple=7, gap_multiple=1, title='percentage of reads\n(log scale)', title_size=8)
        if 'diff' in panels_to_show:
            grid.add_ax('diff', width_multiple=7, gap_multiple=1, title=f'percentage change\nfrom baseline', title_size=8)
        if 'log2_fc' in panels_to_show:
            grid.add_ax('log2_fc', width_multiple=7, gap_multiple=1, title=f'log$_2$ fold change\nfrom baseline', title_size=8)
    else:
        grid_requires_styling = False

    if condition_order is None:
        condition_order = group.conditions

    percentage = lambda s: s * 100

    for i, condition in enumerate(condition_order):
        means = group.category_fraction_condition_means[condition]
        stds = group.category_fraction_condition_stds[condition]
        
        interval_sources = {
            'lower': means - stds,
            'upper': means + stds,
        }

        if condition in manual_colors:
            color = manual_colors[condition]
        elif group.Batch.condition_colors is not None:
            color = group.Batch.condition_colors.loc[condition]
        else:
            color = f'C{i}'

        common_kwargs = dict(marker='o', color=color, markersize=3, y_offset=0.05 * i)
        common_kwargs.update(**plot_kwargs)
        grid.plot_on_ax('freq', means, interval_sources=interval_sources, transform=percentage, **common_kwargs)
        grid.plot_on_ax('log10_freq', means, interval_sources=interval_sources, transform=np.log10, **common_kwargs)
        
        means = group.category_fraction_difference_condition_means[condition].loc[outcome_order]
        stds = group.category_fraction_difference_condition_stds[condition].loc[outcome_order]
        
        interval_sources = {
            'lower': means - stds,
            'upper': means + stds,
        }
        
        grid.plot_on_ax('diff', means, interval_sources=interval_sources, transform=percentage, **common_kwargs)
        
        means = group.category_log2_fold_change_condition_means[condition].loc[outcome_order]
        stds = group.category_log2_fold_change_condition_stds

        interval_sources = {
            'lower': stds['lower'][condition].loc[outcome_order],
            'upper': stds['upper'][condition].loc[outcome_order],
        }
        
        grid.plot_on_ax('log2_fc', means, interval_sources=interval_sources, **common_kwargs)
        
        if draw_heatmaps and condition != group.baseline_condition:
            # weird reversing of outcome_order
            if len(group.condition_keys) == 1:
                level = group.condition_keys[0]
            else:
                level = group.condition_keys

            vs = group.category_log2_fold_changes.xs(condition, axis=1, level=level, drop_level=False).loc[outcome_order[::-1]]
            grid.add_heatmap(vs, str(condition), gap_multiple=(1 if i == 0 else 0.5), color=color, vmin=vmin, vmax=vmax)

    if draw_heatmaps:
        grid.add_colorbar(baseline_condition_name=group.baseline_condition)
        
    #grid.axs_by_name['freq'].legend(bbox_to_anchor=(0.5, 1), loc='lower center')

    if grid_requires_styling:
        grid.set_xlim('freq', freq_xlims)
        grid.set_xlim('log10_freq', log10_freq_xlims)
        grid.set_xlim('log2_fc', log2_fc_xlims)
        grid.set_xlim('log10_freq', log10_freq_xlims)
        grid.style_log10_frequency_ax('log10_freq', label_size=6)
        grid.style_fold_change_ax('log2_fc', label_size=6)

    return grid

def pooled_screen_categories(pool,
                             genes,
                             genes_to_plot=None,
                             group_genes=False,
                             manual_outcome_order=None,
                             top_n_outcomes=None,
                             draw_heatmaps=False,
                             subcategories=False,
                             outcomes_to_drop=None,
                             freq_xlims=(0, 100),
                             log10_freq_xlims=(np.log10(5e-4), np.log10(0.85)),
                             log2_fc_xlims=None,
                             vmin=-2,
                             vmax=2,
                             panels_to_show=['freq', 'log10_freq', 'diff', 'log2_fc', 'heatmaps'],
                             label_aliases=None,
                             manual_colors=None,
                             title=None,
                             **kwargs,
                            ):

    if outcomes_to_drop is None:
        outcomes_to_drop = []

    if genes_to_plot is None:
        genes_to_plot = genes

    if manual_colors is None:
        manual_colors = {}

    if title is None:
        title = pool.short_name

    if isinstance(pool, repair_seq.pooled_screen.PooledScreen):
        if subcategories:
            outcome_order = pool.subcategories_by_baseline_frequency
            fractions = pool.subcategory_fractions
            fraction_differences = pool.subcategory_fraction_differences
            log2_fold_changes = pool.subcategory_log2_fold_changes
        else:
            outcome_order = pool.categories_by_baseline_frequency
            fractions = pool.category_fractions
            fraction_differences = pool.category_fraction_differences
            log2_fold_changes = pool.category_log2_fold_changes
    else:
        outcome_order = pool.categories_by_baseline_frequency
        fractions = pool.category_fraction_means
        fraction_differences = pool.category_fraction_difference_means
        log2_fold_changes = pool.category_log2_fold_change_means

    if manual_outcome_order is not None:
        outcome_order = manual_outcome_order

    outcome_order = outcome_order[:top_n_outcomes]

    outcome_order = [outcome for outcome in outcome_order if outcome not in outcomes_to_drop]

    fractions = fractions.loc[outcome_order]
    fraction_differences = fraction_differences.loc[outcome_order]
    log2_fold_changes = log2_fold_changes.loc[outcome_order]

    grid = outcome_diagrams.DiagramGrid(outcome_order,
                                        pool.target_info,
                                        title=title,
                                        window=(-10, 10),
                                        draw_all_sequence=False,
                                        label_aliases=label_aliases,
                                        inches_per_outcome=kwargs.get('inches_per_outcome', 0.3),
                                        outcome_ax_width=1,
                                        label_size=kwargs.get('label_size', 8),
                                       )

    ax_kwargs = dict(
        width_multiple=5,
        gap_multiple=1,
        title_size=8,
    )

    if 'freq' in panels_to_show:
        grid.add_ax('freq', title='Percentage of reads\n(linear scale)', **ax_kwargs)
    if 'log10_freq' in panels_to_show:
        grid.add_ax('log10_freq', title='Percentage of reads\n(log scale)', **ax_kwargs)
    if 'diff' in panels_to_show:
        grid.add_ax('diff', title=f'Percentage change\nfrom non-targeting', **ax_kwargs)
    if 'log2_fc' in panels_to_show:
        grid.add_ax('log2_fc', title=f'Log$_2$ fold change\nfrom non-targeting', **ax_kwargs)

    nt_guides = pool.variable_guide_library.non_targeting_guides
    nt_freqs = fractions[ALL_NON_TARGETING]
    nt_stds = fractions[nt_guides].std(axis=1)

    interval_sources = {
        'lower': nt_freqs - nt_stds,
        'upper': nt_freqs + nt_stds,
    }
    
    common_kwargs = dict(
        marker='o',
        color=manual_colors.get('negative_control', 'black'),
        markersize=2,
        linewidth=1.5,
        line_alpha=0.6,
        marker_alpha=0.8,
        clip_on=False,
        value_source=nt_freqs,
        interval_sources=interval_sources,
        zorder=6,
        interval_alpha=0.3,
    )

    grid.plot_on_ax('freq', transform='percentage', **common_kwargs)
    grid.plot_on_ax('log10_freq', transform='log10', **common_kwargs)

    #l2fcs = log2_fold_changes[nt_guides].mean(axis=1)
    #stds = log2_fold_changes[nt_guides].std(axis=1)
    #interval_sources = {
    #    'lower': l2fcs - stds,
    #    'upper': l2fcs + stds,
    #}

    if 'log2_fc' in panels_to_show:
        #common_kwargs.update(value_source=l2fcs, interval_sources=interval_sources)
        #grid.plot_on_ax('log2_fc', **common_kwargs)
        grid.axs_by_name['log2_fc'].axvline(0, color='black', linewidth=1.5)

    freq_labels = [('sgRNAs:', 'black')]

    for i, gene in enumerate(genes):
        guide_order = pool.variable_guide_library.gene_guides(gene, only_best_promoter=True)

        color = manual_colors.get(gene, f'C{i}')

        common_kwargs = dict(
            marker='o',
            color=color,
            markersize=2,
            marker_alpha=0.8,
            #y_offset=0.05 * i,
            linewidth=1,
            line_alpha=0.4,
            clip_on=False,
            zorder=5,
            interval_alpha=0.3,
        )

        if gene in genes_to_plot:
            freq_labels.append((gene, color))

            if group_genes:
                freqs = fractions[guide_order].mean(axis=1)
                stds = fractions[guide_order].std(axis=1)
                interval_sources = {
                    'lower': freqs - stds,
                    'upper': freqs + stds,
                }
                grid.plot_on_ax('freq', freqs, interval_sources=interval_sources, transform='percentage', **common_kwargs)
                grid.plot_on_ax('log10_freq', freqs, interval_sources=interval_sources, transform='log10', **common_kwargs)

                l2fcs = log2_fold_changes[guide_order].mean(axis=1)
                stds = log2_fold_changes[guide_order].std(axis=1)
                interval_sources = {
                    'lower': l2fcs - stds,
                    'upper': l2fcs + stds,
                }
                grid.plot_on_ax('log2_fc', l2fcs, interval_sources=interval_sources, **common_kwargs)

            else:
                for guide in guide_order:
                    freqs = fractions[guide]

                    grid.plot_on_ax('freq', freqs, transform='percentage', **common_kwargs)
                    grid.plot_on_ax('log10_freq', freqs, transform='log10', **common_kwargs)
                
                    diffs = fraction_differences[guide]
                    
                    grid.plot_on_ax('diff', diffs, transform='percentage', **common_kwargs)
                    
                    l2fcs = log2_fold_changes[guide]
                    grid.plot_on_ax('log2_fc', l2fcs, **common_kwargs)
        
        if draw_heatmaps:
            # weird reversing of outcome_order
            vs = log2_fold_changes[guide_order].iloc[::-1]

            grid.add_heatmap(vs, 'gene', gap_multiple=(1 if i == 0 else 0.5), color=color, vmin=vmin, vmax=vmax)

    freq_labels.append(('non-targeting', manual_colors.get('negative_control', 'black')))

    if draw_heatmaps:
        grid.add_colorbar()
        
    grid.set_xlim('freq', freq_xlims)
    grid.axs_by_name['log10_freq'].set_xlim(*log10_freq_xlims)

    grid.set_xlim('log2_fc', log2_fc_xlims)

    grid.style_log10_frequency_ax('log10_freq')
    grid.style_fold_change_ax('log2_fc')
    grid.style_fold_change_ax('diff')

    for ax_name in ['freq', 'log10_freq', 'log2_fc']:
        if ax_name in panels_to_show:
            plt.setp(grid.axs_by_name[ax_name].get_xticklabels(), size=6)

    ax_to_label = kwargs.get('ax_to_label', 'freq')
    if ax_to_label == 'outside':
        ax = grid.ordered_axs[1]
    else:
        ax_to_label = 'freq'

        if ax_to_label not in grid.axs_by_name:
            ax_to_label = 'log10_freq'

        if ax_to_label in grid.axs_by_name:
            ax = grid.axs_by_name[ax_to_label]
        else:
            raise ValueError(ax_to_label)

    if ax_to_label == 'outside':
        for i, (label, color) in enumerate(freq_labels):
            ax.annotate(label,
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(5, -3 - 10 * i),
                        textcoords='offset points',
                        ha='right',
                        va='top',
                        color=color,
                        size=kwargs.get('guide_label_size', 8),
            )

    elif ax_to_label == 'freq':
        for i, (label, color) in enumerate(freq_labels[::-1]):
            ax.annotate(label,
                        xy=(0.73, 0),
                        xycoords='axes fraction',
                        xytext=(-3, 8.5 * i),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        color=color,
                        size=kwargs.get('guide_label_size', 8),
            )

    else:
        for i, (label, color) in enumerate(freq_labels):
            ax.annotate(label,
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(-3, -3 - 8 * i),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        color=color,
                        size=kwargs.get('guide_label_size', 8),
            )

    return grid
