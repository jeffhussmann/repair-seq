from itertools import starmap

import matplotlib.pyplot as plt
import matplotlib.gridspec
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import h5py
import bokeh.palettes
from matplotlib.patches import ConnectionPatch

from collections import Counter, defaultdict
from knockin.target_info import degenerate_indel_from_string, SNVs
from knockin import quantiles as quantiles_module
from sequencing import Visualize, utilities

def plot_outcome_diagrams(outcome_order, target_info, num_outcomes=30, title=None, window=70, ax=None, flip_if_reverse=True):
    outcome_order = outcome_order[:num_outcomes]
    num_outcomes = len(outcome_order)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.8 * 20, 0.8 * num_outcomes / 3))
    else:
        fig = ax.figure

    if isinstance(window, int):
        window_left, window_right = -window, window
    else:
        window_left, window_right = window

    ax.plot([0.5, 0.5], [-0.5, num_outcomes - 0.5], color='black', linestyle='--', alpha=0.3)
    
    offset = target_info.cut_after
    guide = target_info.features[target_info.target, target_info.sgRNA]
    if flip_if_reverse and guide.strand == '-':
        flip = True
        transform_seq = utilities.complement
    else:
        flip = False
        transform_seq = utilities.identity
    
    if flip:
        window_left, window_right = -window_right, -window_left

    seq = target_info.target_sequence[offset + window_left:offset + window_right + 1]

    def draw_rect(x0, x1, y0, y1, alpha, color='black'):
        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]

        patch = plt.Polygon(path,
                            fill=True,
                            closed=True,
                            alpha=alpha,
                            color=color,
                            clip_on=True,
                            linewidth=0,
                           )
        ax.add_patch(patch)

    text_size = 8
    block_alpha = 0.1
    wt_height = 0.6

    def draw_sequence(y, xs_to_skip=None, alpha=0.15):
        if xs_to_skip is None:
            xs_to_skip = set()

        for x, b in zip(range(window_left, window_right + 1), seq):
            if x not in xs_to_skip:
                ax.annotate(transform_seq(b),
                            xy=(x, y),
                            xycoords='data', 
                            ha='center',
                            va='center',
                            size=text_size,
                            alpha=alpha,
                           )

    def draw_deletion(y, deletion, color='black'):
        starts = np.array(deletion.starts_ats) - offset
        if len(starts) > 1:
            for x, b in zip(range(window_left, window_right + 1), seq):
                if (starts[0] <= x < starts[-1]) or (starts[0] + deletion.length <= x < starts[-1] + deletion.length):
                    ax.annotate(transform_seq(b),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                               )

        del_height = 0.15
        
        del_start = starts[0] - 0.5
        del_end = starts[0] + deletion.length - 1 + 0.5
        
        draw_rect(del_start, del_end, y - del_height / 2, y + del_height / 2, 0.4, color=color)
        draw_rect(window_left - 0.5, del_start, y - wt_height / 2, y + wt_height / 2, block_alpha)
        draw_rect(del_end, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)


    guide_start = guide.start - 0.5 - offset
    guide_end = guide.end + 0.5 - offset
    
    for i, (category, subcategory, details) in enumerate(outcome_order):
        y = num_outcomes - i - 1
            
        if category == 'deletion':
            deletion = degenerate_indel_from_string(details)
            draw_deletion(y, deletion)
            draw_sequence(y)
        
        elif category == 'insertion':
            insertion = degenerate_indel_from_string(details)
            starts = np.array(insertion.starts_afters) - offset
            draw_rect(window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
            for i, (start, bs) in enumerate(zip(starts, insertion.seqs)):
                ys = [y - 0.3, y + 0.3]
                xs = [start + 0.5, start + 0.5]

                ax.plot(xs, ys, color='purple', linewidth=1.5, alpha=0.6)
                
                if i == 0:
                    width = 0.9
                    center = start + 0.5
                    left_edge = center - (len(bs) * 0.5 * width)
                    for x_offset, b in enumerate(bs):
                        ax.annotate(transform_seq(b),
                                    xy=(left_edge + (x_offset * width) + width / 2, y + (wt_height / 2)),
                                    xycoords='data',
                                    ha='center',
                                    va='center',
                                    size=text_size * 1,
                                    color=Visualize.igv_colors[b],
                                    weight='bold',
                                )
            
            draw_sequence(y)
                
        elif category == 'wild type':
            if subcategory == 'wild type':
                guide_start = guide.start - 0.5 - offset
                guide_end = guide.end + 0.5 - offset

                PAM_start = target_info.PAM_slice.start - 0.5 - offset
                PAM_end = target_info.PAM_slice.stop + 0.5 - 1 - offset

                draw_rect(guide_start, guide_end, y - wt_height / 2, y + wt_height / 2, 0.3, color='blue')
        
                draw_rect(window_left - 0.5, min(PAM_start, guide_start), y - wt_height / 2, y + wt_height / 2, block_alpha)
                draw_rect(max(PAM_end, guide_end), window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

                draw_rect(PAM_start, PAM_end, y - wt_height / 2, y + wt_height / 2, 0.3, color='green')

                draw_sequence(y, alpha=1)

            else:
                SNV_xs = set()
                draw_rect(window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
                snvs = SNVs.from_string(details) 
                for snv in snvs:
                    x = snv.position - offset
                    SNV_xs.add(x)
                    if window_left <= x <= window_right:
                        ax.annotate(transform_seq(snv.basecall),
                                    xy=(x, y),
                                    xycoords='data', 
                                    ha='center',
                                    va='center',
                                    size=text_size,
                                    color=Visualize.igv_colors[snv.basecall.upper()],
                                    weight='bold',
                                )
                draw_sequence(y, xs_to_skip=SNV_xs)

        elif category == 'donor':
            SNP_xs = set()
            variable_locii_details, deletion_details = details.split(';', 1)

            for ((strand, position), ref_base), read_base in zip(target_info.fingerprints[target_info.target], variable_locii_details):
                if ref_base != read_base and read_base != '_':
                    x = position - offset
                    SNP_xs.add(x)
                    ax.annotate(transform_seq(read_base),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=Visualize.igv_colors[read_base],
                                weight='bold',
                               )
            
                draw_rect(position - offset - 0.5, position - offset + 0.5, y - wt_height / 2, y + wt_height / 2, 0.3, color='grey')

            deletion_strings = deletion_details.split(';')
            if len(deletion_strings) > 1:
                raise NotImplementedError
            elif len(deletion_strings) == 1:
                deletion_string = deletion_strings[0]
                if deletion_string == '':
                    # no deletion
                    draw_rect(window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
                else:
                    deletion = degenerate_indel_from_string(deletion_string)
                    draw_deletion(y, deletion, color='red')

            draw_sequence(y, xs_to_skip=SNP_xs)

        else:
            if category == 'uncategorized':
                label = 'uncategorized'
            else:
                label = '{}, {}'.format(category, subcategory)
            ax.annotate(label,
                        xy=(0, y),
                        xycoords='data', 
                        ha='center',
                        va='center',
                        size=text_size,
                        )
                
    x_lims = [window_left - 0.5, window_right + 0.5]
    if flip:
        ax.set_xlim(*x_lims[::-1])
    else:
        ax.set_xlim(*x_lims)

    ax.set_ylim(-0.5, num_outcomes - 0.5)
    ax.set_frame_on(False)

    if title:
        ax.annotate(title,
                    xy=(0, 1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 22),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=14,
                   )
        
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.axhline(num_outcomes + 0.5 - 1, color='black', alpha=0.75, clip_on=False)
    
    return fig

def add_frequencies(fig, ax, pool, outcome_order, text_only=False):
    ax_p = ax.get_position()
    
    width = ax_p.width * 0.1
    offset = ax_p.width * 0.02

    freqs = pool.non_targeting_fractions('perfect').loc[outcome_order]
    counts = pool.non_targeting_counts('perfect').loc[outcome_order]

    ys = np.arange(len(outcome_order) - 1, -1, -1)
    
    for y, freq, count in zip(ys, freqs, counts):
        ax.annotate('{:> 5.2%} {:>8s}'.format(freq, '({:,})'.format(count)),
                    xy=(1, y),
                    xycoords=('axes fraction', 'data'),
                    xytext=(6, 0),
                    textcoords=('offset points'),
                    family='monospace',
                    ha='left',
                    va='center',
                   )

    if not text_only:
        freq_ax = fig.add_axes((ax_p.x1 + 6 * offset, ax_p.y0, width, ax_p.height), sharey=ax)
        freq_ax_p = freq_ax.get_position()
        log_ax = fig.add_axes((freq_ax_p.x1 + offset, ax_p.y0, width, ax_p.height), sharey=ax)
        log_ax_p = log_ax.get_position()
        cumulative_ax = fig.add_axes((log_ax_p.x1 + offset, ax_p.y0, width, ax_p.height), sharey=ax)
        
        freq_ax.plot(freqs, ys, 'o-', markersize=2, color='black')
        log_ax.plot(np.log10(freqs), ys, 'o-', markersize=2, color='black')
        cumulative_ax.plot(freqs.cumsum(), ys, 'o-', markersize=2, color='black')
        
        freq_ax.set_xlim(0, max(freqs) * 1.05)
        cumulative_ax.set_xlim(0, 1)
        cumulative_ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        
        for p_ax in [freq_ax, log_ax, cumulative_ax]:
            p_ax.set_yticks([])
            p_ax.xaxis.tick_top()
            p_ax.spines['left'].set_alpha(0.3)
            p_ax.spines['right'].set_alpha(0.3)
            p_ax.tick_params(labelsize=6)
            p_ax.grid(axis='x', alpha=0.3)
            
            p_ax.spines['bottom'].set_visible(False)
            
            p_ax.xaxis.set_label_position('top')
        
        freq_ax.set_xlabel('frequency', size=8)
        log_ax.set_xlabel('frequency (log10)', size=8)
        cumulative_ax.set_xlabel('cumulative frequency', size=8)
        
        ax.set_ylim(-0.5, len(outcome_order) - 0.5)

def outcome_diagrams_with_frequencies(pool, outcomes, **kwargs):
    fig = plot_outcome_diagrams(outcomes, pool.target_info, **kwargs)
    num_outcomes = kwargs.get('num_outcomes')
    add_frequencies(fig, fig.axes[0], pool, outcomes[:num_outcomes])
    fig.suptitle(pool.group)
    return fig

def plot_guide_specific_frequencies(outcome,
                                    pool,
                                    p_cutoff=4,
                                    num_to_label=20,
                                    genes_to_label=None,
                                    y_lims=None,
                                    p_val_method='binomial',
                                    outcome_name=None,
                                    guide_status='perfect',
                                   ):
    max_cells = max(pool.UMI_counts(guide_status))

    granular_df = pool.outcome_counts(guide_status)
    non_targeting = pool.non_targeting_guides

    if isinstance(outcome, tuple):
        nt_fraction = pool.non_targeting_fractions(guide_status)[outcome]
        outcome_counts = granular_df.loc[outcome]

    else:
        nt_counts = pool.non_targeting_counts(guide_status)
        nt_fraction = nt_counts[outcome].sum() / nt_counts.sum()
        outcome_counts = granular_df.loc[outcome].sum()

    if p_val_method == 'binomial':
        boundary_xs = np.arange(0, 40000)
        boundary_xs[0] = 1
        lower, upper = scipy.stats.binom.interval(1 - 2 * 10**-p_cutoff, boundary_xs, nt_fraction)
        boundary_lower = lower / boundary_xs
        boundary_upper = upper / boundary_xs

    #with h5py.File(pool.fns['quantiles']) as f:
    #    outcome_string = '_'.join(outcome)
    #    num_samples = f.attrs['num_samples']

    #    f_quantiles = f[outcome_string]['quantiles']
    #    f_frequencies = f[outcome_string]['frequencies']

    #    num_cells_list = f['num_cells'][:]
    #    indices = num_cells_list.argsort()
    #    num_cells_list = sorted(num_cells_list)
    #    
    #    geqs = {}
    #    leqs = {}
    #    quantiles = {}
    #    for q_key, q in quantiles_module.quantiles_to_record(num_samples).items():
    #        quantiles[q] = f_quantiles[q_key][:][indices]

    #    medians = dict(zip(num_cells_list, f_quantiles['median'][:][indices]))

    #    closest_sampled = np.zeros(40000, int)
    #    for num_cells in num_cells_list:
    #        closest_sampled[num_cells:] = num_cells

    #    for num_cells in num_cells_list:
    #        frequencies = f_frequencies[str(num_cells)][:]
    #        full_frequencies = np.zeros(num_cells + 1, int)
    #        full_frequencies[:len(frequencies)] = frequencies

    #        leq = np.cumsum(full_frequencies)
    #        geq = leq[-1] - leq + full_frequencies

    #        leqs[num_cells] = leq
    #        geqs[num_cells] = geq
    
    data = {
        'num_cells': pool.UMI_counts(guide_status),
        'count': outcome_counts,
    }

    df = pd.DataFrame(data)
    df.index.name = 'guide'
    df['fraction'] = df['count'] / df['num_cells']

    df['gene'] = pool.guides_df.loc[df.index]['gene']

    #def direction_and_pval(actual_outcome_count, actual_num_cells):
    #    num_cells = closest_sampled[actual_num_cells]
    #    outcome_count = int(np.floor(actual_outcome_count * num_cells / actual_num_cells))

    #    if num_cells > 0:
    #        if outcome_count <= medians[num_cells]:
    #            direction = 'down'
    #            p = leqs[num_cells][outcome_count] / num_samples
    #        else:
    #            direction = 'up'
    #            p = geqs[num_cells][outcome_count] / num_samples
    #    else:
    #        direction = None
    #        p = 1

    #    return direction, p
    if p_val_method == 'binomial':
        def direction_and_pval(actual_outcome_count, actual_num_cells):
            p = scipy.stats.binom.cdf(actual_outcome_count, actual_num_cells, nt_fraction)
            if actual_outcome_count < nt_fraction * actual_num_cells:
                direction = 'down'
            elif actual_outcome_count >= nt_fraction * actual_num_cells:
                direction = 'up'
                p = 1 - p
            
            return direction, p
    
    d_and_ps = starmap(direction_and_pval, zip(df['count'], df['num_cells']))
    df['direction'], df['pval'] = zip(*d_and_ps)

    significant = df[df['pval'] <= 10**-p_cutoff]

    df['color'] = 'silver'
    df.loc[non_targeting, 'color'] = 'C0'
    df.loc[significant.index, 'color'] = 'C1'

    if y_lims is None:
        y_max = significant['fraction'].max() * 1.1
        y_min = significant['fraction'].min() * 0.9
        if y_min < 0.1:
            y_min = 0
    else:
        y_min, y_max = y_lims

    g = sns.JointGrid('num_cells', 'fraction',
                      data=df,
                      height=10,
                      xlim=(-0.01 * max_cells, 1.01 * max_cells),
                      ylim=(y_min, y_max),
                      space=0,
                     )

    g.plot_marginals(sns.distplot, kde=False, color='silver')

    #for q in [10**-p_cutoff, 1 - 10**-p_cutoff]:
    #    ys = quantiles[q] / num_cells_list

    #    g.ax_joint.plot(num_cells_list, ys, color='black', alpha=0.3)

    #    x = min(1.01 * max_cells, max(num_cells_list))
    #    g.ax_joint.annotate(str(q),
    #                        xy=(x, ys[-1]),
    #                        xytext=(-10, -5 if q < 0.5 else 5),
    #                        textcoords='offset points',
    #                        ha='right',
    #                        va='top' if q < 0.5 else 'bottom',
    #                        clip_on=False,
    #                        size=6,
    #                       )
    if p_val_method == 'binomial':
        to_plot = boundary_xs[1:]
        g.ax_joint.plot(to_plot, boundary_lower[to_plot], color='black', alpha=0.3)
        g.ax_joint.plot(to_plot, boundary_upper[to_plot], color='black', alpha=0.3)

        x = int(np.floor(1.01 * max_cells))
        for q, y in [(10**-p_cutoff, boundary_lower[x]),
                     (1 - 10**-p_cutoff, boundary_upper[x]),
                    ]:
            g.ax_joint.annotate(str(q),
                                xy=(x, y),
                                xytext=(-10, -5 if q < 0.5 else 5),
                                textcoords='offset points',
                                ha='right',
                                va='top' if q < 0.5 else 'bottom',
                                clip_on=False,
                                size=6,
                            )

    for text, offset, color in [('non-targeting guides', -10, 'C0'),
                                ('{} p-value < 1e-{}'.format(p_val_method, p_cutoff), -25, 'C1'),
                               ]:
        g.ax_joint.annotate(text,
                            xy=(1, 1),
                            xycoords='axes fraction',
                            xytext=(-5, offset),
                            textcoords='offset points',
                            color=color,
                            ha='right',
                            va='top',
                           )

    g.ax_joint.scatter('num_cells', 'fraction',
                       data=df,
                       s=25,
                       alpha=0.9,
                       color='color',
                       linewidths=(0,),
                       )

    g.ax_joint.axhline(nt_fraction, color='black')

    g.ax_joint.set_xlabel('number of UMIs', size=16)

    if outcome_name is None:
        outcome_name = '_'.join(outcome)
    g.ax_joint.set_ylabel('fraction of UMIs with {}'.format(outcome_name), size=16)

    ax_marg_x_p = g.ax_marg_x.get_position()
    ax_marg_y_p = g.ax_marg_y.get_position()

    diagram_width = ax_marg_x_p.width * 0.5 + ax_marg_y_p.width
    diagram_gap = ax_marg_x_p.height * 0.3

    wt = ('wild type', 'wild type', 'n/a')
    if isinstance(outcome, tuple):
        outcomes_to_plot = [wt, outcome]
    else:
        outcomes_to_plot = [wt] + list(pool.non_targeting_counts('perfect').loc[outcome].sort_values(ascending=False).index.values[:3])

    diagram_height = ax_marg_x_p.height * 0.1 * len(outcomes_to_plot)

    diagram_ax = g.fig.add_axes((ax_marg_y_p.x1 - diagram_width, ax_marg_x_p.y1 - diagram_gap - diagram_height, diagram_width, diagram_height))
    plot_outcome_diagrams(outcomes_to_plot,
                          pool.target_info,
                          window=(-50, 20),
                          ax=diagram_ax,
                          flip_if_reverse=True,
                         )

    if num_to_label is not None:
        up = significant.query('direction == "up"').sort_values('fraction', ascending=False)[:num_to_label]
        down = significant.query('direction == "down"').sort_values('fraction')[:num_to_label]
        to_label = pd.concat([up, down])

        # Don't label any points that will be labeled by gene-labeling below.
        if genes_to_label is not None:
            to_label = to_label[~to_label['gene'].isin(genes_to_label)]

        to_label = to_label.query('fraction > @y_min and fraction < @y_max')

        vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        Visualize.label_scatter_plot(g.ax_joint, 'num_cells', 'fraction', 'guide',
                                     data=to_label,
                                     vector=vector,
                                     text_kwargs=dict(size=8),
                                     initial_distance=5,
                                     distance_increment=5,
                                     arrow_alpha=0.2,
                                     avoid=True,
                                     avoid_axis_labels=True,
                                    )

    if genes_to_label is not None:
        to_label = df[df['gene'].isin(genes_to_label)]
        to_label = to_label.query('fraction > @y_min and fraction < @y_max')

        vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        Visualize.label_scatter_plot(g.ax_joint, 'num_cells', 'fraction', 'guide',
                                     data=to_label,
                                     vector=vector,
                                     text_kwargs=dict(size=8, weight='bold'),
                                     initial_distance=8,
                                     distance_increment=5,
                                     arrow_alpha=0.2,
                                     avoid=True,
                                     avoid_axis_labels=True,
                                     avoid_existing=True,
                                     min_arrow_distance=0,
                                    )

    Visualize.add_commas_to_ticks(g.ax_joint, which='x')

    return g, df

colors_list = [
    bokeh.palettes.Greens9[2:],
    bokeh.palettes.Purples9[2:],
    bokeh.palettes.PuRd9[2:],
    bokeh.palettes.Blues9[2:],
    bokeh.palettes.Oranges9[2:],
] * 2

def plot_genes(pool, genes, guide_status='perfect', zoom=False):
    guides_df = pool.guides_df
    
    if len(genes) > 1:
        gene_to_colors = dict(zip(genes, colors_list))
    else:
        gene = genes[0]
        if len(pool.gene_guides(gene)) > 6:
            gene_to_colors = {gene: ['grey']*1000}
        elif len(pool.gene_guides(gene)) > 3:
            gene_to_colors = {gene: colors_list[0][:3] + colors_list[1][:3] + colors_list[2][:3]}
        else:
            gene_to_colors = {gene: [colors_list[1][0], colors_list[2][0], colors_list[3][0]]}
        
    if zoom:
        ax_order = [
            'frequency_zoom',
            'change_zoom',
            'log2_fold_change',
        ]
    else:
        ax_order = [
            'frequency',
            'change',
            'log2_fold_change',
        ]

    num_outcomes = len(pool.most_frequent_outcomes)
    #num_outcomes = 20

    fig, ax_array = plt.subplots(1, len(ax_order), figsize=(8, 48 * num_outcomes / 200), sharey=True, gridspec_kw={'wspace': 0.05})
    axs = dict(zip(ax_order, ax_array))

    order, sizes = pool.rational_outcome_order()
    #order = pool.most_frequent_outcomes[:num_outcomes]
    sizes = [len(order)]

    ys = np.arange(len(order))[::-1]

    for ax in axs.values():
        ax.xaxis.tick_top()

    guides = guides_df[guides_df.gene.isin(genes)].index

    def dot_and_line(xs, ax, color, label, line_width=1, marker_size=2.5, marker_alpha=0.6, line_alpha=0.25):
        ax.plot(list(xs), ys, 'o', markersize=marker_size, color=color, alpha=marker_alpha, label=label)
        ax.plot(list(xs), ys, '-', linewidth=line_width, color=color, alpha=line_alpha)
    
    nt_fracs = pool.common_non_targeting_fractions[order]
    for key in ('frequency', 'frequency_zoom'):
        if key in axs:
            dot_and_line(nt_fracs * 100, axs[key], 'grey', 'non-targeting', line_width=1.5, line_alpha=0.9, marker_alpha=1, marker_size=6)
    
    def guide_to_color(guide):
        gene = guides_df['gene'][guide]
        i = list(pool.gene_guides(gene)).index(guide)
        color = gene_to_colors[gene][i]
        
        return color
        
    for guide in guides:
        fractions = pool.common_fractions(guide_status)[guide]
        absolute_change = fractions - pool.common_non_targeting_fractions
        fold_changes = pool.fold_changes(guide_status)[guide]
        log2_fold_changes = pool.log2_fold_changes(guide_status)[guide]
        
        color = guide_to_color(guide)
        label = guide

        for key in ('change', 'change_zoom'):
            if key in axs:
                dot_and_line(absolute_change[order] * 100, axs[key], color, label)
            
        if 'fold_change' in axs:
            dot_and_line(fold_changes[order], axs['fold_change'], color, label)
        if 'log2_fold_change' in axs:
            dot_and_line(log2_fold_changes[order], axs['log2_fold_change'], color, label)
        
        for key in ('frequency', 'frequency_zoom'):
            if key in axs:
                dot_and_line(fractions[order] * 100, axs[key], color, label, line_alpha=0.15)
    
    for key, line_x, title, x_lims in [('change', 0, 'change in percentage', (-8, 8)),
                                       ('change_zoom', 0, 'change in percentage\n(zoomed)', (-0.75, 0.75)),
                                       #('fold_change', 1, 'fold change', (0, 5)),
                                       ('log2_fold_change', 0, 'log2 fold change', (-3, 3)),
                                       ('frequency', None, 'percentage', (0, nt_fracs.max() * 100 * 1.05)),
                                       ('frequency_zoom', None, 'percentage\n(zoomed)', (0, 0.75)),
                                      ]:
        if key not in axs:
            continue

        if line_x is not None:
            axs[key].axvline(line_x, color='black', alpha=0.3)
       
        axs[key].annotate(title,
                          xy=(0.5, 1),
                          xycoords='axes fraction',
                          xytext=(0, 30),
                          textcoords='offset points',
                          ha='center',
                          va='bottom',
                         )
                    
        axs[key].xaxis.set_label_coords(0.5, 1.025)

        axs[key].set_xlim(*x_lims)
        
        # Draw lines separating categories.
        for y in np.cumsum(sizes):
            flipped_y = len(order) - y - 0.5
            axs[key].axhline(flipped_y, color='black', alpha=0.1)
    
    plt.draw()
    
    width, height = fig.get_size_inches()
    
    ax_p = axs['log2_fold_change'].get_position()
    
    start_x = ax_p.x1 + 0.05 * ax_p.width
    
    heatmap_axs = {}
    for gene in genes:
        gene_guides = pool.gene_guides(gene)
        
        vals = pool.log2_fold_changes(guide_status).loc[order[::-1], gene_guides]
        
        num_rows, num_cols = vals.shape
    
        heatmap_width = ax_p.height * num_cols / num_rows * height / width
        heatmap_ax = fig.add_axes((start_x, ax_p.y0, heatmap_width , ax_p.height), sharey=axs[ax_order[0]])
                 
        im = heatmap_ax.imshow(vals, cmap=plt.get_cmap('RdBu_r'), vmin=-2, vmax=2, origin='lower')

        heatmap_ax.xaxis.tick_top()
        heatmap_ax.set_xticks(np.arange(len(gene_guides)))
        heatmap_ax.set_xticklabels([guide for guide in gene_guides], rotation=90)
        
        for label in heatmap_ax.get_xticklabels():
            guide = label.get_text()
            color = guide_to_color(guide)
            label.set_color(color)

        for spine in heatmap_ax.spines.values():
            spine.set_visible(False)
            
        start_x += heatmap_width * 1.1

        heatmap_axs[gene] = heatmap_ax
    
    plt.draw()

    for key in ['frequency_zoom', 'change_zoom', 'fold_change']:
        if key in axs:
            axs[key].set_xticklabels(list(axs[key].get_xticklabels())[:-1] + [''])

    ax_p = axs[ax_order[0]].get_position()

    diagram_width = ax_p.width * 3
    diagram_gap = diagram_width * 0.02

    diagram_ax = fig.add_axes((ax_p.x0 - diagram_width - diagram_gap, ax_p.y0, diagram_width, ax_p.height), sharey=axs[ax_order[0]])
    
    plot_outcome_diagrams(order, pool.target_info,
                          num_outcomes=len(order),
                          window=(-50, 50),
                          ax=diagram_ax,
                         )

    heatmap_p = heatmap_axs[genes[0]].get_position()
    heatmap_x0 = heatmap_p.x0
    heatmap_x1 = heatmap_axs[genes[-1]].get_position().x1
    heatmap_height = heatmap_p.height

    log2_p = axs['log2_fold_change'].get_position()

    bar_x0 = np.mean([heatmap_x0, heatmap_x1]) - log2_p.width / 2

    cbar_offset = heatmap_height * 7 / len(order)
    cbar_height = heatmap_height * 1 / len(order)
    cbar_ax = fig.add_axes((bar_x0, heatmap_p.y1 + cbar_offset, log2_p.width, cbar_height)) 

    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(['$\leq$-2', '-1', '0', '1', '$\geq$2'])
    cbar_ax.xaxis.tick_top()

    cbar_ax.annotate('gene activity\npromotes outcome',
                    xy=(0, 0),
                    xycoords='axes fraction',
                    xytext=(0, -5),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    size=8,
                    )

    cbar_ax.annotate('gene activity\nsuppresses outcome',
                    xy=(1, 0),
                    xycoords='axes fraction',
                    xytext=(0, -5),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    size=8,
                    )

    return fig

def plot_guide_scatter(pool, gene, number, ax=None, outcomes_to_draw=15):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    max_fc = 5 
    min_frequency = 6

    guide = pool.gene_guides(gene)[number]

    nt_counts = pool.outcome_counts('perfect')[pool.non_targeting_guides].sum(axis=1)
    nt_fracs = nt_counts / nt_counts.sum()
    
    fracs = pool.outcome_fractions('perfect')[guide]
    ratios = fracs / nt_fracs

    ratios[nt_fracs == 0] = 2**max_fc
    ratios[fracs == 0] = 2**-max_fc

    ratios = np.minimum(2**max_fc, ratios)
    ratios = np.maximum(2**-max_fc, ratios)

    data = {
        'fracs': fracs,
        'nt_fracs': nt_fracs,
        'log2_fc': np.log2(ratios),
        'log10_nt_frac': np.log10(np.maximum(10**-6, nt_fracs)),
    }
    data = pd.DataFrame(data)
    data = data.query('fracs != 0 or nt_fracs != 0').copy()

    data['labels'] = ['_'.join(v) for v in data.index.values]
    data.index.name = 'outcome'

    num_UMIs = pool.UMI_counts('perfect')[guide]

    ax.annotate('{:,} UMIs'.format(num_UMIs),
                xy=(1, 1),
                xycoords='axes fraction',
                xytext=(-5, -5),
                textcoords='offset points',
                ha='right',
                va='top',
               )

    if num_UMIs == 0:
        return fig

    nt_lows, nt_highs = scipy.stats.binom.interval(1 - 10**-4, num_UMIs, data['nt_fracs'])

    nt_lows = nt_lows / num_UMIs
    nt_highs = nt_highs / num_UMIs

    data['color'] = 'grey'
    data['up'] = data['fracs'] > nt_highs
    data['down'] = (data['fracs'] < nt_lows) & (data['nt_fracs'] > 0)

    data.loc[data['down'], 'color'] = 'C0'
    data.loc[data['up'], 'color'] = 'C3'
    data['ignore'] = (data['log2_fc'] == max_fc) & (data['log10_nt_frac'] == -min_frequency)

    ax.scatter('log10_nt_frac', 'log2_fc',
               c='color',
               s=50,
               data=data,
               alpha=0.8,
               linewidths=(0,),
              )
    ax.axhline(0, color='black', alpha=0.8)

    for cutoff in [4]:
        ps = 10**np.linspace(-7, 0, 100)
        lows, highs = scipy.stats.binom.interval(1 - 10**-cutoff, num_UMIs, ps)
        interval_xs = np.log10(ps)
        high_ys = np.log2(np.maximum(2**-6, (highs / num_UMIs) / ps))
        low_ys = np.log2(np.maximum(2**-6, (lows / num_UMIs) / ps))

        for ys in [low_ys, high_ys]:
            ax.plot(interval_xs, ys, color='grey', alpha=0.75)

    ax.set_xlim(-min_frequency - 0.1, 0)
    ax.set_ylim(-5.1, 5.1)

    ax.set_xlabel('log10(fraction in non-targeting)', size=16)
    ax.set_ylabel('log2(fold change in knockdown)', size=16)

    ax.set_yticks(np.arange(-5, 6, 1))

    ax.set_title(guide, size=16)

    plt.draw()
    ax_p = ax.get_position()
    
    up_outcomes = data.query('up and not ignore').sort_values('nt_fracs', ascending=False).iloc[:outcomes_to_draw].sort_values('log2_fc', ascending=False)
    if len(up_outcomes) > 0:
        up_ax = fig.add_axes((ax_p.x1 + ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.55, ax_p.width , ax_p.height * 0.4))
        
        plot_outcome_diagrams(up_outcomes.index, pool.target_info,
                              ax=up_ax,
                              num_outcomes=outcomes_to_draw,
                              window=(-50, 50),
                             )
        left, right = up_ax.get_xlim()
        
        num_up = min(outcomes_to_draw, len(up_outcomes))
        up_ax.set_ylim(num_up - 0.5 - outcomes_to_draw, num_up - 0.5)

        for y, (outcome, row) in enumerate(up_outcomes.iterrows()):
            con = ConnectionPatch(xyA=(row['log10_nt_frac'], row['log2_fc']),
                                xyB=(left, num_up - y - 1),
                                coordsA='data',
                                coordsB='data',
                                axesA=ax,
                                axesB=up_ax,
                                color='grey',
                                alpha=0.15,
                                )
            ax.add_artist(con)
        
    down_outcomes = data.query('down').sort_values('nt_fracs', ascending=False).iloc[:outcomes_to_draw].sort_values('log2_fc', ascending=False)
    if len(down_outcomes) > 0:
        down_ax = fig.add_axes((ax_p.x1 + ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.05, ax_p.width , ax_p.height * 0.4))

        plot_outcome_diagrams(down_outcomes.index, pool.target_info,
                              ax=down_ax,
                              num_outcomes=outcomes_to_draw,
                              window=(-50, 50),
                             )
        left, right = down_ax.get_xlim()

        num_down = min(outcomes_to_draw, len(down_outcomes))
        down_ax.set_ylim(num_down - 0.5 - outcomes_to_draw, num_down - 0.5)
        
        for y, (outcome, row) in enumerate(down_outcomes.iterrows()):
            con = ConnectionPatch(xyA=(row['log10_nt_frac'], row['log2_fc']),
                                xyB=(left, num_down - y - 1),
                                coordsA='data',
                                coordsB='data',
                                axesA=ax,
                                axesB=down_ax,
                                color='grey',
                                alpha=0.15,
                                )
            ax.add_artist(con)

    text_kwargs = dict(
                xycoords='axes fraction',
                textcoords='offset points',
                xy=(0, 0.1),
                ha='left',
                va='center',
    )
    ax.annotate('significant increase (gene activity suppresses outcome)',
                xytext=(8, 0),
                color='C3',
                **text_kwargs)
    ax.annotate('significant decrease (gene activity promotes outcome)',
                xytext=(8, -12),
                color='C0',
                **text_kwargs)

    labels = ax.get_yticklabels()
    for l in labels:
        x, y = l.get_position()
        if y == max_fc:
            l.set_text('$\geq${}'.format(max_fc))
        elif y == -max_fc:
            l.set_text('$\leq${}'.format(-max_fc))
            
    ax.set_yticklabels(labels)

    labels = ax.get_xticklabels()
    for l in labels:
        x, y = l.get_position()
        if x == -min_frequency:
            l.set_text('$\leq${}'.format(min_frequency))

    ax.set_xticklabels(labels)
    
    return fig

def plot_gene_scatter(pool, gene):
    guides = pool.gene_guides(gene)

    fig, axs = plt.subplots(len(guides), 1, figsize=(10, 10 * len(guides)))
    if len(guides) == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        plot_guide_scatter(pool, gene, i, ax=ax)

    return fig

def gene_significance(pool, outcomes, draw_outcomes=False):
    df, nt_fraction, p_df = get_outcome_statistics(pool, outcomes)

    labels = list(pool.guides)

    gene_to_color = {g: 'C{}'.format(i % 10) for i, g in enumerate(pool.genes)}

    fig = plt.figure(figsize=(36, 12))

    axs = {}
    gs = matplotlib.gridspec.GridSpec(2, 4, hspace=0.03)
    axs['up'] = plt.subplot(gs[0, :-1])
    axs['down'] = plt.subplot(gs[1, :-1])
    axs['outcomes'] = plt.subplot(gs[:, -1])

    significant = {}
    global_max_y = 0
    for direction in ['up',
                      'down',
                     ]:
        ax = axs[direction]
        ordered = p_df[direction].sort_values()
        genes_to_label = set(ordered.index[:50])
        subset =  ordered
        subset_df = pd.DataFrame({'gene': subset.index, 'p_val': list(subset)}, index=np.arange(1, len(subset) + 1))
        significant[direction] = subset_df
        
        guides_to_label = {g for g in pool.guides if pool.guide_to_gene(g) in genes_to_label}

        colors = [gene_to_color[pool.guide_to_gene(guide)] for guide in labels]
        colors = matplotlib.colors.to_rgba_array(colors)
        alpha = [0.95 if pool.guide_to_gene(guide) in genes_to_label else 0.15 for guide in pool.guides]
        colors[:, 3] = alpha

        ax.scatter(np.arange(len(df)), df['frequency'], s=25, c=colors, linewidths=(0,))
        ax.set_xlim(-10, len(pool.guides) + 10)

        for x, (y, label) in enumerate(zip(df['frequency'], labels)):
            if label in guides_to_label:
                ax.annotate(label,
                            xy=(x, y),
                            xytext=(2, 0),
                            textcoords='offset points',
                            size=6,
                            color=colors[x],
                            va='center',
                           )

        ax.axhline(nt_fraction, color='black', alpha=0.5)
        ax.annotate('{:0.3f}'.format(nt_fraction),
                    xy=(1, nt_fraction),
                    xycoords=('axes fraction', 'data'),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
                    size=10,
                    va='center',
                   )

        relevant_ys = df.loc[guides_to_label]['frequency']

        max_y = max(nt_fraction * 2, relevant_ys.max() * 1.1)
        global_max_y = max(global_max_y, max_y)
        ax.set_xticklabels([])

        for x, (_, row) in enumerate(df.iterrows()):
            below, above = utilities.clopper_pearson(row['outcome_count'], row['total_UMIs'])
            ax.plot([x, x], [row['frequency'] - below, row['frequency'] + above], color=colors[x])

    for direction in ['up', 'down']:
        axs[direction].set_ylim(0, global_max_y)

    axs['up'].set_title('most significant suppressing genes', y=0.92)
    axs['down'].set_title('most significant promoting genes', y=0.92)

    if draw_outcomes:
        n = 40
        outcome_order = pool.non_targeting_fractions('perfect').loc[outcomes].sort_values(ascending=False).index[:n]
        plot_outcome_diagrams(outcome_order, pool.target_info, num_outcomes=n, window=(-60, 20), flip_if_reverse=True, ax=axs['outcomes'])
        add_frequencies(fig, axs['outcomes'], pool, outcome_order[:n], text_only=True)
    else:
        fig.delaxes(axs['outcomes'])

    fig.suptitle(pool.group, size=16)
    
    return fig, significant