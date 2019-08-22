from collections import Counter, defaultdict
from itertools import starmap, repeat
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import h5py
import bokeh.plotting
import bokeh.palettes
from matplotlib.patches import ConnectionPatch

from knock_knock.target_info import degenerate_indel_from_string, SNVs, effectors
from hits import utilities
import hits.visualize

from . import quantiles as quantiles_module
from .pooled_layout import HDROutcome, DeletionOutcome, HDRPlusDeletionOutcome, DeletionPlusMismatchOutcome

def plot_outcome_diagrams(outcome_order, target_info, num_outcomes=None, title=None, window=70,
                          ax=None,
                          flip_if_reverse=True,
                          center_at_PAM=False,
                          draw_cut_afters=True,
                          size_multiple=0.8,
                          draw_all_sequence=False,
                          draw_imperfect_MH=False,
                          draw_wild_type_on_top=False,
                          ):
    if isinstance(window, int):
        window_left, window_right = -window, window
    else:
        window_left, window_right = window

    window_size = window_right - window_left

    if num_outcomes is None:
        num_outcomes = len(outcome_order)

    outcome_order = outcome_order[:num_outcomes]

    if ax is None:
        fig, ax = plt.subplots(figsize=(size_multiple * 20 * window_size / 140, size_multiple * num_outcomes / 3))
    else:
        fig = ax.figure

    guide = target_info.features[target_info.target, target_info.sgRNA]

    if flip_if_reverse and guide.strand == '-':
        flip = True
        transform_seq = utilities.complement
        cut_offset_sign = -1
    else:
        flip = False
        transform_seq = utilities.identity
        cut_offset_sign = 1
    
    if center_at_PAM:
        if guide.strand == '+':
            offset = target_info.PAM_slice.start
        else:
            offset = target_info.PAM_slice.stop - 1
    else:
        if guide.strand == '-':
            offset = max(target_info.cut_afters.values())
        else:
            offset = max(target_info.cut_afters.values())
    
    guide_start = guide.start - 0.5 - offset
    guide_end = guide.end + 0.5 - offset

    if draw_cut_afters:
        for cut_after in target_info.cut_afters.values():
            x = (cut_after + 0.5 * cut_offset_sign) - offset

            ax.plot([x, x], [-0.5, num_outcomes - 0.5], color='black', linestyle='--', alpha=0.3)
    
    if flip:
        window_left, window_right = -window_right, -window_left

    seq = target_info.target_sequence[offset + window_left:offset + window_right + 1]

    def draw_rect(x0, x1, y0, y1, alpha, color='black'):
        if x0 > window_right or x1 < window_left:
            return

        x0 = max(x0, window_left - 0.5)
        x1 = min(x1, window_right + 0.5)

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
                            linewidth=0,
                            clip_on=False,
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
                            annotation_clip=False,
                           )

    def draw_deletion(y, deletion, color='black', draw_MH=True):
        xs_to_skip = set()

        starts = np.array(deletion.starts_ats) - offset
        if draw_MH and len(starts) > 1:
            for x, b in zip(range(window_left, window_right + 1), seq):
                if (starts[0] <= x < starts[-1]) or (starts[0] + deletion.length <= x < starts[-1] + deletion.length):
                    ax.annotate(transform_seq(b),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=Visualize.igv_colors[transform_seq(b)],
                                weight='bold',
                               )

                    xs_to_skip.add(x)

        if draw_imperfect_MH:
            before_MH = np.arange(starts.min() - 5, starts.min())
            after_MH = np.arange(starts.max(), starts.max() + 5)
            left_xs = np.concatenate((before_MH, after_MH))
            for left_x in left_xs:
                right_x = left_x + deletion.length

                # Ignore if overlaps perfect MH as a heuristic for whether interesting 
                if right_x < starts.max() or left_x >= starts.min() + deletion.length:
                    continue

                if all(0 <= x - window_left < len(seq) for x in [left_x, right_x]):
                    left_b = seq[left_x - window_left]
                    right_b = seq[right_x - window_left]
                    if left_b == right_b:
                        for x, b in ((left_x, left_b), (right_x, right_b)):
                            ax.annotate(transform_seq(b),
                                        xy=(x, y),
                                        xycoords='data', 
                                        ha='center',
                                        va='center',
                                        size=text_size,
                                        color=Visualize.igv_colors[b],
                                    )
                            xs_to_skip.add(x)

        del_height = 0.15
        
        del_start = starts[0] - 0.5
        del_end = starts[0] + deletion.length - 1 + 0.5
        
        draw_rect(del_start, del_end, y - del_height / 2, y + del_height / 2, 0.4, color=color)
        draw_rect(window_left - 0.5, del_start, y - wt_height / 2, y + wt_height / 2, block_alpha)
        draw_rect(del_end, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

        return xs_to_skip

    def draw_wild_type(y, on_top=False):
        PAM_start = target_info.PAM_slice.start - 0.5 - offset
        PAM_end = target_info.PAM_slice.stop + 0.5 - 1 - offset

        draw_rect(guide_start, guide_end, y - wt_height / 2, y + wt_height / 2, 0.3, color='blue')
        draw_rect(PAM_start, PAM_end, y - wt_height / 2, y + wt_height / 2, 0.3, color='green')

        if not on_top:
            draw_rect(window_left - 0.5, min(PAM_start, guide_start), y - wt_height / 2, y + wt_height / 2, block_alpha)
            draw_rect(max(PAM_end, guide_end), window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

        draw_sequence(y, alpha=1)

    for i, (category, subcategory, details) in enumerate(outcome_order):
        y = num_outcomes - i - 1
            
        if category == 'deletion':
            deletion = degenerate_indel_from_string(details)
            xs_to_skip = draw_deletion(y, deletion)
            if draw_all_sequence:
                draw_sequence(y, xs_to_skip)
        
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
                                    color=Visualize.igv_colors[transform_seq(b)],
                                    weight='bold',
                                )
            
            if draw_all_sequence:
                draw_sequence(y)
                
        elif category == 'wild type':
            draw_wild_type(y)

        elif category == 'mismatches':
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
                                color=Visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                weight='bold',
                            )
            
            for (strand, position), ref_base in target_info.fingerprints[target_info.target]:
                color = 'grey'
                alpha = 0.3
                draw_rect(position - offset - 0.5, position - offset + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

            if draw_all_sequence:
                draw_sequence(y, xs_to_skip=SNV_xs)

        elif category == 'deletion + adjacent mismatch':
            outcome = DeletionPlusMismatchOutcome.from_string(details)
            xs_to_skip = draw_deletion(y, outcome.deletion_outcome.deletion, draw_MH=False)
            
            for snv in outcome.mismatch_outcome.snvs:
                x = snv.position - offset
                xs_to_skip.add(x)
                if window_left <= x <= window_right:
                    ax.annotate(transform_seq(snv.basecall),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=Visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                weight='bold',
                            )
            if draw_all_sequence:
                draw_sequence(y, xs_to_skip)


        elif category == 'donor' or category == 'donor + deletion':
            SNP_xs = set()

            if category == 'donor':
                HDR_outcome = HDROutcome.from_string(details)
                deletion_outcome = None
            elif category == 'donor + deletion':
                HDR_plus_deletion_outcome = HDRPlusDeletionOutcome.from_string(details)
                HDR_outcome = HDR_plus_deletion_outcome.HDR_outcome
                deletion_outcome = HDR_plus_deletion_outcome.deletion_outcome

            for ((strand, position), ref_base), read_base in zip(target_info.fingerprints[target_info.target], HDR_outcome.donor_SNV_read_bases):
                # Note: read base of '-' means it was deleted
                if ref_base != read_base and read_base != '_' and read_base != '-':
                    x = position - offset
                    SNP_xs.add(x)
                    ax.annotate(transform_seq(read_base),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                alpha=0.35,
                                #color=Visualize.igv_colors[transform_seq(read_base)],
                                #weight='bold',
                               )
            
                if read_base != '-':
                    if  read_base == '_':
                        color = 'grey'
                        alpha = 0.3
                    else:
                        color = Visualize.igv_colors[transform_seq(read_base)]
                        alpha = 0.7

                    draw_rect(position - offset - 0.5, position - offset + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

            all_deletions = [(d, 'red', False) for d in HDR_outcome.donor_deletions]
            if deletion_outcome is not None:
                all_deletions.append((deletion_outcome.deletion, 'black', True))

            if len(all_deletions) == 0:
                draw_rect(window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
            elif len(all_deletions) == 1:
                deletion, color, draw_MH = all_deletions[0]
                draw_deletion(y, deletion, color=color, draw_MH=draw_MH)
            elif len(all_deletions) > 1:
                raise NotImplementedError

            if draw_all_sequence:
                draw_sequence(y, xs_to_skip=SNP_xs)

        else:
            label = '{}, {}, {}'.format(category, subcategory, details)
            ax.annotate(label,
                        xy=(0, y),
                        xycoords='data', 
                        ha='center',
                        va='center',
                        size=text_size,
                        )

    if draw_wild_type_on_top:
        draw_wild_type(num_outcomes, on_top=True)
        ax.set_xticks([])
                
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

def add_frequencies(fig, ax, count_source, outcome_order, text_only=False):
    ax_p = ax.get_position()
    
    width = 0.2
    offset = 0.04

    if isinstance(count_source, dict):
        counts = np.array([count_source[outcome] for outcome in outcome_order])
        freqs = counts / sum(count_source.values())
    else:
        pool = count_source
        freqs = pool.non_targeting_fractions('perfect').loc[outcome_order]
        counts = pool.non_targeting_counts('perfect').loc[outcome_order]

    ys = np.arange(len(outcome_order) - 1, -1, -1)
    
    for y, freq, count in zip(ys, freqs, counts):
        ax.annotate('{:> 7.2%} {:>8s}'.format(freq, '({:,})'.format(count)),
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

def outcome_diagrams_with_frequencies(pool, outcomes, text_only=False, **kwargs):
    fig = plot_outcome_diagrams(outcomes, pool.target_info, **kwargs)
    num_outcomes = kwargs.get('num_outcomes')
    add_frequencies(fig, fig.axes[0], pool, outcomes[:num_outcomes], text_only=text_only)
    return fig

def plot_guide_specific_frequencies(outcome,
                                    pool,
                                    p_cutoff=4,
                                    num_to_label=20,
                                    genes_to_label=None,
                                    guides_to_label=None,
                                    y_lims=None,
                                    p_val_method='binomial',
                                    outcome_name=None,
                                    guide_status='perfect',
                                    guide_subset=None,
                                    draw_marginals=True,
                                    draw_diagram=True,
                                    gene_to_color=None,
                                    color_labels=False,
                                    initial_label_offset=7,
                                    only_color_best_promoter=False,
                                    guide_aliases=None,
                                    big_labels=False,
                                   ):
    if guide_aliases is None:
        guide_aliases = {}

    if big_labels:
        axis_label_size = 20
        tick_label_size = 14
        guide_label_size = 18
    else:
        axis_label_size = 12
        tick_label_size = 10
        guide_label_size = 12

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
    
    if guide_subset is None:
        guide_subset = df.index

    df = df.loc[guide_subset]
    
    def convert_alias(guide):
        for old, new in guide_aliases.items(): 
            if old in guide:
                guide = guide.replace(old, new)

        return guide

    df['alias'] = [convert_alias(g) for g in df.index]

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

    if gene_to_color is not None:
        for gene, color in gene_to_color.items():
            guides = df.index.intersection(pool.gene_guides(gene, only_color_best_promoter))
            df.loc[guides, 'color'] = color

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

    if draw_marginals:
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
        # Draw grey lines at significance thresholds away from the bulk non-targeting fraction.
        
        to_plot = boundary_xs[1:]
        g.ax_joint.plot(to_plot, boundary_lower[to_plot], color='black', alpha=0.3)
        g.ax_joint.plot(to_plot, boundary_upper[to_plot], color='black', alpha=0.3)

        # Annotate the lines with their significance level.
        #x = int(np.floor(1.01 * max_cells))
        #for q, y in [(10**-p_cutoff, boundary_lower[x]),
        #             (1 - 10**-p_cutoff, boundary_upper[x]),
        #            ]:
        #    g.ax_joint.annotate(str(q),
        #                        xy=(x, y),
        #                        xytext=(-10, -5 if q < 0.5 else 5),
        #                        textcoords='offset points',
        #                        ha='right',
        #                        va='top' if q < 0.5 else 'bottom',
        #                        clip_on=False,
        #                        size=6,
        #                    )

    for text, offset, color in [('non-targeting guides', -10, 'C0'),
                                #('{} p-value < 1e-{}'.format(p_val_method, p_cutoff), -25, 'C1'),
                               ]:
        g.ax_joint.annotate(text,
                            xy=(1, 1),
                            xycoords='axes fraction',
                            xytext=(-5, offset),
                            textcoords='offset points',
                            color=color,
                            ha='right',
                            va='top',
                            #size=18,
                           )

    g.ax_joint.scatter('num_cells', 'fraction',
                       data=df,
                       s=25,
                       alpha=0.9,
                       color='color',
                       linewidths=(0,),
                       )

    g.ax_joint.axhline(nt_fraction, color='black')

    g.ax_joint.set_xlabel('number of cells per CRISPRi guide', size=axis_label_size)

    if outcome_name is None:
        try:
            outcome_name = '_'.join(outcome)
        except TypeError:
            outcome_name = 'PH'

    g.ax_joint.set_ylabel('fraction of cells with {}'.format(outcome_name), size=axis_label_size)

    if draw_diagram:
        ax_marg_x_p = g.ax_marg_x.get_position()
        ax_marg_y_p = g.ax_marg_y.get_position()

        diagram_width = ax_marg_x_p.width * 0.5 + ax_marg_y_p.width
        diagram_gap = ax_marg_x_p.height * 0.3

        if isinstance(outcome, tuple):
            outcomes_to_plot = [outcome]
        else:
            outcomes_to_plot = list(pool.non_targeting_counts('perfect').loc[outcome].sort_values(ascending=False).index.values[:4])

        diagram_height = ax_marg_x_p.height * 0.1 * len(outcomes_to_plot)

        diagram_ax = g.fig.add_axes((ax_marg_y_p.x1 - diagram_width, ax_marg_x_p.y1 - diagram_gap - diagram_height, diagram_width, diagram_height))
        plot_outcome_diagrams(outcomes_to_plot,
                              pool.target_info,
                              window=(-50, 20),
                              ax=diagram_ax,
                              flip_if_reverse=True,
                              draw_all_sequence=False,
                              draw_wild_type_on_top=True,
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

        Visualize.label_scatter_plot(g.ax_joint, 'num_cells', 'fraction', 'alias',
                                     data=to_label,
                                     vector=vector,
                                     text_kwargs=dict(size=8),
                                     initial_distance=5,
                                     distance_increment=5,
                                     arrow_alpha=0.2,
                                     avoid=True,
                                     avoid_axis_labels=True,
                                    )

    if guides_to_label is not None or genes_to_label is not None:
        if guides_to_label is not None:
            to_label = df[df.index.isin(guides_to_label)]
        elif genes_to_label is not None:
            to_label = df[df['gene'].isin(genes_to_label)]

        to_label = to_label.query('fraction > @y_min and fraction < @y_max')

        vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        if color_labels:
            kwargs = dict(color='color',
                          text_kwargs=dict(size=guide_label_size),
                         )
        else:
            kwargs = dict(color=None,
                          text_kwargs=dict(size=guide_label_size),
                         )
        Visualize.label_scatter_plot(g.ax_joint, 'num_cells', 'fraction', 'alias',
                                     data=to_label,
                                     vector=vector,
                                     initial_distance=initial_label_offset,
                                     distance_increment=5,
                                     arrow_alpha=0.2,
                                     avoid=True,
                                     avoid_axis_labels=True,
                                     avoid_existing=True,
                                     min_arrow_distance=0,
                                     **kwargs)

    Visualize.add_commas_to_ticks(g.ax_joint, which='x')
    g.ax_joint.tick_params(labelsize=tick_label_size)

    if not draw_marginals:
        g.fig.delaxes(g.ax_marg_x)
        g.fig.delaxes(g.ax_marg_y)

    return g, df

single_colors = [
    bokeh.palettes.Greens9,
    bokeh.palettes.Purples9,
    bokeh.palettes.PuRd9,
    #bokeh.palettes.Blues9,
    bokeh.palettes.Oranges9,
]
colors_list = [color[2:2 + 3] + [color[2 + 3]]*6 for color in single_colors]*10
#colors_list = [
#    bokeh.palettes.Greens9[2:],
#    bokeh.palettes.Purples9[2:],
#    bokeh.palettes.PuRd9[2:],
#    bokeh.palettes.Blues9[2:],
#    bokeh.palettes.Oranges9[2:],
#] * 2

colors = bokeh.palettes.Category10[10]
good_colors = (colors[2:7] + colors[8:])*100

colors_list = [[c]*10 for c in good_colors]

def plot_genes(pool, genes,
               only_best_promoter=False,
               guide_status='perfect',
               outcomes=None,
               zoom=False,
               just_percentages=False,
               ax_order=None,
               log_2=True,
               layout_kwargs=None,
               v_min=-2, v_max=2,
               frequency_max_multiple=1.05,
               draw_nt_fracs=True,
               draw_heatmaps=True,
               draw_colorbar=True,
               highlight_targeting_guides=False,
               genes_to_draw=None,
               gene_to_colors=None,
               guides_to_draw=None,
               panel_width_multiple=2.7,
               just_heatmaps=False,
               show_pool_name=True,
               guide_aliases=None,
              ):

    if guide_aliases is None:
        guide_aliases = {}

    if layout_kwargs is None:
        layout_kwargs = {'draw_all_sequence': False}

    guides_df = pool.guides_df
    
    if gene_to_colors is None:
        gene_to_colors = {'negative_control': ['C0']*1000}
        real_genes = [g for g in genes if g != 'negative_control']
        if len(real_genes) > 1:
            gene_to_colors.update(dict(zip(real_genes, colors_list)))

        elif len(real_genes) == 1:
            gene = real_genes[0]
            guides = pool.gene_guides(gene, only_best_promoter)
            if len(guides) > 6:
                gene_to_colors = {gene: ['C0']*1000}
            elif len(guides) > 3:
                gene_to_colors = {gene: colors_list[0][:3] + colors_list[1][:3] + colors_list[2][:3]}
            else:
                gene_to_colors = {gene: [colors_list[1][0], colors_list[2][0], colors_list[3][0]]}
        
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
        order, sizes = pool.rational_outcome_order()
    elif isinstance(outcomes, int):
        order, sizes = pool.rational_outcome_order(outcomes)
    else:
        order = outcomes
        sizes = [len(order)]

    num_outcomes = len(order)

    ys = np.arange(len(order))[::-1]
    
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

    guides = pool.gene_guides(genes, only_best_promoter)

    def dot_and_line(xs, ax, color, label, line_width=1, marker_size=3, marker_alpha=0.6, line_alpha=0.25):
        ax.plot(list(xs), ys, 'o', markeredgewidth=0, markersize=marker_size, color=color, alpha=marker_alpha, label=label, clip_on=False)
        ax.plot(list(xs), ys, '-', linewidth=line_width, color=color, alpha=line_alpha, clip_on=False)
    
    nt_fracs = pool.non_targeting_fractions('all')[order]

    if draw_nt_fracs:
        for key in ('frequency', 'frequency_zoom'):
            if key in axs:
                dot_and_line(nt_fracs * 100, axs[key], 'grey', 'non-targeting', line_width=1.5, line_alpha=0.9, marker_alpha=1, marker_size=6)
    
    def guide_to_color(guide):
        gene = guides_df['gene'][guide]
        i = list(pool.gene_guides(gene)).index(guide)
        color = gene_to_colors[gene][i]
        
        return color

    fractions = pool.outcome_fractions(guide_status).loc[order]
    absolute_change = fractions.sub(pool.non_targeting_fractions('all')[order], axis=0)
    fold_changes = pool.fold_changes(guide_status).loc[order]
    log2_fold_changes = pool.log2_fold_changes(guide_status).loc[order]

    for guide in guides:
        if genes_to_draw is not None:
            if pool.guide_to_gene(guide) not in genes_to_draw:
                continue

        if guides_to_draw is not None:
            if guide not in guides_to_draw:
                continue

        color = guide_to_color(guide)
        label = guide

        for key in ('change', 'change_zoom'):
            if key in axs:
                dot_and_line(absolute_change[guide] * 100, axs[key], color, label)
            
        if pool.guide_to_gene(guide) != 'negative_control':
            if highlight_targeting_guides:
                kwargs = dict(marker_size=5, line_width=2)
            else:
                kwargs = dict()

            if 'fold_change' in axs:
                dot_and_line(fold_changes[guide], axs['fold_change'], color, label, **kwargs)

            if 'log2_fold_change' in axs:
                dot_and_line(log2_fold_changes[guide], axs['log2_fold_change'], color, label, **kwargs)
        
        for key in ('frequency', 'frequency_zoom'):
            if key in axs:
                if highlight_targeting_guides:
                    if pool.guide_to_gene(guide) == 'negative_control':
                        kwargs = dict(line_alpha=0.15)
                    else:
                        kwargs = dict(line_alpha=0.45, marker_size=5, line_width=2)
                else:
                    kwargs = dict(line_alpha=0.15)

                dot_and_line(fractions[guide] * 100, axs[key], color, label, **kwargs)
    
    for key, line_x, title, x_lims in [('change', 0, 'change in percentage', (-8, 8)),
                                       ('change_zoom', 0, 'change in percentage\n(zoomed)', (-0.75, 0.75)),
                                       ('fold_change', 1, 'fold change', (0, 5)),
                                       ('log2_fold_change', 0, 'log2 fold change\nfrom non-targeting', (-3, 3)),
                                       ('frequency', None, 'percentage of\nrepair outcomes', (0, nt_fracs.max() * 100 * frequency_max_multiple)),
                                       ('frequency_zoom', None, 'percentage\n(zoomed)', (0, nt_fracs.max() * 100 * 0.3)),
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
                          #size=14,
                         )
                    
        axs[key].xaxis.set_label_coords(0.5, 1.025)

        axs[key].set_xlim(*x_lims)
        
        # Draw lines separating categories.
        for y in np.cumsum(sizes):
            flipped_y = len(order) - y - 0.5
            axs[key].axhline(flipped_y, color='black', alpha=0.1)
    
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
        if highlight_targeting_guides:
            genes_to_plot = [g for g in genes if g != 'negative_control']
        else:
            genes_to_plot = genes

        def convert_alias(guide):
            for old, new in guide_aliases.items(): 
                if old in guide:
                    guide = guide.replace(old, new)

            return guide

        first_gene = None

        for gene in genes_to_plot:
            gene_guides = pool.gene_guides(gene, only_best_promoter)
            if guides_to_draw is not None:
                gene_guides = [guide for guide in gene_guides if guide in guides_to_draw]

            if len(gene_guides) == 0:
                continue

            if first_gene is None:
                first_gene = gene
            
            vals = pool.log2_fold_changes(guide_status).loc[order[::-1], gene_guides]
            
            num_rows, num_cols = vals.shape
        
            heatmap_width = ax_p.height * num_cols / num_rows * height / width
            heatmap_ax = fig.add_axes((start_x, ax_p.y0, heatmap_width, ax_p.height), sharey=axs[ax_order[0]])
                    
            try:
                im = heatmap_ax.imshow(vals, cmap=plt.get_cmap('RdBu_r'), vmin=v_min, vmax=v_max, origin='lower')
            except:
                print(gene)
                raise

            heatmap_ax.xaxis.tick_top()
            heatmap_ax.set_xticks(np.arange(len(gene_guides)))

            heatmap_ax.set_xticklabels([guide for guide in gene_guides], rotation=90)
            
            # Have to jump through weird hoops because label.set_text() doesn't update in-place.
            labels = []
            for label in heatmap_ax.get_xticklabels():
                guide = label.get_text()
                color = guide_to_color(guide)
                label.set_color(color)
                label.set_text(convert_alias(guide))
                labels.append(label)

            heatmap_ax.set_xticklabels(labels)

            for spine in heatmap_ax.spines.values():
                spine.set_visible(False)
                
            start_x += heatmap_width  + heatmap_width * 3 / num_cols * 0.1

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
            cbar_height = heatmap_height * 1 / len(order)
            cbar_width = ax_p.width * 0.75
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
    
    plt.draw()

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
    
    plot_outcome_diagrams(order, pool.target_info,
                          num_outcomes=len(order),
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

    if just_heatmaps:
        for ax in ax_array:
            fig.delaxes(ax)

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
    
    up_rows = data.query('up and not ignore').sort_values('nt_fracs', ascending=False).iloc[:outcomes_to_draw].sort_values('log2_fc', ascending=False)
    if len(up_rows) > 0:
        up_outcomes = up_rows.index.values
        up_height = ax_p.height * 0.55 * len(up_outcomes) / 20
        up_ax = fig.add_axes((ax_p.x1 + ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.55, ax_p.width, up_height))
        
        plot_outcome_diagrams(up_outcomes, pool.target_info,
                              ax=up_ax,
                              window=(-50, 50),
                              draw_all_sequence=False,
                              draw_wild_type_on_top=True,
                             )
        left, right = up_ax.get_xlim()
        
        for y, (outcome, row) in enumerate(up_rows.iterrows()):
            con = ConnectionPatch(xyA=(row['log10_nt_frac'], row['log2_fc']),
                                  xyB=(left, len(up_outcomes) - y - 1),
                                  coordsA='data',
                                  coordsB='data',
                                  axesA=ax,
                                  axesB=up_ax,
                                  color='grey',
                                  alpha=0.15,
                                 )
            ax.add_artist(con)
        
    down_rows = data.query('down').sort_values('nt_fracs', ascending=False).iloc[:outcomes_to_draw].sort_values('log2_fc', ascending=False)
    if len(down_rows) > 0:
        down_outcomes = down_rows.index.values
        down_height = ax_p.height * 0.55 * len(down_outcomes) / 20
        down_ax = fig.add_axes((ax_p.x1 + ax_p.width * 0.1, ax_p.y0 + ax_p.height * 0.45 - down_height, ax_p.width, down_height))

        plot_outcome_diagrams(down_outcomes, pool.target_info,
                              ax=down_ax,
                              window=(-50, 50),
                              draw_all_sequence=False,
                              draw_wild_type_on_top=True,
                             )
        left, right = down_ax.get_xlim()

        for y, (outcome, row) in enumerate(down_rows.iterrows()):
            con = ConnectionPatch(xyA=(row['log10_nt_frac'], row['log2_fc']),
                                xyB=(left, len(down_outcomes) - y - 1),
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
            l.set_text(r'$\geq${}'.format(max_fc))
        elif y == -max_fc:
            l.set_text(r'$\leq${}'.format(-max_fc))
            
    ax.set_yticklabels(labels)

    labels = ax.get_xticklabels()
    for l in labels:
        x, y = l.get_position()
        if x == -min_frequency:
            l.set_text(r'$\leq${}'.format(min_frequency))

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

def big_heatmap(pool_list, genes=None, guides=None, outcomes_list=None, windows=40, cluster_guides=False, cluster_outcomes=False, title=None, layout_kwargs=None):
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
        windows = repeat(windows)

    pool = pool_list[0]
    rows = len(outcomes_list[0])

    if guides is None:
        if genes is None:
            genes = pool.genes
            guides = pool.guides
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
        
        fold_changes = pool.log2_fold_changes('perfect').loc[outcome_order, guides]

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

        colors = pool.log2_fold_changes('perfect').loc[outcome_order, guide_order]
        to_plot = colors.iloc[:, :max_cols]
        rows, cols = to_plot.shape

        nt_fracs = pool.non_targeting_fractions('all')[outcome_order]
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

        if cluster_guides:
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

        _ = plot_outcome_diagrams(outcome_order, pool.target_info,
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

def cluster(pool, num_outcomes, num_guides, metric='correlation'):
    relevant_outcomes = pool.most_frequent_outcomes[:num_outcomes]

    l2_fcs = pool.log2_fold_changes('perfect').loc[relevant_outcomes]

    phenotype_strengths = pool.chi_squared_per_guide(relevant_outcomes)

    guides = phenotype_strengths.index[:num_guides]

    guide_linkage = scipy.cluster.hierarchy.linkage(l2_fcs[guides].T,
                                                    optimal_ordering=True,
                                                    metric=metric,
                                                   )
    guide_dendro = scipy.cluster.hierarchy.dendrogram(guide_linkage,
                                                      no_plot=True,
                                                      labels=guides,
                                                     )

    guide_order = guide_dendro['ivl']
    
    outcome_linkage = scipy.cluster.hierarchy.linkage(l2_fcs[guides],
                                                      optimal_ordering=True,
                                                      metric=metric,
                                                     )
    outcome_dendro = scipy.cluster.hierarchy.dendrogram(outcome_linkage,
                                                        no_plot=True,
                                                        labels=relevant_outcomes,
                                                       )

    outcome_order = outcome_dendro['ivl']

    correlations = l2_fcs[guide_order].corr()

    return guide_order, outcome_order, correlations

def bokeh_heatmap(pool, num_outcomes=50, num_guides=None, cmap='RdBu_r'):
    guide_order, outcome_order, correlations = cluster(pool, num_outcomes, num_guides)

    c_map = plt.get_cmap(cmap)

    normed = (correlations + 1) / 2
    rgba_float = c_map(normed)
    rgba_int = (rgba_float * 255).astype(np.uint8)[::-1] # flip row order for plotting

    size = 1000
    fig = bokeh.plotting.figure(height=size, width=size, active_scroll='wheel_zoom')
    lower_bound = -0.5
    upper_bound = lower_bound + len(guide_order)

    fig.image_rgba(image=[rgba_int], x=lower_bound, y=lower_bound, dw=len(guide_order), dh=(len(guide_order)))

    top_axis = bokeh.models.axes.LinearAxis()
    right_axis = bokeh.models.axes.LinearAxis()
    fig.add_layout(top_axis, 'above')
    fig.add_layout(right_axis, 'right')

    for axis in [fig.yaxis, fig.xaxis]:
        axis.ticker = np.arange(len(guide_order))
        axis.major_label_text_font_size = '0pt'

    fig.xaxis.major_label_overrides = {i: guide for i, guide in enumerate(guide_order)}
    fig.yaxis.major_label_overrides = {i: guide for i, guide in enumerate(guide_order[::-1])}

    fig.xaxis.major_label_orientation = 'vertical'

    fig.x_range = bokeh.models.Range1d(lower_bound, upper_bound, name='x_range', tags=[upper_bound - lower_bound])
    fig.y_range = bokeh.models.Range1d(lower_bound, upper_bound, name='y_range')

    fig.grid.visible = False

    #range_callback = bokeh.models.CustomJS.from_coffeescript(code=Path('heatmap_range.coffee').read_text().format(lower_bound=lower_bound, upper_bound=upper_bound))
    range_callback = bokeh.models.CustomJS.from_coffeescript(code=f'''
models = cb_obj.document._all_models_by_name._dict
size = cb_obj.end - cb_obj.start
font_size = Math.round(3000 / size).toString() + '%'
l.text_font_size = font_size for l in models['labels']
cb_obj.start = {lower_bound} if cb_obj.start < {lower_bound}
cb_obj.end = {upper_bound} if cb_obj.end > {upper_bound}
''')

    for r in (fig.x_range, fig.y_range):
        r.callback = range_callback

    fig.axis.major_tick_line_color = None
    fig.axis.major_tick_out = 0
    fig.axis.major_tick_in = 0
    
    quad_source = bokeh.models.ColumnDataSource(data={
        'left': [],
        'right': [],
        'bottom': [],
        'top': [],
    }, name='quad_source',
    )
    fig.quad('left', 'right', 'top', 'bottom',
             fill_alpha=0,
             line_color='black',
             source=quad_source,
            )
    
    search_code = Path('heatmap_search.coffee').read_text().format(guides=list(guide_order), lower_bound=lower_bound, upper_bound=upper_bound)
    search_callback = bokeh.models.CustomJS.from_coffeescript(search_code)
    text_input = bokeh.models.TextInput(title='Search:', name='search')
    text_input.js_on_change('value', search_callback)

    label_fig_size = 100
    label_figs = {
        'left': bokeh.plotting.figure(width=label_fig_size, height=size, x_range=(-10, 10), y_range=fig.y_range),
        'right': bokeh.plotting.figure(width=label_fig_size, height=size, x_range=(-10, 10), y_range=fig.y_range),
        'top': bokeh.plotting.figure(height=label_fig_size, width=size, y_range=(-10, 10), x_range=fig.x_range),
        'bottom': bokeh.plotting.figure(height=label_fig_size, width=size, y_range=(-10, 10), x_range=fig.x_range),
    }

    fig.min_border = 0
    fig.outline_line_color = None
    fig.axis.axis_line_color = None

    for label_fig in label_figs.values():
        label_fig.outline_line_color = None
        label_fig.axis.visible = False
        label_fig.grid.visible = False
        label_fig.min_border = 0
        label_fig.toolbar_location = None
    
    label_source = bokeh.models.ColumnDataSource(data={
        'text': guide_order,
        'smallest': [0]*len(guide_order),
        'biggest': [label_fig_size]*len(guide_order),
        'ascending': np.arange(len(guide_order)),
        'descending': np.arange(len(guide_order))[::-1],
        'alpha': [1]*len(guide_order),
    }, name='label_source')

    common_kwargs = {
        'text': 'text',
        'text_alpha': 'alpha',
        'text_font_size': '{}%'.format(3000 // (upper_bound - lower_bound)),
        'source': label_source, 
        'text_baseline': 'middle',
        'name': 'labels',
    }
    vertical_kwargs = {
        'y': 'descending',
        'x_units': 'screen',
        'y_units': 'data',
        **common_kwargs,
    }
    horizontal_kwargs = {
        'x': 'ascending',
        'x_units': 'data',
        'y_units': 'screen',
        **common_kwargs,
    }

    labels = {
        'left': bokeh.models.LabelSet(x='biggest',
                                      text_align='right',
                                      x_offset=-5,
                                      **vertical_kwargs,
                                     ),
        'right': bokeh.models.LabelSet(x='smallest',
                                       text_align='left',
                                       x_offset=5,
                                       **vertical_kwargs,
                                      ),
        'top': bokeh.models.LabelSet(y='smallest',
                                     text_align='left',
                                     angle=np.pi / 2,
                                     y_offset=5,
                                     **horizontal_kwargs,
                                    ),
        'bottom': bokeh.models.LabelSet(y='biggest',
                                        text_align='right',
                                        angle=np.pi / 2,
                                        y_offset=-5,
                                        **horizontal_kwargs,
                                       ),
    }

    for which in labels:
        label_figs[which].add_layout(labels[which])
    
    fig.toolbar_location = None
    toolbar = bokeh.models.ToolbarBox(toolbar=fig.toolbar)

    rows = [
        [bokeh.layouts.Spacer(width=label_fig_size), label_figs['top'], bokeh.layouts.Spacer(width=label_fig_size)],
        [label_figs['left'], fig, label_figs['right'], toolbar, text_input],
        [bokeh.layouts.Spacer(width=label_fig_size), label_figs['bottom']],
    ]

    layout = bokeh.layouts.column([bokeh.layouts.row(row) for row in rows])

    bokeh.io.show(layout)
