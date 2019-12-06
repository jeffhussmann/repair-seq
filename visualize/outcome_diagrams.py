import numpy as np
import matplotlib.pyplot as plt

import hits.utilities
import hits.visualize
from knock_knock.target_info import degenerate_indel_from_string, SNVs, effectors
from ddr.pooled_layout import HDROutcome, DeletionOutcome, HDRPlusDeletionOutcome, DeletionPlusMismatchOutcome

def plot(outcome_order,
         target_info,
         num_outcomes=None,
         title=None,
         window=70,
         ax=None,
         flip_if_reverse=True,
         center_at_PAM=False,
         draw_cut_afters=True,
         size_multiple=0.8,
         draw_all_sequence=False,
         draw_imperfect_MH=False,
         draw_wild_type_on_top=False,
         draw_donor_on_top=False,
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
        transform_seq = hits.utilities.complement
        cut_offset_sign = 1
    else:
        flip = False
        transform_seq = hits.utilities.identity
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

            if draw_wild_type_on_top:
                ys = [-0.5, num_outcomes + 0.5]
            else:
                ys = [-0.5, num_outcomes - 0.5]

            ax.plot([x, x], ys, color='black', linestyle='--', alpha=0.5, clip_on=False)
    
    if flip:
        window_left, window_right = -window_right, -window_left

    seq = target_info.target_sequence[offset + window_left:offset + window_right + 1]

    def draw_rect(x0, x1, y0, y1, alpha, color='black', fill=True):
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
                            fill=fill,
                            closed=True,
                            alpha=alpha,
                            color=color,
                            linewidth=0 if fill else 1.5,
                            clip_on=False,
                           )
        ax.add_patch(patch)

    text_size = 8
    block_alpha = 0.1
    wt_height = 0.6

    def draw_sequence(y, xs_to_skip=None, alpha=0.1):
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
                                color=hits.visualize.igv_colors[transform_seq(b)],
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
                                        color=hits.visualize.igv_colors[b],
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

    def draw_donor(y, HDR_outcome, deletion_outcome, on_top=False):
        SNP_ps = sorted(p for (s, p), b in target_info.fingerprints[target_info.target])

        p_to_i = SNP_ps.index
        i_to_p = dict(enumerate(SNP_ps))

        SNP_xs = set()
        observed_SNP_idxs = set()

        for ((strand, position), ref_base), read_base in zip(target_info.fingerprints[target_info.target], HDR_outcome.donor_SNV_read_bases):
            x = position - offset
            if window_left <= x <= window_right:
                # Note: read base of '-' means it was deleted
                if ref_base != read_base and read_base != '_' and read_base != '-':
                    SNP_xs.add(x)
                    observed_SNP_idxs.add(p_to_i(position))

                    ax.annotate(transform_seq(read_base),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                alpha=0.35,
                                #color=hits.visualize.igv_colors[transform_seq(read_base)],
                                #weight='bold',
                                annotation_clip=False,
                                )
            
                if read_base != '-':
                    if  read_base == '_':
                        color = 'grey'
                        alpha = 0.3
                    else:
                        color = hits.visualize.igv_colors[transform_seq(read_base)]
                        alpha = 0.7

                    draw_rect(x - 0.5, x + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

        if not on_top:
            # Draw rectangles around blocks of consecutive incorporated donor SNVs. 
            observed_SNP_idxs = sorted(observed_SNP_idxs)
            if observed_SNP_idxs:
                # no SNPs if just a donor deletion
                blocks = []
                block = [observed_SNP_idxs[0]]

                for i in observed_SNP_idxs[1:]:
                    if block == [] or i == block[-1] + 1:
                        block.append(i)
                    else:
                        blocks.append(block)
                        block = [i]

                blocks.append(block)
                for block in blocks:
                    start = i_to_p[block[0]] - offset
                    end = i_to_p[block[-1]] - offset
                    x_buffer = 0.7
                    y_buffer = 0.7
                    draw_rect(start - x_buffer, end + x_buffer, y - y_buffer * wt_height, y + y_buffer * wt_height, 0.5, fill=False)
        
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

        if on_top:
            strands = set(SNV['strand'] for SNV in target_info.donor_SNVs['donor'].values())
            if len(strands) > 1:
                raise ValueError('donor strand is weird')
            else:
                strand = strands.pop()

            arrow_ys = [y + wt_height * 0.4, y, y - wt_height * 0.4]

            for x in range(window_left, window_right + 1, 1):
                if x in SNP_xs:
                    continue

                if strand == '+':
                    arrow_xs = [x - 0.5, x + 0.5, x - 0.5]
                else:
                    arrow_xs = [x + 0.5, x - 0.5, x + 0.5]

                ax.plot(arrow_xs, arrow_ys,
                        color='black',
                        alpha=0.2,
                        clip_on=False,
                )

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
                                    color=hits.visualize.igv_colors[transform_seq(b)],
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
                                color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
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
            xs_to_skip = draw_deletion(y, outcome.deletion_outcome.deletion, draw_MH=True)
            
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
                                color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                weight='bold',
                            )

                # Draw box around mismatch to distinguish from MH.
                x_buffer = 0.7
                y_buffer = 0.7
                draw_rect(x - x_buffer, x + x_buffer, y - y_buffer * wt_height, y + y_buffer * wt_height, 0.5, fill=False)

            if draw_all_sequence:
                draw_sequence(y, xs_to_skip)

        elif category == 'donor' or category == 'donor + deletion':
            if category == 'donor':
                HDR_outcome = HDROutcome.from_string(details)
                deletion_outcome = None
            elif category == 'donor + deletion':
                HDR_plus_deletion_outcome = HDRPlusDeletionOutcome.from_string(details)
                HDR_outcome = HDR_plus_deletion_outcome.HDR_outcome
                deletion_outcome = HDR_plus_deletion_outcome.deletion_outcome
    
            draw_donor(y, HDR_outcome, deletion_outcome, False)
            
        else:
            label = '{}, {}, {}'.format(category, subcategory, details)
            ax.annotate(label,
                        xy=(0, y),
                        xycoords='data', 
                        ha='center',
                        va='center',
                        size=text_size,
                        )

    if draw_donor_on_top and len(target_info.donor_SNVs['target']) > 0:
        donor_SNV_read_bases = ''.join(d['base'] for name, d in sorted(target_info.donor_SNVs['donor'].items()))
        HDR_outcome = HDROutcome(donor_SNV_read_bases, [])
        draw_donor(num_outcomes + 0.75, HDR_outcome, None, on_top=True)

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
                    xytext=(0, 28),
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

def plot_with_frequencies(pool, outcomes, text_only=False, **kwargs):
    fig = outcome_diagrams.plot(outcomes, pool.target_info, **kwargs)
    num_outcomes = kwargs.get('num_outcomes')
    add_frequencies(fig, fig.axes[0], pool, outcomes[:num_outcomes], text_only=text_only)
    return fig