import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import Counter

def compute_outcome_counts(outcomes):
    relevant_outcomes = outcomes.query('subcategory == "deletion" or subcategory == "insertion" or category == "no indel"')

    counts = Counter(zip(relevant_outcomes['subcategory'], relevant_outcomes['details'])).most_common()
    counts = pd.DataFrame(counts).set_index(0)[1]

    wts = [(sc, d) for sc, d in counts.index if sc == 'wild type']
    counts = pd.concat([counts[wts], counts.drop(wts)])
    return counts

def plot_outcomes(outcomes, target_info, num_outcomes=30, title=None):
    counts = compute_outcome_counts(outcomes)
    
    num_outcomes = min(num_outcomes, len(counts))

    fig, ax = plt.subplots(figsize=(20, num_outcomes / 3))

    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    
    window = 70

    offset = target_info.cut_after
    seq = target_info.target_sequence[offset - window:offset + window + 1]
    guide = target_info.features[target_info.target, target_info.sgRNA]

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
                            clip_on=False,
                            linewidth=0,
                           )
        ax.add_patch(patch)
    
    for i, ((kind, details), count) in enumerate(counts[:num_outcomes].items()):
        if kind == 'deletion':
            starts, length = details.split(':')[1].split(',')
            length = int(length)
            
        elif kind == 'insertion':
            starts, bases = details.split(':')[1].split(',')
            
            if '{' in bases:
                bases = [bs for bs in bases.strip('{}').split('|')]
            else:
                bases = [bases]
                
        else:
            starts = '0'
        
        if '{' in starts:
            starts = [int(s) for s in starts.strip('{}').split('|')]
        else:
            starts = [int(starts)]

        starts = np.array(starts) - offset
        
        y = num_outcomes - i
            
        if kind == 'deletion':
            if len(starts) > 1:
                for x, b in zip(range(-window, window + 1), seq):
                    if (starts[0] <= x < starts[-1]) or (starts[0] + length <= x < starts[-1] + length):
                        ax.annotate(b,
                                    xy=(x, y),
                                    xycoords='data', 
                                    ha='center',
                                    va='center',
                                    size=9,
                                   )

            del_height = 0.15
            wt_height = 0.6
            
            del_start = starts[0] - 0.5
            del_end = starts[0] + length - 1 + 0.5
            
            draw_rect(del_start, del_end, y - del_height / 2, y + del_height / 2, 0.4)
            draw_rect(-window - 0.5, del_start, y - wt_height / 2, y + wt_height / 2, 0.15)
            draw_rect(del_end, window + 0.5, y - wt_height / 2, y + wt_height / 2, 0.15)
        
        elif kind == 'insertion':
            wt_height = 0.6
            draw_rect(-window - 0.5, window + 0.5, y - wt_height / 2, y + wt_height / 2, 0.15)
            for i, (start, bs) in enumerate(zip(starts, bases)):
                ys = [y - 0.3, y + 0.3]
                xs = [start + 0.5, start + 0.5]

                ax.plot(xs, ys, color='purple', linewidth=2, alpha=0.6)
                
                ax.annotate(bs,
                            xy=(start + 0.5, y + (wt_height / 2) - (i * 0.1)),
                            xycoords='data',
                            ha='center',
                            va='top',
                            size=8,
                           )
                
        elif kind == 'wild type':
            wt_height = 0.6
            
            guide_start = guide.start - 0.5 - offset
            guide_end = guide.end + 0.5 - offset
            
            draw_rect(guide_start, guide_end, y - wt_height / 2, y + wt_height / 2, 0.3, color='blue')
    
            if guide.strand == '+':
                PAM_start = guide_end
                draw_rect(-window - 0.5, guide_start, y - wt_height / 2, y + wt_height / 2, 0.15)
                draw_rect(PAM_start + 3, window + 0.5, y - wt_height / 2, y + wt_height / 2, 0.15)
            else:
                PAM_start = guide_start - 3
                draw_rect(-window - 0.5, PAM_start, y - wt_height / 2, y + wt_height / 2, 0.15)
                draw_rect(guide_end, window + 0.5, y - wt_height / 2, y + wt_height / 2, 0.15)

            draw_rect(PAM_start, PAM_start + 3, y - wt_height / 2, y + wt_height / 2, 0.3, color='green')

            for x, b in zip(range(-window, window + 1), seq):
                ax.annotate(b,
                            xy=(x, y),
                            xycoords='data', 
                            ha='center',
                            va='center',
                            size=9,
                           )
                
        elif kind == 'other' or kind == 'donor':
            for ((strand, position), ref_base), read_base in zip(target_info.fingerprints[target_info.target], details):
                if ref_base != read_base:
                    ax.annotate(read_base,
                                xy=(position - offset, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=9,
                               )
                
            wt_height = 0.6
            
            draw_rect(-window - 0.5, window + 0.5, y - wt_height / 2, y + wt_height / 2, 0.15)
                
    ax_p = ax.get_position()
    
    width = ax_p.width * 0.1
    offset = ax_p.width * 0.02
    freq_ax = fig.add_axes((ax_p.x1 + offset, ax_p.y0, width, ax_p.height), sharey=ax)
    cumulative_ax = fig.add_axes((ax_p.x1 + 2 * offset + width, ax_p.y0, width, ax_p.height), sharey=ax)
    
    freqs = counts[:num_outcomes] / counts.sum()
    ys = np.arange(num_outcomes, 0, -1)
    
    freq_ax.plot(freqs, ys, 'o-', markersize=2, color='black')
    cumulative_ax.plot(freqs.cumsum(), ys, 'o-', markersize=2, color='black')
    
    freq_ax.set_xlim(0, max(freqs) * 1.05)
    cumulative_ax.set_xlim(0, 1)
    cumulative_ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    
    for p_ax in [freq_ax, cumulative_ax]:
        p_ax.set_yticks([])
        p_ax.xaxis.tick_top()
        p_ax.spines['left'].set_alpha(0.3)
        p_ax.spines['right'].set_alpha(0.3)
        p_ax.tick_params(labelsize=6)
        p_ax.grid(axis='x', alpha=0.3)
        
        p_ax.spines['bottom'].set_visible(False)
        
        p_ax.xaxis.set_label_position('top')
    
    freq_ax.set_xlabel('Frequency', size=8)
    cumulative_ax.set_xlabel('Cumulative frequency', size=8)

    ax.set_xlim(-window - 0.5, window + 0.5)
    ax.set_ylim(0.5, num_outcomes + 0.5);
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
    ax.axhline(num_outcomes + 0.5, color='black', alpha=0.75, clip_on=False)
    
    return fig