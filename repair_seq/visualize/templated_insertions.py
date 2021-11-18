import bokeh.palettes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
idx = pd.IndexSlice

def plot_length_distributions(pool,
                              guide_combinations_to_plot,
                              species='hg19',
                              smoothing_window=10,
                              draw_legend=True,
                             ):

    def get_length_distributions_for_multiple_guides(pool, species, guide_pairs):
        # counts for each specified guide
        length_counts = pool.genomic_insertion_length_counts.loc[species].loc[guide_pairs]

        # total counts at each length across all specified guides
        all_guide_length_counts = length_counts.sum(axis='rows')

        # total UMIs across all specified guides
        total_UMIs = pool.UMI_counts_for_all_fixed_guides().loc[guide_pairs].sum()
        
        all_guide_length_fractions = all_guide_length_counts / total_UMIs

        smoothed_fractions = all_guide_length_fractions.rolling(2 * smoothing_window + 1, center=True, min_periods=1).mean()

        return all_guide_length_fractions, smoothed_fractions

    relevant_lengths = idx[25:pool.blunt_insertion_length_detection_limit]

    upper_limit = pool.blunt_insertion_length_detection_limit
    x_for_above_limit = upper_limit + 75
    x_max = x_for_above_limit + 25

    fig_width = 0.0075 * x_max
    axs = {}
    fig, (axs['frequency'], axs['log2_fc']) = plt.subplots(2, 1,
                                                           figsize=(fig_width, 3),
                                                           gridspec_kw=dict(hspace=0.4,
                                                                            height_ratios=[3, 2],
                                                                           ),
                                                          )

    axs['frequency_high'] = axs['frequency'].twinx()

    nt_fs, smoothed_nt_fs = get_length_distributions_for_multiple_guides(pool, species, pool.non_targeting_guide_pairs)

    for label, guides, color, line_width, alpha in guide_combinations_to_plot:
        fs, smoothed_fs = get_length_distributions_for_multiple_guides(pool, species, guides)
        
        axs['frequency'].plot(smoothed_fs.loc[relevant_lengths] * 100, color=color, alpha=alpha, linewidth=line_width, label=label)
        
        axs['frequency_high'].plot(x_for_above_limit, fs.iloc[-1] * 100, 'o', markersize=2, color=color)
        
        l2fcs = np.log2(smoothed_fs / smoothed_nt_fs)

        axs['log2_fc'].plot(np.maximum(-5, l2fcs.loc[relevant_lengths]), color=color, alpha=alpha, linewidth=line_width)
        
        fc = fs.iloc[-1] / nt_fs.iloc[-1]
        log2_fc = np.log2(fc)
        
        axs['log2_fc'].plot(x_for_above_limit, log2_fc, 'o', markersize=2, color=color)
        
    for ax in (axs['log2_fc'],):
        min_fc = -4
        max_fc = 4
        
        ax.set_ylim(min_fc, max_fc)
        for fc in np.arange(min_fc, max_fc + 1):
            ax.axhline(fc, color='black', alpha=0.1)
            
    for ax in axs.values():
        ax.set_xlim(0, x_max)

    axs['frequency'].set_ylabel('Percentage of\nrepair outcomes', size=7)
    axs['log2_fc'].set_ylabel('Log$_2$ fold change\nfrom non-targeting', size=7)
    axs['log2_fc'].set_yticks(np.arange(-4, 6, 2))

    for ax in axs.values():
        main_ticks = list(range(0, upper_limit, 50))
        main_tick_labels = [f'{x:,}' for x in main_ticks]

        extra_ticks = [x_for_above_limit] + list(range(25, upper_limit, 50))
        extra_tick_labels = ['longer than\ndetection limit'] + ['']*len(list(range(25, upper_limit, 50)))

        ax.set_xticks(main_ticks + extra_ticks)
        ax.set_xticklabels(main_tick_labels + extra_tick_labels, size=6)
        
        ax.axvline(25, color='black', alpha=0.2)
        ax.axvline(pool.blunt_insertion_length_detection_limit, color='black', alpha=0.2)

    axs['frequency_high'].annotate(
        'upper\ndetection limit',
        xy=(pool.blunt_insertion_length_detection_limit, 1),
        xycoords=('data', 'axes fraction'),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        size=6,
    )

    axs['frequency_high'].annotate(
        'lower\ndetection limit',
        xy=(25, 1),
        xycoords=('data', 'axes fraction'),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        size=6,
    )

    axs['log2_fc'].set_xlabel('Length of captured\ngenomic sequence', size=7)

    axs['log2_fc'].axhline(0, color=bokeh.palettes.Greys9[2])

    axs['frequency_high'].spines['bottom'].set_visible(False)
    axs['frequency'].spines['top'].set_visible(False)

    axs['frequency'].set_ylim(0, 0.072)
    axs['frequency_high'].set_ylim(0, 6)

    if draw_legend:
        axs['frequency'].legend(bbox_to_anchor=(0.5, 1.25),
                                loc='lower center',
                                fontsize=6,
                            )

    for ax in axs.values():
        ax.tick_params(labelsize=6, width=0.5)
        plt.setp(ax.spines.values(), linewidth=0.5)

    return fig
