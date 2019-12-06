from itertools import starmap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.stats.proportion
import scipy.stats

from matplotlib.patches import ConnectionPatch

import hits.visualize
import hits.utilities

from . import outcome_diagrams

def outcome(outcome,
            pool,
            p_cutoff=4,
            num_to_label=20,
            genes_to_label=None,
            guides_to_label=None,
            fraction_lims=None,
            num_cells_lims=None,
            p_val_method='binomial',
            outcome_name=None,
            guide_status='perfect',
            guide_subset=None,
            draw_marginals='hist',
            draw_diagram=True,
            gene_to_color=None,
            color_labels=False,
            initial_label_offset=7,
            only_best_promoter=False,
            guide_aliases=None,
            big_labels=False,
            as_percentage=False,
            non_targeting_label_location='floating',
            avoid_label_overlap=True,
            flip_axes=False,
           ):
    if guide_aliases is None:
        guide_aliases = {}

    if big_labels:
        axis_label_size = 20
        tick_label_size = 14
        big_guide_label_size = 18
        small_guide_label_size = 8
    else:
        axis_label_size = 12
        tick_label_size = 10
        big_guide_label_size = 12
        small_guide_label_size = 10

    fraction_scaling_factor = 100 if as_percentage else 1
    fraction_key = 'percentage' if as_percentage else 'fraction'

    if flip_axes:
        axis_to_quantity = {
            'y': 'num_cells',
            'x': fraction_key,
        }
    else:
        axis_to_quantity = {
            'x': 'num_cells',
            'y': fraction_key,
        }

    quantity_to_axis = hits.utilities.reverse_dictionary(axis_to_quantity)

    max_cells = max(pool.UMI_counts(guide_status))

    granular_df = pool.outcome_counts(guide_status)
    non_targeting = pool.guide_library.non_targeting_guides

    if isinstance(outcome, tuple):
        nt_fraction = pool.non_targeting_fractions(guide_status)[outcome]
        outcome_counts = granular_df.loc[outcome]

    else:
        nt_counts = pool.non_targeting_counts(guide_status)
        nt_fraction = nt_counts[outcome].sum() / nt_counts.sum()
        outcome_counts = granular_df.loc[outcome].sum()

    if p_val_method == 'binomial':
        boundary_cells = np.arange(0, 60000)
        boundary_cells[0] = 1
        lower, upper = scipy.stats.binom.interval(1 - 2 * 10**-p_cutoff, boundary_cells, nt_fraction)
        boundary_lower = lower / boundary_cells
        boundary_upper = upper / boundary_cells

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
    df['percentage'] = df['fraction'] * 100

    df['gene'] = pool.guide_library.guides_df.loc[df.index]['gene']
    
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

    df['significant'] = df['pval'] <= 10**-p_cutoff

    df['color'] = 'silver'
    df['label_color'] = 'black'
    df.loc[df.query('significant').index, 'color'] = 'C1'
    df.loc[non_targeting, 'color'] = 'C0'

    if gene_to_color is not None:
        for gene, color in gene_to_color.items():
            guides = df.index.intersection(pool.gene_guides(gene, only_best_promoter))
            df.loc[guides, 'color'] = color
            df.loc[guides, 'label_color'] = color

    if fraction_lims is None:
        fraction_max = df.query('significant')['fraction'].max() * 1.1 * fraction_scaling_factor
        fraction_min = df.query('significant')['fraction'].min() * 0.9 * fraction_scaling_factor
        if fraction_min < 0.1 * fraction_scaling_factor:
            fraction_min = 0
    else:
        fraction_min, fraction_max = fraction_lims

    if num_cells_lims is None:
        num_cells_min, num_cells_max = (max(0, -0.01 * max_cells), 1.05 * max_cells)
    else:
        num_cells_min, num_cells_max = num_cells_lims

    lims = {
        'num_cells': (num_cells_min, num_cells_max),
        fraction_key: (fraction_min, fraction_max),
    }

    g = sns.JointGrid(x=axis_to_quantity['x'],
                      y=axis_to_quantity['y'],
                      data=df,
                      height=10,
                      xlim=lims[axis_to_quantity['x']],
                      ylim=lims[axis_to_quantity['y']],
                      space=0,
                      ratio=8,
                     )

    if draw_marginals:

        marg_axs_by_axis = {
            'x': g.ax_marg_x,
            'y': g.ax_marg_y,
        }

        marg_axs = {
            'num_cells': marg_axs_by_axis[quantity_to_axis['num_cells']],
            fraction_key: marg_axs_by_axis[quantity_to_axis[fraction_key]],
        }

        if draw_marginals == 'kde':

            vertical = {k: axis == 'y' for k, axis in quantity_to_axis.items()}

            fraction_kwargs = dict(
                ax=marg_axs[fraction_key],
                vertical=vertical[fraction_key],
                legend=False,
                linewidth=2,
                shade=True,
            )
            sns.kdeplot(df[fraction_key], color='grey', **fraction_kwargs)
            sns.kdeplot(df.query('gene == "negative_control"')[fraction_key], color='C0', alpha=0.4, **fraction_kwargs)

            num_cells_kwargs = dict(
                ax=marg_axs['num_cells'],
                vertical=vertical['num_cells'],
                legend=False,
                linewidth=1.5,
                shade=True,
            )
            sns.kdeplot(df['num_cells'], color='grey', **num_cells_kwargs)
            #sns.kdeplot(df.query('gene == "negative_control"')['num_cells'], color='C0', **num_cells_kwargs)

        elif draw_marginals == 'hist':
            orientation = {k: 'horizontal' if axis == 'y' else 'vertical' for k, axis in quantity_to_axis.items()}

            bins = np.linspace(fraction_min, fraction_max, 100)
            hist_kwargs = dict(orientation=orientation[fraction_key], alpha=0.5, density=True, bins=bins, linewidth=2)
            marg_axs[fraction_key].hist(fraction_key, data=df, color='silver', **hist_kwargs)
            #g.ax_marg_y.hist(y_key, data=df.query('gene == "negative_control"'), color='C0', **hist_kwargs)

            bins = np.linspace(0, max_cells, 30)
            hist_kwargs = dict(orientation=orientation['num_cells'], alpha=0.5, density=True, bins=bins, linewidth=2)
            marg_axs['num_cells'].hist('num_cells', data=df, color='silver', **hist_kwargs)
            #g.ax_marg_x.hist('num_cells', data=df.query('gene == "negative_control"'), color='C0', **hist_kwargs)

        median_UMIs = int(df['num_cells'].median())

        if quantity_to_axis['num_cells'] == 'x':
            line_func = marg_axs['num_cells'].axvline
            annotate_kwargs = dict(
                s=f'median = {median_UMIs:,} cells / guide',
                xy=(median_UMIs, 1.1),
                xycoords=('data', 'axes fraction'),
                xytext=(3, 0),
                va='top',
                ha='left',
            )

        else:
            line_func = marg_axs['num_cells'].axhline
            annotate_kwargs = dict(
                s=f'median = {median_UMIs:,}\ncells / guide',
                xy=(1, median_UMIs),
                xycoords=('axes fraction', 'data'),
                xytext=(-5, 5),
                va='bottom',
                ha='center',
            )

        line_func(median_UMIs, color='black', linestyle='--', alpha=0.7)
        marg_axs['num_cells'].annotate(textcoords='offset points',
                                       size=12,
                                       **annotate_kwargs,
                                      )

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
        
        cells = boundary_cells[1:]

        for fractions in [boundary_lower, boundary_upper]:
            vals = {
                'num_cells': cells,
                fraction_key: fractions[cells] * fraction_scaling_factor,
            }

            xs = vals[axis_to_quantity['x']]
            ys = vals[axis_to_quantity['y']]
            
            g.ax_joint.plot(xs, ys, color='black', alpha=0.3)

        # Annotate the lines with their significance level.
        x = int(np.floor(1.01 * max_cells))

        cells = int(np.floor(0.8 * num_cells_max))
        vals = {
            'num_cells': cells,
            fraction_key: boundary_upper[cells] * fraction_scaling_factor,
        }
        x = vals[axis_to_quantity['x']]
        y = vals[axis_to_quantity['y']]

        if flip_axes:
            annotate_kwargs = dict(
                xytext=(20, 0),
                ha='left',
                va='center',
            )
        else:
            annotate_kwargs = dict(
                xytext=(0, 30),
                ha='center',
                va='bottom',
            )

        g.ax_joint.annotate(f'p = $10^{{-{p_cutoff}}}$ significance threshold',
                            xy=(x, y),
                            xycoords='data',
                            textcoords='offset points',
                            color='black',
                            size=10,
                            arrowprops={'arrowstyle': '-',
                                        'alpha': 0.5,
                                        'color': 'black',
                                        },
                            **annotate_kwargs,
                        )
        
    if non_targeting_label_location is None:
        pass
    else:
        if non_targeting_label_location == 'floating':
            floating_labels = [
                ('individual non-targeting guides', 20, 'C0'),
                #('{} p-value < 1e-{}'.format(p_val_method, p_cutoff), -25, 'C1'),
            ]
            for text, offset, color in floating_labels:
                g.ax_joint.annotate(text,
                                    xy=(1, nt_fraction * fraction_scaling_factor),
                                    xycoords=('axes fraction', 'data'),
                                    xytext=(-5, offset),
                                    textcoords='offset points',
                                    color=color,
                                    ha='right',
                                    va='bottom',
                                    size=14,
                                )

        else:
            highest_nt_guide = df.query('gene == "negative_control"').sort_values('num_cells').iloc[-1]

            vals = {
                'num_cells': highest_nt_guide['num_cells'],
                fraction_key: highest_nt_guide[fraction_key],
            }
            x = vals[axis_to_quantity['x']]
            y = vals[axis_to_quantity['y']]

            if flip_axes:
                annotate_kwargs = dict(
                    s='individual non-targeting guides',
                    xytext=(-40, 0),
                    ha='right',
                    va='center',
                )
            else:
                annotate_kwargs = dict(
                    s='individual\nnon-targeting guides',
                    xytext=(0, 30),
                    ha='center',
                    va='bottom',
                )

            g.ax_joint.annotate(xy=(x, y),
                                xycoords='data',
                                textcoords='offset points',
                                color='C0',
                                size=10,
                                arrowprops={'arrowstyle': '-',
                                            'alpha': 0.5,
                                            'color': 'C0',
                                            },
                                **annotate_kwargs,
                            )
            
            vals = {
                'num_cells': num_cells_max * 0.9,
                fraction_key: nt_fraction * fraction_scaling_factor,
            }
            x = vals[axis_to_quantity['x']]
            y = vals[axis_to_quantity['y']]

            if flip_axes:
                annotate_kwargs = dict(
                    s='average of all non-targeting guides',
                    xytext=(-40, 0),
                    ha='right',
                    va='center',
                )
            else:
                annotate_kwargs = dict(
                    s='average of all\nnon-targeting guides',
                    xytext=(0, 30),
                    ha='center',
                    va='bottom',
                )
    
            g.ax_joint.annotate(xy=(x, y),
                                xycoords='data',
                                textcoords='offset points',
                                color='black',
                                size=10,
                                arrowprops={'arrowstyle': '-',
                                            'alpha': 0.5,
                                            'color': 'black',
                                            },
                                **annotate_kwargs,
                            )

    g.ax_joint.scatter(x=axis_to_quantity['x'],
                       y=axis_to_quantity['y'],
                       data=df,
                       s=25,
                       alpha=0.9,
                       color='color',
                       linewidths=(0,),
                      )

    if quantity_to_axis[fraction_key] == 'y':
        line_func = g.ax_joint.axhline
    else:
        line_func = g.ax_joint.axvline

    line_func(nt_fraction * fraction_scaling_factor, color='black')

    axis_label_funcs = {
        'x': g.ax_joint.set_xlabel,     
        'y': g.ax_joint.set_ylabel,     
    }

    num_cells_label = 'number of cells per CRISPRi guide'
    axis_label_funcs[quantity_to_axis['num_cells']](num_cells_label, size=axis_label_size)

    if outcome_name is None:
        try:
            outcome_name = '_'.join(outcome)
        except TypeError:
            outcome_name = 'PH'

    fraction_label = f'{fraction_key} of CRISPRi-guide-containing cells with {outcome_name}'
    axis_label_funcs[quantity_to_axis[fraction_key]](fraction_label, size=axis_label_size)

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
        outcome_diagrams.plot(outcomes_to_plot,
                              pool.target_info,
                              window=(-50, 20),
                              ax=diagram_ax,
                              flip_if_reverse=True,
                              draw_all_sequence=False,
                              draw_wild_type_on_top=True,
                             )

    if num_to_label is not None:
        up = df.query('significant and direction == "up"').sort_values('fraction', ascending=False)[:num_to_label]
        down = df.query('significant and direction == "down"').sort_values('fraction')[:num_to_label]
        to_label = pd.concat([up, down])

        # Don't label any points that will be labeled by gene-labeling below.
        if genes_to_label is not None:
            to_label = to_label[~to_label['gene'].isin(genes_to_label)]

        to_label = to_label.query(f'{fraction_key} >= @fraction_min and {fraction_key} <= @fraction_max')

        if flip_axes:
            vector = ['upper right' if v == 'up' else 'upper left' for v in to_label['direction']]
        else:
            vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        hits.visualize.label_scatter_plot(g.ax_joint,
                                          axis_to_quantity['x'],
                                          axis_to_quantity['y'],
                                          'alias',
                                          color='label_color',
                                          data=to_label,
                                          vector=vector,
                                          text_kwargs=dict(size=small_guide_label_size),
                                          initial_distance=5,
                                          distance_increment=5,
                                          arrow_alpha=0.2,
                                          avoid=avoid_label_overlap,
                                          avoid_axis_labels=True,
                                         )

    if guides_to_label is not None or genes_to_label is not None:
        if guides_to_label is not None:
            to_label = df[df.index.isin(guides_to_label)]
        elif genes_to_label is not None:
            to_label = df[df['gene'].isin(genes_to_label)]

        to_label = to_label.query(f'{fraction_key} >= @fraction_min and {fraction_key} <= @fraction_max')

        if flip_axes:
            vector = ['upper right' if v == 'up' else 'upper left' for v in to_label['direction']]
        else:
            vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        if color_labels:
            kwargs = dict(color='color',
                         )
        else:
            kwargs = dict(color=None,
                         )

        hits.visualize.label_scatter_plot(g.ax_joint,
                                          axis_to_quantity['x'],
                                          axis_to_quantity['y'],
                                          'alias',
                                          data=to_label,
                                          vector=vector,
                                          initial_distance=initial_label_offset,
                                          distance_increment=5,
                                          arrow_alpha=0.2,
                                          avoid=avoid_label_overlap,
                                          avoid_axis_labels=True,
                                          avoid_existing=True,
                                          min_arrow_distance=0,
                                          text_kwargs=dict(size=big_guide_label_size),
                                          **kwargs,
                                         )

    hits.visualize.add_commas_to_ticks(g.ax_joint, which=quantity_to_axis['num_cells'])
    g.ax_joint.tick_params(labelsize=tick_label_size)

    if not draw_marginals:
        g.fig.delaxes(g.ax_marg_x)
        g.fig.delaxes(g.ax_marg_y)

    return g, df


def guide(pool, gene, number, ax=None, outcomes_to_draw=15, subset_query=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = ax.get_figure()

    if subset_query is None:
        subset_query = 'significant'

    fig_width, fig_height = fig.get_size_inches()

    max_fc = 6 
    min_frequency = 5

    guide = pool.guide_library.gene_guides(gene)[number]

    nt_fracs = pool.non_targeting_fractions('perfect')
    fracs = pool.outcome_fractions('perfect')[guide]
    counts = pool.outcome_counts('perfect')[guide]
    ratios = pool.fold_changes('perfect')[guide]

    num_UMIs = pool.UMI_counts('perfect')[guide]

    ratios[nt_fracs == 0] = 2**max_fc
    ratios[fracs == 0] = 2**-max_fc

    ratios = np.minimum(2**max_fc, ratios)
    ratios = np.maximum(2**-max_fc, ratios)

    data = {
        'fracs': fracs,
        'counts': counts,
        'nt_fracs': nt_fracs,
        'log2_fc': np.log2(ratios),
        'log10_nt_frac': np.log10(np.maximum(10**-6, nt_fracs)),
    }
    data = pd.DataFrame(data)
    data = data.query('nt_fracs >= 1e-5').copy()

    data['lower'], data['upper'] = statsmodels.stats.proportion.proportion_confint(data['counts'], num_UMIs, method='beta')
    data['log2_fc_lower'] = np.log2(np.maximum(data['lower'] / data['nt_fracs'], 2**-(max_fc + 0.1)))
    data['log2_fc_upper'] = np.log2(np.minimum(data['upper'] / data['nt_fracs'], 2**(max_fc + 0.1)))

    data.index.name = 'outcome'

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

    data['significant_up'] = data['log2_fc_lower'] > 0
    data['significant_down'] = data['log2_fc_upper'] < 0
    data['significant'] = data['significant_up'] | data['significant_down']

    data['up'] = data['fracs'] > data['nt_fracs']
    data['down'] = data['fracs'] < data['nt_fracs']

    data.loc[data['significant_down'], 'color'] = 'C0'
    data.loc[data['significant_up'], 'color'] = 'C3'

    if subset_query is not None:
        data['alpha'] = 0.2
        data.loc[data.query(subset_query).index, 'alpha'] = 1
    else:
        data['alpha'] = 0.8

    data['color'] = [hits.visualize.apply_alpha(c, a) for c, a in zip(data['color'], data['alpha'])]

    ax.scatter('log10_nt_frac', 'log2_fc',
               c='color',
               s=50,
               data=data,
               linewidths=(0,),
              )

    # Plot binomial confidence intervals.
    for _, row in data.query(subset_query).iterrows():
        x = row['log10_nt_frac']
        ax.plot([x, x], [row['log2_fc_lower'], row['log2_fc_upper']], color='black', alpha=0.2)

    ax.axhline(0, color='black', alpha=0.8)

    for cutoff in [0.05]:
        ps = np.logspace(-7, 0, 10000)
        lows, highs = scipy.stats.binom.interval(1 - cutoff, num_UMIs, ps)
        interval_xs = np.log10(ps)
        high_ys = np.log2(np.maximum(2**-6, (highs / num_UMIs) / ps))
        low_ys = np.log2(np.maximum(2**-6, (lows / num_UMIs) / ps))

        for ys in [low_ys, high_ys]:
            ax.plot(interval_xs, ys, color='grey', alpha=0.75)

    # Draw line representing the fc that would be achieved by observing a single count.
    nt_fracs = np.logspace(-6, 0)
    ys = np.log2((1. / num_UMIs) / nt_fracs)
    ax.plot(np.log10(nt_fracs), ys, linestyle='--', color='black', linewidth=2)

    ax.set_xlim(-min_frequency - 0.1, data['log10_nt_frac'].max() + 0.1)
    ax.set_ylim(-max_fc - 0.1, max_fc + 0.1)

    ax.set_xlabel('log10(fraction in non-targeting)', size=16)
    ax.set_ylabel('log2(fold change upon knockdown)', size=16)

    ax.set_yticks(np.arange(-max_fc, max_fc + 1, 1))

    ax.set_title(guide, size=16)

    plt.draw()
    ax_p = ax.get_position()
    
    for direction in ['up', 'down']:
        if direction == 'up':
            label = 'gene activity suppresses outcome'
            color = 'C3'
        else:
            label = 'gene activity promotes outcome'
            color = 'C0'

        rows = data.query(f'{direction} and {subset_query}').sort_values('log2_fc', ascending=False).iloc[:outcomes_to_draw]
        if len(rows) > 0:
            diagram_window = (-30, 30)
            outcomes = rows.index.values
            height = ax_p.height * 0.55 * len(outcomes) / 20
            width = ax_p.width * (10 / fig_width) * (diagram_window[1] - diagram_window[0] + 1) / 100

            if direction == 'up':
                y0 = ax_p.y0 + ax_p.height * 0.55
            else:
                y0 = ax_p.y0 + ax_p.height * 0.45 - height

            diagram_ax = fig.add_axes((ax_p.x1 + width * 0.1, y0, width, height))
            
            outcome_diagrams.plot(outcomes, pool.target_info,
                                  ax=diagram_ax,
                                  window=diagram_window,
                                  draw_all_sequence=False,
                                  draw_wild_type_on_top=True,
                                 )
            left, right = diagram_ax.get_xlim()
            
            for y, (outcome, row) in enumerate(rows.iterrows()):
                con = ConnectionPatch(xyA=(row['log10_nt_frac'], row['log2_fc']),
                                    xyB=(left, len(outcomes) - y - 1),
                                    coordsA='data',
                                    coordsB='data',
                                    axesA=ax,
                                    axesB=diagram_ax,
                                    color=color,
                                    alpha=0.25,
                                    )
                ax.add_artist(con)

            text_kwargs = dict(
                xycoords='axes fraction',
                xytext=(0, 20),
                textcoords='offset points',
                xy=(0.5, 1),
                size=14,
                ha='center',
                va='bottom',
            )

            diagram_ax.annotate(label, color=color, **text_kwargs)
        
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
    
    return fig, data

def gene(pool, gene, subset_query=None):
    guides = pool.guide_library.gene_guides(gene)

    fig, axs = plt.subplots(len(guides), 1, figsize=(10, 10 * len(guides)))
    if len(guides) == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        plot_guide_scatter(pool, gene, i, ax=ax, subset_query=subset_query)

    return fig