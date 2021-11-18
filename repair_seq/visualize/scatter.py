from itertools import starmap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
import bokeh.palettes

from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D

import hits.visualize
import hits.utilities

from repair_seq.pooled_screen import ALL_NON_TARGETING
from . import outcome_diagrams

def outcome(outcome,
            pool,
            fixed_guide='none',
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
            draw_negative_control_marginal=False,
            draw_diagram=True,
            gene_to_color=None,
            color_labels=False,
            initial_label_offset=7,
            only_best_promoter=False,
            guide_aliases=None,
            big_labels=False,
            as_percentage=False,
            non_targeting_label_location='floating',
            avoid_label_overlap=False,
            flip_axes=False,
            draw_intervals=True,
            nt_guide_color='C0',
            nt_fraction=None,
            fig_size=10,
            marker_size=25,
            label_median_UMIs=True,
            use_high_frequency_counts=False,
            **kwargs,
           ):
    if guide_aliases is None:
        guide_aliases = {}

    if gene_to_color is None:
        gene_to_color = {}
    gene_to_color = dict(gene_to_color)

    if big_labels:
        axis_label_size = 18
        tick_label_size = 14
        big_guide_label_size = kwargs.get('big_guide_label_size', 10)
        small_guide_label_size = 8
    else:
        axis_label_size = 12
        tick_label_size = 10
        big_guide_label_size = kwargs.get('big_guide_label_size', 10)
        small_guide_label_size = 6

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

    if use_high_frequency_counts:
        UMI_counts = pool.UMI_counts_from_high_frequency_counts

        if isinstance(outcome, tuple):
            outcome_counts = pool.high_frequency_outcome_counts.loc[outcome]
            nt_fraction = pool.high_frequency_outcome_fractions.loc[outcome, ALL_NON_TARGETING]
        else:
            outcome_counts = pool.high_frequency_outcome_counts.loc[outcome].sum()
            nt_fraction = pool.high_frequency_outcome_fractions.loc[outcome, ALL_NON_TARGETING].sum()

    else:
        UMI_counts = pool.UMI_counts_for_all_fixed_guides(guide_status=guide_status)

        granular_df = pool.outcome_counts(guide_status=guide_status).xs(fixed_guide, level=0, axis=1, drop_level=False)

        if isinstance(outcome, tuple):
            nt_fraction = pool.non_targeting_fractions(guide_status=guide_status, fixed_guide=fixed_guide)[outcome]
            outcome_counts = granular_df.loc[outcome]

        else:
            nt_counts = pool.non_targeting_counts(guide_status, fixed_guide)
            nt_fraction = nt_counts[outcome].sum() / nt_counts.sum()
            outcome_counts = granular_df.loc[outcome].sum()

    max_cells = max(UMI_counts)

    if p_val_method == 'binomial':
        boundary_cells = np.arange(0, 60000)
        boundary_cells[0] = 1
        lower, upper = scipy.stats.binom.interval(1 - 2 * 10**-p_cutoff, boundary_cells, nt_fraction)
        boundary_lower = lower / boundary_cells
        boundary_upper = upper / boundary_cells

    data = {
        'num_cells': UMI_counts,
        'count': outcome_counts,
    }

    df = pd.DataFrame(data).fillna(0)
    df.index.name = 'guide'
    df['fraction'] = df['count'] / df['num_cells']
    df['percentage'] = df['fraction'] * 100

    if use_high_frequency_counts:
        df['fixed_gene'] = 'none'
        df['variable_gene'] = [pool.variable_guide_library.guide_to_gene[v_g] for v_g in df.index.values]
        df['variable_guide_best_promoter'] = [pool.variable_guide_library.guides_df.loc[v_g, 'best_promoter'] for v_g in df.index.values]
    else:
        df['fixed_gene'] = [pool.fixed_guide_library.guide_to_gene[f_g] for f_g, v_g in df.index.values]
        df['variable_gene'] = [pool.variable_guide_library.guide_to_gene[v_g] for f_g, v_g in df.index.values]
        df['variable_guide_best_promoter'] = [pool.variable_guide_library.guides_df.loc[v_g, 'best_promoter'] for f_g, v_g in df.index.values]

    if guide_subset is None:
        guide_subset = df.index

    df = df.loc[guide_subset]
    
    def convert_alias(guide):
        for old, new in guide_aliases.items(): 
            if old in guide:
                guide = guide.replace(old, new)

        return guide

    if isinstance(df.index, pd.MultiIndex):
        if np.array_equal(pool.fixed_guide_library.guides, ['none']):
            df['alias'] = [convert_alias(v) for f, v in df.index]
        else:
            df['alias'] = [f'{convert_alias(f)}-{convert_alias(v)}' for f, v in df.index.values]
    else:
        df['alias'] = [convert_alias(g) for g in df.index]

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

    if use_high_frequency_counts:
        nt_guide_pairs = pool.variable_guide_library.non_targeting_guides
    else:
        nt_guide_pairs = [(fg, vg) for fg, vg in pool.guide_combinations if fg == fixed_guide and vg in pool.variable_guide_library.non_targeting_guides]

    df.loc[nt_guide_pairs, 'color'] = nt_guide_color

    if gene_to_color is not None:
        for gene, color in gene_to_color.items():
            query = '(fixed_gene == @gene or variable_gene == @gene)'
            if only_best_promoter:
                query += ' and variable_guide_best_promoter'
            gene_guides = df.query(query).index
            df.loc[gene_guides, 'color'] = color
            df.loc[gene_guides, 'label_color'] = color

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

    grid = sns.JointGrid(x=axis_to_quantity['x'],
                         y=axis_to_quantity['y'],
                         data=df,
                         height=fig_size,
                         xlim=lims[axis_to_quantity['x']],
                         ylim=lims[axis_to_quantity['y']],
                         space=kwargs.get('JointGrid_space', 0.5),
                         ratio=kwargs.get('JointGrid_ratio', 4),
                        )

    if draw_marginals:

        marg_axs_by_axis = {
            'x': grid.ax_marg_x,
            'y': grid.ax_marg_y,
        }

        marg_axs = {
            'num_cells': marg_axs_by_axis[quantity_to_axis['num_cells']],
            fraction_key: marg_axs_by_axis[quantity_to_axis[fraction_key]],
        }

        if draw_marginals == 'kde':
            fraction_kwargs = dict(
                ax=marg_axs[fraction_key],
                legend=False,
                linewidth=0,
                shade=True,
            )

            values = df[fraction_key]
            fraction_kwargs[quantity_to_axis[fraction_key]] = values

            sns.kdeplot(color='grey', **fraction_kwargs)

            if draw_negative_control_marginal:
                values = df.query('variable_gene == "negative_control"')[fraction_key]
                fraction_kwargs[quantity_to_axis[fraction_key]] = values

                sns.kdeplot(color=nt_guide_color, alpha=0.5, **fraction_kwargs)

            num_cells_kwargs = dict(
                ax=marg_axs['num_cells'],
                legend=False,
                linewidth=0,
                shade=True,
                color='grey',
            )

            values = df['num_cells']
            num_cells_kwargs[quantity_to_axis['num_cells']] = values

            sns.kdeplot(**num_cells_kwargs)

        elif draw_marginals == 'hist':
            orientation = {k: 'horizontal' if axis == 'y' else 'vertical' for k, axis in quantity_to_axis.items()}

            bins = np.linspace(fraction_min, fraction_max, 100)
            hist_kwargs = dict(orientation=orientation[fraction_key], alpha=0.5, density=True, bins=bins, linewidth=2)
            marg_axs[fraction_key].hist(fraction_key, data=df, color='silver', **hist_kwargs)
            if draw_negative_control_marginal:
                marg_axs[fraction_key].hist(fraction_key, data=df.query('variable_gene == "negative_control"'), color=nt_guide_color, **hist_kwargs)

            bins = np.linspace(0, max_cells, 30)
            hist_kwargs = dict(orientation=orientation['num_cells'], alpha=0.5, density=True, bins=bins, linewidth=2)
            marg_axs['num_cells'].hist('num_cells', data=df, color='silver', **hist_kwargs)

        if label_median_UMIs:
            median_UMIs = int(df['num_cells'].median())

            if quantity_to_axis['num_cells'] == 'x':
                line_func = marg_axs['num_cells'].axvline
                annotate_kwargs = dict(
                    text=f'median = {median_UMIs:,} UMIs / guide',
                    xy=(median_UMIs, 1.1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(3, 0),
                    va='top',
                    ha='left',
                )

            else:
                line_func = marg_axs['num_cells'].axhline
                annotate_kwargs = dict(
                    text=f'median = {median_UMIs:,}\nUMIs / guide',
                    xy=(0, median_UMIs),
                    xycoords=('axes fraction', 'data'),
                    xytext=(5, 5),
                    va='bottom',
                    ha='left',
                )

            line_func(median_UMIs, color='black', linestyle='--', alpha=0.7)
            marg_axs['num_cells'].annotate(textcoords='offset points',
                                        size=8,
                                        **annotate_kwargs,
                                        )

    if draw_intervals:
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
                
                grid.ax_joint.plot(xs, ys, color='black', alpha=0.3)

            # Annotate the lines with their significance level.
            x = int(np.floor(1.01 * max_cells))

            cells = int(np.floor(0.85 * num_cells_max))

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

            grid.ax_joint.annotate(f'p = $10^{{-{p_cutoff}}}$\nsignificance\nthreshold',
                                xy=(x, y),
                                xycoords='data',
                                textcoords='offset points',
                                color='black',
                                size=7,
                                arrowprops={'arrowstyle': '-',
                                            'alpha': 0.5,
                                            'color': 'black',
                                            'shrinkB': 0,
                                            },
                                **annotate_kwargs,
                            )
        
    if non_targeting_label_location is None:
        pass
    else:
        if non_targeting_label_location == 'floating':
            floating_labels = [
                ('individual non-targeting guide', -50, nt_guide_color),
                #('{} p-value < 1e-{}'.format(p_val_method, p_cutoff), -25, 'C1'),
            ]
            for text, offset, color in floating_labels:
                grid.ax_joint.annotate(text,
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
            highest_nt_guide = df.query('variable_gene == "negative_control"').sort_values('num_cells').iloc[-1]

            vals = {
                'num_cells': highest_nt_guide['num_cells'],
                fraction_key: highest_nt_guide[fraction_key],
            }
            x = vals[axis_to_quantity['x']]
            y = vals[axis_to_quantity['y']]

            if flip_axes:
                annotate_kwargs = dict(
                    text='individual\nnon-targeting\nguide',
                    xytext=(-30, 0),
                    ha='right',
                    va='center',
                )
            else:
                annotate_kwargs = dict(
                    text='individual\nnon-targeting\nguide',
                    xytext=(0, 30),
                    ha='center',
                    va='bottom',
                )

            grid.ax_joint.annotate(xy=(x, y),
                                xycoords='data',
                                textcoords='offset points',
                                color=nt_guide_color,
                                size=7,
                                arrowprops={'arrowstyle': '-',
                                            'alpha': 0.9,
                                            'color': nt_guide_color,
                                            'shrinkB': 0,
                                            #'linestyle': 'dashed',
                                            'linewidth': 0.5,
                                            },
                                **annotate_kwargs,
                            )
            
    vals = {
        'num_cells': num_cells_max * 0.90,
        fraction_key: nt_fraction * fraction_scaling_factor,
    }
    x = vals[axis_to_quantity['x']]
    y = vals[axis_to_quantity['y']]

    if flip_axes:
        annotate_kwargs = dict(
            text='average of all\nnon-targeting\nguides',
            xytext=(-20, 0),
            ha='right',
            va='center',
        )
    else:
        annotate_kwargs = dict(
            text='average of all\nnon-targeting\nguides',
            xytext=(0, 30),
            ha='center',
            va='bottom',
        )

    grid.ax_joint.annotate(xy=(x, y),
                        xycoords='data',
                        textcoords='offset points',
                        color='black',
                        size=7,
                        arrowprops={'arrowstyle': '-',
                                    'alpha': 0.5,
                                    'color': 'black',
                                    'shrinkB': 0,
                                   },
                        **annotate_kwargs,
                    )

    # Draw in order of color frequency from most common to least common
    # so that rare colors are drawn on top.
    color_order_to_plot = df.value_counts('color').index

    for color_i, color in enumerate(color_order_to_plot):
        to_plot = df.query('color == @color')
        grid.ax_joint.scatter(x=axis_to_quantity['x'],
                        y=axis_to_quantity['y'],
                        data=to_plot,
                        s=marker_size,
                        alpha=0.9,
                        color='color',
                        linewidths=(0,),
                        zorder=1 if color_i == 0 else 10,
                        clip_on=kwargs.get('clip_on', True),
                       )

    if gene_to_color is not None:
        query = '(fixed_gene in @gene_to_color or variable_gene in @gene_to_color)'
        if only_best_promoter:
            query += ' and variable_guide_best_promoter'

#        grid.ax_joint.scatter(x=axis_to_quantity['x'],
#                        y=axis_to_quantity['y'],
#                        data=df.query(query),
#                        s=marker_size,
#                        alpha=0.9,
#                        color='color',
#                        linewidths=(0,),
#                        zorder=10,
#                        )
#
        legend_elements = [Line2D([0], [0], marker='o', color=color, label=f'{gene} fixed guide', linestyle='none') for gene, color in gene_to_color.items()]
        legend_elements.append(Line2D([0], [0], marker='o', color='C0', label=f'non-targeting in both positions', linestyle='none'))
        #grid.ax_joint.legend(handles=legend_elements)

    if quantity_to_axis[fraction_key] == 'y':
        line_func = grid.ax_joint.axhline
    else:
        line_func = grid.ax_joint.axvline

    line_func(nt_fraction * fraction_scaling_factor, color='black')

    axis_label_funcs = {
        'x': grid.ax_joint.set_xlabel,     
        'y': grid.ax_joint.set_ylabel,     
    }

    num_cells_label = 'number of UMIs per CRISPRi guide'
    axis_label_funcs[quantity_to_axis['num_cells']](num_cells_label, size=axis_label_size)

    if outcome_name is None:
        try:
            outcome_name = '_'.join(outcome)
        except TypeError:
            outcome_name = 'PH'

    fraction_label = f'{fraction_key} of CRISPRi-guide-containing cells with {outcome_name}'
    axis_label_funcs[quantity_to_axis[fraction_key]](fraction_label, size=axis_label_size)

    if draw_diagram:
        ax_marg_x_p = grid.ax_marg_x.get_position()
        ax_marg_y_p = grid.ax_marg_y.get_position()

        diagram_width = ax_marg_x_p.width * 0.5 + ax_marg_y_p.width
        diagram_gap = ax_marg_x_p.height * 0.3

        if isinstance(outcome, tuple):
            outcomes_to_plot = [outcome]
        else:
            outcomes_to_plot = list(pool.non_targeting_counts('perfect', fixed_guide).loc[outcome].sort_values(ascending=False).index.values[:4])

        diagram_height = ax_marg_x_p.height * 0.1 * len(outcomes_to_plot)

        diagram_ax = grid.fig.add_axes((ax_marg_y_p.x1 - diagram_width, ax_marg_x_p.y1 - diagram_gap - diagram_height, diagram_width, diagram_height))
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
            to_label = to_label[~to_label['variable_gene'].isin(genes_to_label)]

        to_label = to_label.query(f'{fraction_key} >= @fraction_min and {fraction_key} <= @fraction_max')

        if flip_axes:
            vector = ['upper right' if v == 'up' else 'upper left' for v in to_label['direction']]
        else:
            vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        hits.visualize.label_scatter_plot(grid.ax_joint,
                                          axis_to_quantity['x'],
                                          axis_to_quantity['y'],
                                          'alias',
                                          color='label_color',
                                          data=to_label,
                                          vector=kwargs.get('label_vector', vector),
                                          text_kwargs=dict(size=small_guide_label_size, weight='bold'),
                                          initial_distance=initial_label_offset,
                                          distance_increment=5,
                                          arrow_alpha=0.2,
                                          avoid=avoid_label_overlap,
                                          avoid_axis_labels=True,
                                         )

    if guides_to_label is not None or genes_to_label is not None:
        if guides_to_label is not None:
            to_label = df[df.index.isin(guides_to_label)]
        elif genes_to_label is not None:
            query = '(fixed_gene in @genes_to_label or variable_gene in @genes_to_label)'
            if only_best_promoter:
                query += ' and variable_guide_best_promoter'
            to_label = df.query(query)

        to_label = to_label.query(f'{fraction_key} >= @fraction_min and {fraction_key} <= @fraction_max')

        if flip_axes:
            vector = ['upper right' if v == 'up' else 'upper left' for v in to_label['direction']]
        else:
            vector = ['upper right' if v == 'up' else 'lower right' for v in to_label['direction']]

        if color_labels:
            label_kwargs = dict(color='color',
                         )
        else:
            label_kwargs = dict(color=None,
                         )

        hits.visualize.label_scatter_plot(grid.ax_joint,
                                          axis_to_quantity['x'],
                                          axis_to_quantity['y'],
                                          'alias',
                                          data=to_label,
                                          vector=kwargs.get('label_vector', vector),
                                          initial_distance=initial_label_offset,
                                          distance_increment=5,
                                          arrow_alpha=0.2,
                                          avoid=avoid_label_overlap,
                                          avoid_axis_labels=True,
                                          avoid_existing=True,
                                          min_arrow_distance=10,
                                          text_kwargs=dict(size=big_guide_label_size, weight='bold'),
                                          **label_kwargs,
                                         )

    hits.visualize.add_commas_to_ticks(grid.ax_joint, which=quantity_to_axis['num_cells'])
    grid.ax_joint.tick_params(labelsize=tick_label_size)

    if not draw_marginals:
        grid.fig.delaxes(grid.ax_marg_x)
        grid.fig.delaxes(grid.ax_marg_y)

    return grid, df

def guide(pool, gene, number,
          ax=None,
          fixed_guide='none',
          outcomes_to_draw=15,
          subset_query=None,
          sort_by='log2_fc',
          min_frequency=1e-4,
          max_fc=6,
         ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    if subset_query is None:
        subset_query = 'significant'

    fig_width, fig_height = fig.get_size_inches()

    guide = pool.variable_guide_library.gene_guides(gene)[number]

    nt_fracs = pool.non_targeting_fractions('perfect', fixed_guide)
    fracs = pool.outcome_fractions('perfect')[fixed_guide, guide]
    counts = pool.outcome_counts('perfect')[fixed_guide, guide]
    ratios = pool.fold_changes('perfect', fixed_guide)[fixed_guide, guide]

    num_UMIs = pool.UMI_counts('perfect')[fixed_guide, guide]

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
    data = data.query(f'nt_fracs >= {min_frequency}').copy()

    data['lower_0.05'], data['upper_0.05'] = hits.utilities.clopper_pearson_fast(data['counts'], num_UMIs)
    data['log2_fc_lower_0.05'] = np.log2(np.maximum(data['lower_0.05'] / data['nt_fracs'], 2**-(max_fc + 0.1)))
    data['log2_fc_upper_0.05'] = np.log2(np.minimum(data['upper_0.05'] / data['nt_fracs'], 2**(max_fc + 0.1)))

    data.index.name = 'outcome'

    ax.annotate('{:,} UMIs'.format(num_UMIs),
                xy=(1, 1),
                xycoords='axes fraction',
                xytext=(-5, 5),
                textcoords='offset points',
                ha='right',
                va='bottom',
               )

    if num_UMIs == 0:
        return fig

    data['lower_above_0'] = data['log2_fc_lower_0.05'] > 0
    data['upper_below_0'] = data['log2_fc_upper_0.05'] < 0
    data['nt_not_in_ci'] = data['lower_above_0'] | data['upper_below_0']

    data['up'] = data['fracs'] > data['nt_fracs']
    data['down'] = data['fracs'] < data['nt_fracs']

    data['significant'] = data['lower_above_0'] | data['upper_below_0']
    #data.loc[data['significant_down'], 'color'] = 'C0'
    #data.loc[data['significant_up'], 'color'] = 'C3'
    colors = bokeh.palettes.Category10[10][1:3] + bokeh.palettes.Category10[10][4:]

    categories = [
        'wild type',
        'insertion',
        'deletion',
    ]
    category_to_color = {cat: color for cat, color in zip(categories, colors)}

    legend_elements = [Line2D([0], [0], marker='o', color=category_to_color.get(cat, 'grey'), label=cat, linestyle='none') for cat in categories + ['other']]
    ax.legend(handles=legend_elements)

    data['color'] = [category_to_color.get(c, 'grey') for c, s, d in data.index.values]

    if subset_query is not None:
        data['alpha'] = 0.4
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

    # Draw binomial confidence intervals for each guide.
    for _, row in data.query(subset_query).iterrows():
        x = row['log10_nt_frac']
        ax.plot([x, x], [row['log2_fc_lower_0.05'], row['log2_fc_upper_0.05']], color='black', alpha=0.2)

    ax.axhline(0, color='black', alpha=0.8)

    # Draw binomial confidence interval around 0.
    for cutoff in [1e-3]:
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

    ax.set_xlim(np.log10(min_frequency) - 0.1, data['log10_nt_frac'].max() + 0.1)
    ax.set_ylim(-max_fc - 0.1, max_fc + 0.1)

    ax.set_xlabel('log10(fraction in non-targeting)', size=16)
    ax.set_ylabel('log2(fold change relative to non-targeting)', size=16)

    ax.set_yticks(np.arange(-max_fc, max_fc + 1, 1))

    ax.set_title(f'{pool.target_info.sgRNA}\n{guide}', size=16)

    plt.draw()
    ax_p = ax.get_position()
    
    for direction in ['up', 'down']:
        if direction == 'up':
            label = 'gene activity suppresses outcome'
            color = 'C3'
            sorted_slice = slice(None, outcomes_to_draw)
        else:
            label = 'gene activity promotes outcome'
            color = 'C0'
            sorted_slice = slice(-outcomes_to_draw, None)

        outcome_subset = data.query(f'{direction} and {subset_query}')
        rows = outcome_subset.sort_values(sort_by, ascending=False).iloc[sorted_slice]
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
        if x == np.log10(min_frequency):
            l.set_text(r'$\leq${}'.format(np.log10(min_frequency)))

    ax.set_xticklabels(labels)
    
    return fig, data

def gene(pool, gene, **kwargs):
    guides = pool.variable_guide_library.gene_guides(gene)

    fig, axs = plt.subplots(len(guides), 1, figsize=(10, 10 * len(guides)))
    if len(guides) == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        guide(pool, gene, i, ax=ax, **kwargs)

    return fig