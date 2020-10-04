import pickle
import string
from collections import defaultdict

import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import bokeh.palettes
import bokeh.models

from ddr import pooled_screen
from ddr.visualize import outcome_diagrams
from hits import utilities
from hits.visualize.interactive.external_coffeescript import build_callback

def get_outcome_statistics(pool, outcomes, omit_bad_guides=True, fixed_guide='none', denominator_outcomes=None):
    def pval_down(outcome_count, UMI_count, nt_fraction):
        return scipy.stats.binom.cdf(outcome_count, UMI_count, nt_fraction)

    def pval_up(outcome_count, UMI_count, nt_fraction):
        # sf required to maintain precision
        # since sf is P(> n), outcome_count - 1 required to produce P(>= n)
        return scipy.stats.binom.sf(outcome_count - 1, UMI_count, nt_fraction)

    n_choose_k = scipy.special.comb

    def p_k_of_n_less(n, k, sorted_ps):
        if k > n:
            return 1
        else:
            a = sorted_ps[k - 1]
            
            total = 0
            for free_dimensions in range(0, n - k + 1):
                total += n_choose_k(n, free_dimensions) * (1 - a)**free_dimensions * a**(n - free_dimensions)
            
            return total

    if omit_bad_guides:
        guides_to_omit = ['eGFP_NT2', 'BAZ1B_3', 'MUTYH_2', 'MBD3_2', 'CCNE1_2', 'SSRP1_2', 'NABP1_2']
    else:
        guides_to_omit = []

    if denominator_outcomes is None:
        UMI_counts = pool.UMI_counts('perfect')[fixed_guide]
    else:
        UMI_counts = pool.outcome_counts('perfect')[fixed_guide].loc[denominator_outcomes].sum()

    if isinstance(outcomes, pd.Series):
        numerator_counts = outcomes.loc[fixed_guide]
    else:
        numerator_counts = pool.outcome_counts('perfect')[fixed_guide].loc[outcomes].sum()

    frequencies = numerator_counts / UMI_counts
    
    nt_guides = pool.variable_guide_library.non_targeting_guides
    nt_fraction = numerator_counts.loc[nt_guides].sum() / UMI_counts.loc[nt_guides].sum()

    ps_down = pval_down(numerator_counts.values, UMI_counts.values, nt_fraction)
    ps_up = pval_up(numerator_counts.values, UMI_counts.values, nt_fraction)

    genes = pool.variable_guide_library.guides_df['gene']

    capped_fc = np.minimum(2**5, np.maximum(2**-5, frequencies / nt_fraction))

    guides_df = pd.DataFrame({'total_UMIs': UMI_counts,
                              'outcome_count': numerator_counts,
                              'frequency': frequencies,
                              'log2_fold_change': np.log2(capped_fc),
                              'p_down': ps_down,
                              'p_up': ps_up,
                              'gene': genes,
                             })
    guides_df = guides_df.drop(guides_to_omit, errors='ignore')
    
    ps = defaultdict(list)

    max_k = 9

    gene_order = []
    for gene, rows in guides_df.groupby('gene'):
        gene_order.append(gene)
        for direction in ('down', 'up'):
            sorted_ps = sorted(rows[f'p_{direction}'].values)
            n = len(sorted_ps)
            for k in range(1, max_k + 1):
                ps[direction, k].append(p_k_of_n_less(n, k, sorted_ps))
            
    uncorrected_ps_df = pd.DataFrame(ps, index=gene_order).min(axis=1, level=0)

    guides_per_gene = guides_df.groupby('gene').size()
    bonferonni_factor = np.minimum(max_k, guides_per_gene)
    corrected_ps_df = np.minimum(1, uncorrected_ps_df.multiply(bonferonni_factor, axis=0))

    guides_df['interval_bottom'], guides_df['interval_top'] = utilities.clopper_pearson_fast(guides_df['outcome_count'], guides_df['total_UMIs'])

    guides_df['log2_fold_change_interval_bottom'] = np.maximum(-6, np.log2(guides_df['interval_bottom'] / nt_fraction))
    guides_df['log2_fold_change_interval_top'] = np.minimum(5, np.log2(guides_df['interval_top'] / nt_fraction))

    guides_df['p_relevant'] = guides_df[['p_down', 'p_up']].min(axis=1)
    guides_df['-log10_p_relevant'] = -np.log10(np.maximum(np.finfo(np.float64).tiny, guides_df['p_relevant']))

    for direction in ['down', 'up']:
        guides_df[f'gene_p_{direction}'] = np.array([corrected_ps_df.loc[row['gene'], direction] for guide, row in guides_df.iterrows()])

    guides_df.index.name = 'guide'

    targeting_ps = corrected_ps_df.drop(index='negative_control')

    up_genes = targeting_ps.query('up < down')['up'].sort_values(ascending=False).index

    grouped_fcs = guides_df.groupby('gene')['log2_fold_change']

    gene_log2_fold_changes = pd.DataFrame({'up': grouped_fcs.nlargest(2).mean(level=0), 'down': grouped_fcs.nsmallest(2).mean(level=0)})

    gene_log2_fold_changes['relevant'] = gene_log2_fold_changes['down']
    gene_log2_fold_changes.loc[up_genes, 'relevant'] = gene_log2_fold_changes.loc[up_genes, 'up']

    corrected_ps_df['relevant'] = corrected_ps_df['down']
    corrected_ps_df.loc[up_genes, 'relevant'] = corrected_ps_df.loc[up_genes, 'up']

    genes_df = pd.concat({'log2 fold change': gene_log2_fold_changes, 'p': corrected_ps_df}, axis=1)

    negative_log10_p = -np.log10(np.maximum(np.finfo(np.float64).tiny, genes_df['p']))

    negative_log10_p = pd.concat({'-log10 p': negative_log10_p}, axis=1)

    genes_df = pd.concat([genes_df, negative_log10_p], axis=1)

    genes_df.index.name = 'gene'

    return guides_df, nt_fraction, genes_df

def compute_table(base_dir, pool_names, outcome_groups,
                  pickle_fn=None,
                  initial_dataset=None,
                  initial_outcome=None,
                  progress=None,
                  use_short_names=False,
                 ):
    if progress is None:
        progress = utilities.identity

    all_columns = {}
    nt_fractions = {}

    all_gene_columns = {}

    guide_column_names = ['log2_fold_change', 'frequency', 'ys', 'total_UMIs', 'gene_p_up', 'gene_p_down']
    outcome_names = []

    guide_to_gene = {}

    for pool_name in progress(pool_names):
        pool = pooled_screen.PooledScreen(base_dir, pool_name)

        if use_short_names:
            name_to_use = pool.short_name
        else:
            name_to_use = pool_name
        
        for outcome_specification in progress(outcome_groups):
            if isinstance(outcome_specification, tuple) and len(outcome_specification) == 3:
                # It is a just a single outcome.
                outcome_name = '_'.join(outcome_specification)
                outcomes = [outcome_specification]
            elif len(outcome_specification) == 2:
                outcome_name, rule = outcome_specification
                if callable(rule):
                    outcomes = rule(pool)
                else:
                    if isinstance(rule, tuple) and len(rule) == 3:
                        # It is a just a single outcome.
                        outcomes = [rule]
                    else:
                        # a list of outcomes
                        outcomes = rule
            
            outcome_names.append(outcome_name)

            full_name = f'{name_to_use}_{outcome_name}'

            df, nt_fraction, genes_df = get_outcome_statistics(pool, outcomes)

            all_gene_columns[full_name] = genes_df['log2 fold change', 'relevant']

            guide_to_gene.update(df['gene'].items())

            df['x'] = np.arange(len(df))
            df['xs'] = [[x, x] for x in df['x']]
            df['ys'] = [[row['interval_bottom'], row['interval_top']] for _, row in df.iterrows()]

            for col_name in guide_column_names:
                all_columns[full_name, col_name] = df[col_name]
                
            nt_fractions[full_name] = nt_fraction
            
    full_df = pd.DataFrame(all_columns)

    genes_df = pd.DataFrame(all_gene_columns)

    if initial_dataset is None:
        initial_dataset = pool_names[0]
    if initial_outcome is None:
        initial_outcome = outcome_names[0]

    #for col_name in column_names:
    #    full_df[col_name] = full_df[f'{initial_dataset}_{initial_outcome}_{col_name}']

    color_list = [c for i, c in enumerate(bokeh.palettes.Category10[10]) if i != 7]
    grey = bokeh.palettes.Category10[10][7]

    gene_to_color = {g: color_list[i % len(color_list)] if g != 'negative_control' else grey for i, g in enumerate(pool.variable_guide_library.genes)}

    #full_df['color'] = [gene_to_color[row['gene']] for guide, row in df.iterrows()]
    full_df.index.name = 'guide'

    for common_key in ['x', 'xs']:
        full_df[common_key] = df[common_key]

    full_df['gene'] = pd.Series(guide_to_gene)

    to_pickle = {
        'pool_names': pool_names,
        'outcome_names': outcome_names,
        'full_df': full_df,
        'nt_fractions': nt_fractions,
        'initial_dataset': initial_dataset,
        'initial_outcome': initial_outcome,
        'genes_df': genes_df,
    }

    if pickle_fn is not None:
        with open(pickle_fn, 'wb') as fh:
            pickle.dump(to_pickle, fh)
    
    return to_pickle

def scatter(pickle_fn,
            plot_width=2000,
            plot_height=800,
           ):
    with open(pickle_fn, 'rb') as fh:
        data = pickle.load(fh)

    pool_names = data['pool_names']
    outcome_names = data['outcome_names']
    full_df = data['full_df']
    nt_fractions = data['nt_fractions']
    initial_dataset = data['initial_dataset']
    initial_outcome = data['initial_outcome']

    scatter_source = bokeh.models.ColumnDataSource(data=full_df, name='scatter_source')
    scatter_source.data[full_df.index.name] = full_df.index

    initial_indices = np.array(full_df.query('gene_p_down <= 1e-5')['x'])
    scatter_source.selected = bokeh.models.Selection(indices=initial_indices)
    scatter_source.selected.js_on_change('indices', build_callback('screen_scatter_selection'))

    filtered_data = {k: [scatter_source.data[k][i] for i in initial_indices] for k in scatter_source.data}
    filtered_source = bokeh.models.ColumnDataSource(data=filtered_data, name='filtered_source')

    x_min = -1
    x_max = len(full_df)

    y_min = 0
    y_max = 1

    tools = [
        'reset',
        'undo',
        'pan',
        'box_zoom',
        'box_select',
        'tap',
        'wheel_zoom',
        'save',
    ]

    fig = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height,
                                tools=tools, active_drag='box_select', active_scroll='wheel_zoom',
                               )
    fig.x_range = bokeh.models.Range1d(x_min, x_max, name='x_range')
    fig.y_range = bokeh.models.Range1d(y_min, y_max, name='y_range')

    fig.x_range.callback = build_callback('screen_range', format_kwargs={'lower_bound': x_min, 'upper_bound': x_max})
    fig.y_range.callback = build_callback('screen_range', format_kwargs={'lower_bound': y_min, 'upper_bound': y_max})

    circles = fig.circle(x='x', y='frequency',
                        source=scatter_source,
                        color='color', selection_color='color', nonselection_color='color',
                        alpha=0.8,
                        selection_line_alpha=0, nonselection_line_alpha=0, line_alpha=0,
                        size=5,
                        )

    tooltips = [
        ("guide", "@guide"),
        ("frequency of outcome", "@frequency"),
        ("total UMIs", "@total_UMIs"),
    ]

    hover = bokeh.models.HoverTool(renderers=[circles])
    hover.tooltips = tooltips

    fig.add_tools(hover)

    fig.multi_line(xs='xs', ys='ys',
                   source=scatter_source,
                   color='color', selection_color='color', nonselection_color='color',
                   alpha=0.8,
                   name='confidence_intervals',
                )

    labels = bokeh.models.LabelSet(x='x', y='frequency', text='guide',
                                   source=filtered_source,
                                   level='glyph',
                                   x_offset=4, y_offset=0,
                                   text_font_size='6pt',
                                   text_color='color',
                                   text_baseline='middle',
                                   name='labels',
                                  )
    fig.add_layout(labels)
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    dataset_menu = bokeh.models.widgets.MultiSelect(options=pool_names, value=[initial_dataset], name='dataset_menu', title='dataset', size=len(pool_names))
    outcome_menu = bokeh.models.widgets.MultiSelect(options=outcome_names, value=[initial_outcome], name='outcome_menu', title='outcome', size=len(outcome_names))

    menu_js = build_callback('screen_menu', format_kwargs={'nt_fractions': str(nt_fractions)})
    for menu in [dataset_menu, outcome_menu]:
        menu.js_on_change('value', menu_js)

    fig.add_layout(bokeh.models.Span(location=nt_fractions[f'{initial_dataset}_{initial_outcome}'], dimension='width', line_alpha=0.5, name='nt_fraction'))

    interval_button = bokeh.models.widgets.Toggle(label='show confidence intervals', active=True)
    interval_button.js_on_change('active', build_callback('screen_errorbars_button'))

    cutoff_slider = bokeh.models.Slider(start=-10, end=-2, value=-5, step=1, name='cutoff_slider', title='log10 p-val cutoff')

    down_button = bokeh.models.widgets.Button(label='filter significant down', name='filter_down')
    up_button = bokeh.models.widgets.Button(label='filter significant up', name='filter_up')
    for button in [down_button, up_button]:
        button.js_on_click(build_callback('screen_significance_filter'))

    text_input = bokeh.models.TextInput(title='Search:', name='search')
    text_input.js_on_change('value', build_callback('screen_search'))

    fig.outline_line_color = 'black'
    first_letters = [g[0] for g in full_df.index]
    x_tick_labels = {first_letters.index(c): c for c in string.ascii_uppercase if c in first_letters}
    fig.xaxis.ticker = sorted(x_tick_labels)
    fig.xaxis.major_label_overrides = x_tick_labels

    table_col_names = ['guide', 'gene', 'frequency', 'total_UMIs', 'gene_p_up', 'gene_p_down']
    columns = []
    for col_name in table_col_names:
        lengths = [len(str(v)) for v in scatter_source.data[col_name]]
        mean_length = np.mean(lengths)

        width = 50
        if col_name == 'gene':
            formatter = bokeh.models.widgets.HTMLTemplateFormatter(template='<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=<%= value %>" target="_blank"><%= value %></a>')
        elif 'gene_p' in col_name:
            formatter = bokeh.models.widgets.NumberFormatter(format='0.0000000')
        elif col_name == 'frequency':
            formatter = bokeh.models.widgets.NumberFormatter(format='0.00')
        else:
            formatter = None
            width = min(500, int(12 * mean_length))

        column = bokeh.models.widgets.TableColumn(field=col_name,
                                                  title=col_name,
                                                  formatter=formatter,
                                                  width=width,
                                                 )
        columns.append(column)
        
    save_button = bokeh.models.widgets.Button(label='save table', name='save_button')
    save_button.js_on_click(build_callback('scatter_save_button', format_kwargs={'column_names': table_col_names}))

    table = bokeh.models.widgets.DataTable(source=filtered_source,
                                           columns=columns,
                                           width=800,
                                           height=600,
                                           sortable=False,
                                           reorderable=False,
                                           name='table',
                                           index_position=None,
                                          )

    fig.xaxis.axis_label = 'guides (ordered alphabetically)'
    fig.yaxis.axis_label = 'frequency of outcome'
    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_label_text_font_size = '16pt'
        axis.axis_label_text_font_style = 'normal'
        
    fig.title.name = 'title'
    fig.title.text = f'{initial_dataset}      {initial_outcome}'

    fig.title.text_font_size = '16pt'

    widgets = bokeh.layouts.column([dataset_menu, outcome_menu, interval_button, cutoff_slider, up_button, down_button, text_input, save_button])
    bokeh.io.show(bokeh.layouts.column([fig, bokeh.layouts.row([table, bokeh.layouts.Spacer(width=40), widgets])]))

def gene_significance(pool, outcomes, draw_outcomes=False, p_val_threshold=0.05, as_percentage=False):
    df, nt_fraction, p_df = get_outcome_statistics(pool, outcomes)
    df['x'] = np.arange(len(df))

    if as_percentage:
        nt_percentage = nt_fraction * 100
        df['percentage'] = df['frequency'] * 100
        df['percentage_interval_bottom'] = df['interval_bottom'] * 100
        df['interval_top'] = df['interval_top'] * 100

        frequency_key = 'percentage'
        bottom_key = 'percentage_interval_bottom'
        top_key = 'percentage_interval_top'
        nt_value = nt_percentage
    else:
        frequency_key = 'frequency'
        bottom_key = 'interval_bottom'
        top_key = 'interval_top'
        nt_value = nt_fraction

    labels = list(df.index)

    gene_to_color = {g: f'C{i % 10}' for i, g in enumerate(pool.variable_guide_library.genes)}

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
        ordered = p_df[direction][p_df[direction] < p_val_threshold].sort_values()
        genes_to_label = set(ordered.index[:50])
        subset =  ordered
        subset_df = pd.DataFrame({'gene': subset.index, 'p_val': list(subset)}, index=np.arange(1, len(subset) + 1))
        significant[direction] = subset_df
        
        for gene in genes_to_label:
            gene_rows = df.query('gene == @gene')

            x = gene_rows['x'].mean()
            if direction == 'up':
                y = gene_rows[top_key].max()
                va = 'bottom'
                y_offset = 5
            else:
                y = gene_rows[bottom_key].min()
                va = 'top'
                y_offset = -5

            ax.annotate(gene,
                        xy=(x, y),
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        size=8,
                        color=gene_to_color[gene],
                        va=va,
                        ha='center',
                       )

        guides_to_label = {g for g in df.index if pool.variable_guide_library.guide_to_gene(g) in genes_to_label}

        colors = [gene_to_color[pool.variable_guide_library.guide_to_gene(guide)] for guide in labels]
        colors = matplotlib.colors.to_rgba_array(colors)
        alpha = [0.95 if guide in guides_to_label else 0.15 for guide in df.index]
        colors[:, 3] = alpha

        ax.scatter('x', frequency_key, data=df, s=15, c=colors, linewidths=(0,))
        ax.set_xlim(-10, len(pool.variable_guide_library.guides) + 10)

        for (x, y, label) in zip(df['x'], df[frequency_key], labels):
            if label in guides_to_label:
                ax.annotate(label.split('_')[-1],
                            xy=(x, y),
                            xytext=(2, 0),
                            textcoords='offset points',
                            size=6,
                            color=colors[x],
                            va='center',
                           )

        ax.axhline(nt_value, color='black', alpha=0.5)
        ax.annotate(f'{nt_value:0.3f}',
                    xy=(1, nt_value),
                    xycoords=('axes fraction', 'data'),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
                    size=10,
                    va='center',
                   )

        relevant_ys = df.loc[guides_to_label][top_key]

        max_y = max(nt_fraction * 2, relevant_ys.max() * 1.1)
        global_max_y = max(global_max_y, max_y)
        ax.set_xticklabels([])

        for _, row in df.iterrows():
            x = row['x']
            ax.plot([x, x], [row[bottom_key], row[top_key]], color=colors[x])

    for direction in ['up', 'down']:
        axs[direction].set_ylim(0, global_max_y)

    first_letters = [guide[0] for guide in df.index]
    x_tick_to_label = {first_letters.index(c): c for c in string.ascii_uppercase if c in first_letters}
    axs['down'].set_xticks(sorted(x_tick_to_label))
    axs['down'].set_xticklabels([l for x, l in sorted(x_tick_to_label.items())])

    #axs['up'].set_title('most significant suppressing genes', y=0.92)
    #axs['down'].set_title('most significant promoting genes', y=0.92)

    if draw_outcomes:
        n = 40
        outcome_order = pool.non_targeting_fractions('perfect').loc[outcomes].sort_values(ascending=False).index[:n]
        outcome_diagrams.plot(outcome_order, pool.target_info, num_outcomes=n, window=(-60, 20), flip_if_reverse=True, ax=axs['outcomes'])
        outcome_diagrams.add_frequencies(fig, axs['outcomes'], pool, outcome_order[:n], text_only=True)
    else:
        fig.delaxes(axs['outcomes'])

    axs['up'].annotate(pool.group,
                       xy=(0.5, 1),
                       xycoords='axes fraction',
                       xytext=(0, 20),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       size=16,
                      )
    
    return fig, significant

def gene_significance_simple(pool, outcomes,
                             p_val_threshold=1e-4,
                             quantity_to_plot='frequency',
                             draw_fold_change_grid=False,
                             fixed_guide='none',
                             denominator_outcomes=None,
                             max_num_to_label=50,
                             title=None,
                             label_by='significance',
                             figsize=(27, 6),
                             y_lims=None,
                             genes_to_label=None,
                             manual_label_offsets=None,
                            ):

    if manual_label_offsets is None:
        manual_label_offsets = {}

    df, nt_fraction, genes_df = get_outcome_statistics(pool, outcomes, fixed_guide=fixed_guide, denominator_outcomes=denominator_outcomes)

    df['x'] = np.arange(len(df))

    if quantity_to_plot == 'frequency':
        y_key = 'frequency'
        bottom_key = 'interval_bottom'
        top_key = 'interval_top'
        nt_value = nt_fraction
        y_label = 'fraction of outcomes'

    elif quantity_to_plot == 'percentage':
        nt_percentage = nt_fraction * 100
        df['percentage'] = df['frequency'] * 100
        df['percentage_interval_bottom'] = df['interval_bottom'] * 100
        df['percentage_interval_top'] = df['interval_top'] * 100

        y_key = 'percentage'
        bottom_key = 'percentage_interval_bottom'
        top_key = 'percentage_interval_top'
        nt_value = nt_percentage
        y_label = 'percentage of outcomes'

    elif quantity_to_plot == 'log2_fold_change':
        nt_value = 0
        y_key = 'log2_fold_change'
        bottom_key = 'log2_fold_change_interval_bottom'
        top_key = 'log2_fold_change_interval_top'
        y_label = 'log2 fold change from non-targeting'

    if y_lims is not None:
        min_y, max_y = y_lims
    else:
        enough_UMIs = df['total_UMIs'] > 500
        min_y = df.loc[enough_UMIs, bottom_key].min() * 1.1
        max_y = df.loc[enough_UMIs, top_key].max() * 1.1

    if denominator_outcomes is not None:
        y_label = 'relative ' + y_label

    labels = list(df.index)

    gene_to_color = {g: f'C{i % 10}' for i, g in enumerate(pool.variable_guide_library.genes)}
    gene_to_color['negative_control'] = 'grey'

    fig, ax = plt.subplots(figsize=figsize)

    global_max_y = 0

    p_df = genes_df['p']

    gene_label_size = 8
    if genes_to_label is not None:
        all_genes_to_label = set(genes_to_label)
        gene_label_size = 12
    elif label_by == 'significance':
        ordered = p_df['relevant'].sort_values()
        below_threshold = ordered[ordered <= p_val_threshold]
        all_genes_to_label = set(below_threshold.index[:max_num_to_label])
    else:
        if quantity_to_plot == 'log2_fold_change':
            magnitude = df['log2_fold_change']
        else:
            magnitude = df['frequency'] - nt_fraction
           
        guide_order = np.abs(magnitude).sort_values(ascending=False).index
        gene_order = df['gene'].loc[guide_order].unique()
        all_genes_to_label = set(gene_order[:max_num_to_label])

    for direction in ['up', 'down']:
        if direction == 'up':
            other_direction = 'down'
        else:
            other_direction = 'up'

        correct_direction_genes = set(p_df[p_df[direction] < p_df[other_direction]].index)
        genes_to_label = all_genes_to_label & correct_direction_genes

        blocks = []
        current_block = []

        for gene in p_df.index:
            if not gene in genes_to_label:
                if len(current_block) > 0:
                    blocks.append(current_block)
                current_block = []
            else:
                current_block.append(gene)

        for block in blocks:
            block_rows = df.query('gene in @block')
            if direction == 'up':
                y = block_rows[top_key].max()
                va = 'bottom'
                y_offset = 5
            else:
                y = block_rows[bottom_key].min()
                va = 'top'
                y_offset = -5

            y = max(y, min_y)
            y = min(y, max_y)

            for gene_i, gene in enumerate(block):
                gene_rows = df.query('gene == @gene')

                x = gene_rows['x'].mean()
                
                extra_offset = manual_label_offsets.get(gene, 0)

                total_offset = (1.5 * gene_i + 1 + extra_offset) * y_offset

                ax.annotate(gene,
                            xy=(x, y),
                            xytext=(0, total_offset),
                            textcoords='offset points',
                            size=gene_label_size,
                            color=gene_to_color[gene],
                            va=va,
                            ha='center',
                           )

    guides_to_label = set(pool.variable_guide_library.gene_guides(all_genes_to_label))

    colors = df['gene'].map(gene_to_color)

    point_colors = matplotlib.colors.to_rgba_array(colors)
    point_alphas = [0.95 if guide in guides_to_label else 0.25 for guide in df.index]
    point_colors[:, 3] = point_alphas
    
    line_colors = matplotlib.colors.to_rgba_array(colors)
    line_alphas = [0.3 if guide in guides_to_label else 0.15 for guide in df.index]
    line_colors[:, 3] = line_alphas

    ax.scatter('x', y_key, data=df, s=15, c=point_colors, linewidths=(0,))

    for x, y, label in zip(df['x'], df[y_key], labels):
        if label in guides_to_label:
            ax.annotate(label.split('_')[-1],
                        xy=(x, y),
                        xytext=(2, 0),
                        textcoords='offset points',
                        size=6,
                        color=colors[x],
                        va='center',
                        annotation_clip=False,
                       )

    label = f'{nt_fraction:0.2%}'

    ax.annotate(label,
                xy=(1, nt_value),
                xycoords=('axes fraction', 'data'),
                xytext=(5, 0),
                textcoords='offset points',
                ha='left',
                size=10,
                va='center',
               )

    relevant_ys = df[top_key]

    max_y = max(nt_value * 2, relevant_ys.max() * 1.1)
    global_max_y = max(global_max_y, max_y)

    for _, row in df.iterrows():
        x = row['x']
        ax.plot([x, x], [row[bottom_key], row[top_key]], color=line_colors[x])

    if quantity_to_plot != 'log2_fold_change':
        ax.set_ylim(0, global_max_y)
    else:
        ax.set_ylim(min_y, max_y)

    ax.axhline(nt_value, color='black', alpha=0.5)

    if draw_fold_change_grid:
        for fold_change in [2**i for i in [-2, -1, 1, 2]]:
            y = nt_value * fold_change
            if y < global_max_y:
                ax.axhline(y, color='black', alpha=0.1)
                if fold_change < 1:
                    label = f'{int(1 / fold_change)}-fold down'
                else:
                    label = f'{fold_change}-fold up'

                ax.annotate(label,
                            xy=(1, y),
                            xycoords=('axes fraction', 'data'),
                            xytext=(5, 0),
                            textcoords='offset points',
                            ha='left',
                            size=10,
                            va='center',
                            )


    letter_xs = defaultdict(list)

    for x, guide in enumerate(df.index):
        first_letter = guide[0]
        letter_xs[first_letter].append(x)
        
    first_letter_xs = [min(xs) for l, xs in sorted(letter_xs.items())]
    boundaries = list(np.array(first_letter_xs) - 0.5) + [len(df) - 1 + 0.5]
    mean_xs = {l: np.mean(xs) for l, xs in letter_xs.items()}

    ax.set_xticks(boundaries[1:-1])
    ax.set_xticklabels([])

    for l, x in mean_xs.items():
        if l == 'n':
            l = 'non-\ntargeting'
        ax.annotate(l,
                    xy=(x, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -18),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                )

    ax.set_xlim(-0.005 * len(df), len(df))

    if title is None:
        title = pool.group
    
    if title == '':
        title = pool.group
        alpha = 0
    else:
        alpha = 1

    ax.annotate(title,
                xy=(0.5, 1),
                xycoords='axes fraction',
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                va='center',
                size=16,
                alpha=alpha,
               )

    ax.set_ylabel(y_label, size=12)

    ax.set_xlabel('guides (alphabetical order)', labelpad=20, size=12)

    for side in ['top', 'bottom', 'right']:
        ax.spines[side].set_visible(False)
    
    return fig, df, p_df

def genetics_of_stat(stat_series,
                     pool,
                     genes_to_label=3,
                     title=None,
                     y_label=None,
                     y_lims=None,
                     figsize=(27, 6),
                    ):
    stat_series = pd.Series(stat_series)
    stat_series.name = 'stat'
    nt_value = stat_series[pooled_screen.ALL_NON_TARGETING]
    df = pd.DataFrame(stat_series, columns=['stat']).drop(index=pooled_screen.ALL_NON_TARGETING)
    df['gene'] = [pool.variable_guide_library.guide_to_gene[g] for g in df.index]
    df['x'] = np.arange(len(df))

    labels = list(df.index)

    gene_to_color = {g: f'C{i % 10}' for i, g in enumerate(pool.variable_guide_library.genes)}

    gene_to_color['negative_control'] = 'grey'

    fig, ax = plt.subplots(figsize=figsize)

    global_max_y = 0

    low_genes = set()

    for gene in df.sort_values('stat')['gene']:
        if len(low_genes) >= genes_to_label:
            break
        low_genes.add(gene)
            
    high_genes = set()
            
    for gene in df.sort_values('stat', ascending=False)['gene']:
        if len(high_genes) >= genes_to_label:
            break
        high_genes.add(gene)

    genes_to_label = high_genes | low_genes

    for gene in genes_to_label:
        gene_rows = df.query('gene == @gene')

        x = gene_rows['x'].mean()
        if gene in high_genes:
            y = df.loc[pool.variable_guide_library.gene_guides(gene)]['stat'].max()
            va = 'bottom'
            y_offset = 5
        else:
            y = df.loc[pool.variable_guide_library.gene_guides(gene)]['stat'].min()
            va = 'top'
            y_offset = -5

        ax.annotate(gene,
                    xy=(x, y),
                    xytext=(0, y_offset),
                    textcoords='offset points',
                    size=8,
                    color=gene_to_color[gene],
                    va=va,
                    ha='center',
                    )

    guides_to_label = set(pool.variable_guide_library.gene_guides(genes_to_label))

    colors = df['gene'].map(gene_to_color)

    point_colors = matplotlib.colors.to_rgba_array(colors)
    point_alphas = [0.95 if guide in guides_to_label else (0.5 if len(genes_to_label) > 0 else 0.95) for guide in df.index]
    point_colors[:, 3] = point_alphas
    
    #line_colors = matplotlib.colors.to_rgba_array(colors)
    #line_alphas = [0.3 if guide in guides_to_label else 0.15 for guide in df.index]
    #line_colors[:, 3] = line_alphas

    ax.scatter('x', 'stat', data=df, s=15, c=point_colors, linewidths=(0,))

    #for x, y, label in zip(df['x'], df[frequency_key], labels):
    #    if label in guides_to_label:
    #        ax.annotate(label.split('_')[-1],
    #                    xy=(x, y),
    #                    xytext=(2, 0),
    #                    textcoords='offset points',
    #                    size=6,
    #                    color=colors[x],
    #                    va='center',
    #                   )

    #if as_percentage:
    #    label = f'{nt_fraction:0.2%}'
    #else:
    #    label = f'{nt_fraction:0.2f}'

    #ax.annotate(label,
    #            xy=(1, nt_value),
    #            xycoords=('axes fraction', 'data'),
    #            xytext=(5, 0),
    #            textcoords='offset points',
    #            ha='left',
    #            size=10,
    #            va='center',
    #           )

    #relevant_ys = df[top_key]

    #max_y = max(nt_value * 2, relevant_ys.max() * 1.1)
    #global_max_y = max(global_max_y, max_y)

    #for _, row in df.iterrows():
    #    x = row['x']
    #    ax.plot([x, x], [row[bottom_key], row[top_key]], color=line_colors[x])

    y_range = df['stat'].max() - df['stat'].min()
    ax.set_ylim(df['stat'].min() - y_range * 0.1, df['stat'].max() + y_range * 0.1)

    ax.axhline(nt_value, color='black', alpha=0.5)

    letter_xs = defaultdict(list)

    for x, guide in enumerate(df.index):
        first_letter = guide[0]
        letter_xs[first_letter].append(x)
        
    first_letter_xs = [min(xs) for l, xs in sorted(letter_xs.items())]
    boundaries = list(np.array(first_letter_xs) - 0.5) + [len(df) - 1 + 0.5]
    mean_xs = {l: np.mean(xs) for l, xs in letter_xs.items()}

    ax.set_xticks(boundaries)
    ax.set_xticklabels([])

    for l, x in mean_xs.items():
        if l == 'n':
            l = 'non-\ntargeting'
        ax.annotate(l,
                    xy=(x, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -12),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                )

    ax.set_xlim(-0.005 * len(df), 1.005 * len(df))

    if title is None:
        title = pool.group

    ax.annotate(title,
                xy=(0.5, 1),
                xycoords='axes fraction',
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                va='center',
                size=16,
               )

    ax.set_ylabel(y_label, size=12)

    ax.set_xlabel('guides (alphabetical order)', labelpad=20, size=12)

    if y_lims is not None:
        ax.set_ylim(*y_lims)
    
    return fig, df
