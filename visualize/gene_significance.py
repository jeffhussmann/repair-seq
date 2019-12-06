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

from . import pooled_screen
from ddr.visualize import outcome_diagrams
from hits import utilities
from hits.visualize.interactive.external_coffeescript import build_callback

def get_outcome_statistics(pool, outcomes, omit_bad_guides=True):
    def pval_down(outcome_count, UMI_count, nt_fraction):
        if UMI_count == 0:
            p = 0.5
        else:
            p = scipy.stats.binom.cdf(outcome_count, UMI_count, nt_fraction)
        return p

    def pval_up(outcome_count, UMI_count, nt_fraction):
        if UMI_count == 0:
            p = 0.5
        else:
            p = 1 - scipy.stats.binom.cdf(outcome_count - 1, UMI_count, nt_fraction)
        return p

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
        guides_to_omit = ['BAZ1B_3', 'MUTYH_2', 'MBD3_2', 'CCNE1_2', 'SSRP1_2', 'NABP1_2']
    else:
        guides_to_omit = []

    UMI_counts = pool.UMI_counts('perfect')
    outcome_counts = pool.outcome_counts('perfect').loc[outcomes].sum()
    frequencies = pool.outcome_fractions('perfect').loc[outcomes].sum().drop('non_targeting')
    
    nt_fraction = pool.non_targeting_fractions('perfect').loc[outcomes].sum()

    ps_down = np.array([pval_down(o, u, nt_fraction) for o, u in zip(outcome_counts, UMI_counts)])
    ps_up = np.array([pval_up(o, u, nt_fraction) for o, u in zip(outcome_counts, UMI_counts)])

    genes = [pool.guide_library.guide_to_gene(g) for g in pool.guide_library.guides]

    capped_fc = np.minimum(2**5, np.maximum(2**-5, frequencies / nt_fraction))

    df = pd.DataFrame({'total_UMIs': UMI_counts,
                       'outcome_count': outcome_counts,
                       'frequency': frequencies,
                       'log2_fold_change': np.log2(capped_fc),
                       'p_down': ps_down,
                       'p_up': ps_up,
                       'gene': genes,
                      })
    df = df.drop(guides_to_omit, errors='ignore')
    
    ps = defaultdict(list)

    max_k = 9

    for direction in ('down', 'up'):
        for gene in pool.guide_library.genes:
            sorted_ps = df[df['gene'] == gene][f'p_{direction}'].sort_values()
            n = len(sorted_ps)
            for k in range(1, max_k + 1):
                ps[direction, k].append(p_k_of_n_less(n, k, sorted_ps))
            
    p_df = pd.DataFrame(ps, index=pool.guide_library.genes).min(axis=1, level=0)

    guides_per_gene = df.groupby('gene').size()
    bonferonni_factor = np.minimum(max_k, guides_per_gene)
    corrected_ps = np.minimum(1, p_df.multiply(bonferonni_factor, axis=0))

    clopper_pairs = [utilities.clopper_pearson(row['outcome_count'], row['total_UMIs']) for _, row in df.iterrows()]
    df['interval_bottom'] = [r - b for r, (b, a) in zip(df['frequency'], clopper_pairs)]
    df['interval_top'] = [r + a for r, (b, a) in zip(df['frequency'], clopper_pairs)]

    return df, nt_fraction, corrected_ps

def compute_table(base_dir, pool_names, outcome_groups, pickle_fn, initial_dataset=None, initial_outcome=None, progress=None):
    if progress is None:
        progress = utilities.identity

    if initial_dataset is None:
        initial_dataset = pool_names[0]
    if initial_outcome is None:
        initial_outcome = outcome_groups[0][0]

    all_columns = {}
    nt_fractions = {}

    column_names = ['frequency', 'ys', 'total_UMIs', 'gene_p_up', 'gene_p_down']

    for pool_name in progress(pool_names):
        pool = pooled_screen.PooledScreen(base_dir, pool_name)
        
        for outcome_name, is_eligible in progress(outcome_groups):
            outcomes = [csd for csd in pool.outcome_counts('perfect').index.values if is_eligible(csd)]
            df, nt_fraction, p_df = get_outcome_statistics(pool, outcomes)

            df['x'] = np.arange(len(df))
            df['xs'] = [[x, x] for x in df['x']]

            clopper_pairs = [utilities.clopper_pearson(row['outcome_count'], row['total_UMIs']) for _, row in df.iterrows()]
            df['ys'] = [[r - b, r + a] for r, (b, a) in zip(df['frequency'], clopper_pairs)]
            for direction in ['down', 'up']:
                df[f'gene_p_{direction}'] = np.array([p_df.loc[pool.guide_to_gene(guide), direction] for guide in df.index])

            for col_name in column_names:
                all_columns[f'{pool_name}_{outcome_name}_{col_name}'] = df[col_name]
                
            nt_fractions[f'{pool_name}_{outcome_name}'] = nt_fraction
            
    full_df = pd.DataFrame(all_columns)

    for col_name in column_names:
        full_df[col_name] = full_df[f'{initial_dataset}_{initial_outcome}_{col_name}']

    color_list = [c for i, c in enumerate(bokeh.palettes.Category10[10]) if i != 7]
    grey = bokeh.palettes.Category10[10][7]

    gene_to_color = {g: color_list[i % len(color_list)] if g != 'negative_control' else grey for i, g in enumerate(pool.genes)}

    full_df['color'] = [gene_to_color[pool.guide_to_gene(guide)] for guide in df.index]
    full_df.index.name = 'guide'

    for common_key in ['gene', 'x', 'xs']:
        full_df[common_key] = df[common_key]

    to_pickle = {
        'pool_names': pool_names,
        'outcome_names': [name for name, _ in outcome_groups],
        'full_df': full_df,
        'nt_fractions': nt_fractions,
        'initial_dataset': initial_dataset,
        'initial_outcome': initial_outcome,
    }

    with open(pickle_fn, 'wb') as fh:
        pickle.dump(to_pickle, fh)

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

def gene_significance(pool, outcomes, draw_outcomes=False, p_val_threshold=0.05):
    df, nt_fraction, p_df = get_outcome_statistics(pool, outcomes, omit_bad_guides=True)
    df['x'] = np.arange(len(df))

    labels = list(df.index)

    gene_to_color = {g: f'C{i % 10}' for i, g in enumerate(pool.guide_library.genes)}

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
                y = gene_rows['interval_top'].max()
                va = 'bottom'
                y_offset = 5
            else:
                y = gene_rows['interval_bottom'].min()
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

        guides_to_label = {g for g in df.index if pool.guide_library.guide_to_gene(g) in genes_to_label}

        colors = [gene_to_color[pool.guide_library.guide_to_gene(guide)] for guide in labels]
        colors = matplotlib.colors.to_rgba_array(colors)
        alpha = [0.95 if guide in guides_to_label else 0.15 for guide in df.index]
        colors[:, 3] = alpha

        ax.scatter(np.arange(len(df)), df['frequency'], s=25, c=colors, linewidths=(0,))
        ax.set_xlim(-10, len(pool.guide_library.guides) + 10)

        for x, (y, label) in enumerate(zip(df['frequency'], labels)):
            if label in guides_to_label:
                ax.annotate(label.split('_')[-1],
                            xy=(x, y),
                            xytext=(2, 0),
                            textcoords='offset points',
                            size=6,
                            color=colors[x],
                            va='center',
                           )

        ax.axhline(nt_fraction, color='black', alpha=0.5)
        ax.annotate(f'{nt_fraction:0.3f}',
                    xy=(1, nt_fraction),
                    xycoords=('axes fraction', 'data'),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
                    size=10,
                    va='center',
                   )

        relevant_ys = df.loc[guides_to_label]['interval_top']

        max_y = max(nt_fraction * 2, relevant_ys.max() * 1.1)
        global_max_y = max(global_max_y, max_y)
        ax.set_xticklabels([])

        for x, (_, row) in enumerate(df.iterrows()):
            ax.plot([x, x], [row['interval_bottom'], row['interval_top']], color=colors[x])

    for direction in ['up', 'down']:
        axs[direction].set_ylim(0, global_max_y)

    axs['up'].set_title('most significant suppressing genes', y=0.92)
    axs['down'].set_title('most significant promoting genes', y=0.92)

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