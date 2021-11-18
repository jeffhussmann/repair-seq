import copy
import pickle
import string

import bokeh.io
import bokeh.models
import bokeh.plotting
import numpy as np

from hits.visualize.callback import build_js_callback

def scatter(data_source,
            outcome_names=None,
            plot_width=2000,
            plot_height=800,
            initial_guides=None,
            initial_genes=None,
            save_as=None,
           ):

    if save_as is not None and save_as != 'layout':
        bokeh.io.output_file(save_as)

    if initial_guides is not None and initial_genes is not None:
        raise ValueError('can only specify one of initial_guides or initial_genes')

    if initial_guides is None:
        initial_guides = []

    if isinstance(data_source, str):
        with open(data_source, 'rb') as fh:
            data = pickle.load(fh)
    else:
        data = copy.deepcopy(data_source)

    pool_names = data['pool_names']

    if outcome_names is None:
        outcome_names = data['outcome_names']

    guides_df = data['guides_df']
    nt_fractions = data['nt_percentages']
    initial_dataset = data['initial_dataset']
    initial_outcome = data['initial_outcome']

    guides_df.columns = ['_'.join(t) if t[1] != '' else t[0] for t in data['guides_df'].columns.values]

    table_keys = [
        'frequency',
        'percentage',
        'ys',
        'total_UMIs',
        'gene_p_up',
        'gene_p_down',
        'log2_fold_change',
    ]
    for key in table_keys:
        guides_df[key] = guides_df[f'{initial_dataset}_{initial_outcome}_{key}'] 

    scatter_source = bokeh.models.ColumnDataSource(data=guides_df, name='scatter_source')
    scatter_source.data[guides_df.index.name] = guides_df.index

    if initial_genes is not None:
        initial_indices = np.array(guides_df.query('gene in @initial_genes')['x'])
    else:
        initial_indices = np.array(guides_df.loc[initial_guides]['x'])

    scatter_source.selected = bokeh.models.Selection(indices=initial_indices)
    scatter_source.selected.js_on_change('indices', build_js_callback(__file__, 'screen_scatter_selection'))

    filtered_data = {k: [scatter_source.data[k][i] for i in initial_indices] for k in scatter_source.data}
    filtered_source = bokeh.models.ColumnDataSource(data=filtered_data, name='filtered_source')

    x_min = -1
    x_max = len(guides_df)

    y_min = 0
    y_max = guides_df['percentage'].max() * 1.2

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

    fig.x_range.callback = build_js_callback(__file__, 'screen_range', format_kwargs={'lower_bound': x_min, 'upper_bound': x_max})
    fig.y_range.callback = build_js_callback(__file__, 'screen_range', format_kwargs={'lower_bound': 0, 'upper_bound': 100})

    circles = fig.circle(x='x',
                         y='percentage',
                         source=scatter_source,
                         color='color', selection_color='color', nonselection_color='color',
                         alpha=0.8,
                         selection_line_alpha=0, nonselection_line_alpha=0, line_alpha=0,
                         size=5,
                        )

    tooltips = [
        ('CRISPRi sgRNA', '@guide'),
        ('Frequency of outcome', '@frequency'),
        ('Log2 fold change from nt', '@log2_fold_change'),
        ('Total UMIs', '@total_UMIs'),
    ]

    hover = bokeh.models.HoverTool(renderers=[circles])
    hover.tooltips = tooltips

    fig.add_tools(hover)

    fig.multi_line(xs='xs', ys='ys',
                   source=scatter_source,
                   color='color', selection_color='color', nonselection_color='color',
                   alpha=0.4,
                   name='confidence_intervals',
                )

    labels = bokeh.models.LabelSet(x='x', y='percentage', text='guide',
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

    dataset_menu = bokeh.models.widgets.MultiSelect(options=pool_names, value=[initial_dataset], name='dataset_menu', title='Screen condition:', size=len(pool_names) + 2, width=400)
    outcome_menu = bokeh.models.widgets.MultiSelect(options=outcome_names, value=[initial_outcome], name='outcome_menu', title='Outcome category:', size=len(outcome_names) + 2, width=400)

    menu_js = build_js_callback(__file__, 'screen_menu', format_kwargs={'nt_fractions': str(nt_fractions)})
    for menu in [dataset_menu, outcome_menu]:
        menu.js_on_change('value', menu_js)

    fig.add_layout(bokeh.models.Span(location=nt_fractions[f'{initial_dataset}_{initial_outcome}'], dimension='width', line_alpha=0.5, name='nt_fraction'))

    interval_button = bokeh.models.widgets.Toggle(label='Show confidence intervals', active=True)
    interval_button.js_on_change('active', build_js_callback(__file__, 'screen_errorbars_button'))

    cutoff_slider = bokeh.models.Slider(start=-10, end=-2, value=-5, step=1, name='cutoff_slider', title='log10 p-value significance threshold')

    down_button = bokeh.models.widgets.Button(label='Filter to genes that significantly decrease', name='filter_down')
    up_button = bokeh.models.widgets.Button(label='Filter to genes that significantly increase', name='filter_up')
    for button in [down_button, up_button]:
        button.js_on_click(build_js_callback(__file__, 'screen_significance_filter'))

    text_input = bokeh.models.TextInput(title='Search sgRNAs:', name='search')
    text_input.js_on_change('value', build_js_callback(__file__, 'screen_search'))

    fig.outline_line_color = 'black'
    first_letters = [g[0] for g in guides_df.index]
    x_tick_labels = {first_letters.index(c): c for c in string.ascii_uppercase if c in first_letters}
    x_tick_labels[first_letters.index('n')] = 'nt'
    fig.xaxis.ticker = sorted(x_tick_labels)
    fig.xaxis.major_label_overrides = x_tick_labels

    table_col_names = [
        ('guide', 'CRISPRi sgRNA', 50),
        ('gene', 'Gene', 50),
        ('frequency', 'Frequency of outcome', 50),
        ('log2_fold_change', 'Log2 fold change', 50),
        ('total_UMIs', 'Total UMIs', 50),
    ]
    columns = []
    for col_name, col_label, width in table_col_names:
        width = 50
        if col_name == 'gene':
            formatter = bokeh.models.widgets.HTMLTemplateFormatter(template='<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=<%= value %>" target="_blank"><%= value %></a>')
        elif col_name == 'log2_fold_change':
            formatter = bokeh.models.widgets.NumberFormatter(format='0.00')
        elif col_name == 'total_UMIs':
            formatter = bokeh.models.widgets.NumberFormatter(format='0,0')
        elif col_name == 'frequency':
            formatter = bokeh.models.widgets.NumberFormatter(format='0.00%')
        else:
            formatter = None

        column = bokeh.models.widgets.TableColumn(field=col_name,
                                                  title=col_label,
                                                  formatter=formatter,
                                                  width=width,
                                                 )
        columns.append(column)
        
    save_button = bokeh.models.widgets.Button(label='Save table', name='save_button')
    save_button.js_on_click(build_js_callback(__file__, 'scatter_save_button', format_kwargs={'column_names': table_col_names}))

    table = bokeh.models.widgets.DataTable(source=filtered_source,
                                           columns=columns,
                                           width=600,
                                           height=300,
                                           sortable=False,
                                           reorderable=False,
                                           name='table',
                                           index_position=None,
                                          )

    fig.xaxis.axis_label = 'CRISPRi sgRNAs (ordered alphabetically)'
    fig.yaxis.axis_label = 'Frequency of outcome'
    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_label_text_font_size = '14pt'
        axis.axis_label_text_font_style = 'bold'

    #fig.yaxis.formatter = bokeh.models.NumeralTickFormatter(format=' 0.00%')
    fig.yaxis.formatter = bokeh.models.PrintfTickFormatter(format="%6.2f%%")
    fig.yaxis.major_label_text_font = 'courier'
    fig.yaxis.major_label_text_font_style = 'bold'
        
    #fig.title.name = 'title'
    #fig.title.text = f'{initial_dataset}\ntest      {initial_outcome}'

    #fig.title.text_font_size = '16pt'
    fig.add_layout(bokeh.models.Title(text=f'Outcome category: {initial_outcome}', text_font_size='14pt', name='subtitle'), 'above')
    fig.add_layout(bokeh.models.Title(text=f'Screen condition: {initial_dataset}', text_font_size='14pt', name='title'), 'above')

    top_widgets = bokeh.layouts.column([bokeh.layouts.Spacer(height=70),
                                        dataset_menu,
                                        outcome_menu,
                                        text_input,
                                       ])

    bottom_widgets = bokeh.layouts.column([interval_button,
                                           cutoff_slider,
                                           up_button,
                                           down_button,
                                           save_button,
                                          ])

    first_row = bokeh.layouts.row([fig, top_widgets])
    second_row = bokeh.layouts.row([bokeh.layouts.Spacer(width=90), table, bottom_widgets])
    final_layout = bokeh.layouts.column([first_row, bokeh.layouts.Spacer(height=50), second_row])
    
    if save_as == 'layout':
        return final_layout
    elif save_as is not None:
        bokeh.io.save(final_layout)
    else:
        bokeh.io.show(final_layout)
