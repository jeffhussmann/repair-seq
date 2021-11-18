import bokeh.plotting
import matplotlib.colors
import matplotlib.cm
import numpy as np
import pandas as pd

from hits.visualize import callback
from repair_seq import visualize

def plot(clusterer):
    all_l2fcs = clusterer.all_log2_fold_changes

    corrs = all_l2fcs.corr()

    initial_guide = 'POLQ_1'

    corrs['x'] = np.arange(len(corrs))
    corrs['y'] = corrs[initial_guide]
    sorted_pairs = sorted(enumerate(corrs['y']), key=lambda pair: pair[1], reverse=True)
    corrs['ranked_indices'] = [pair[0] for pair in sorted_pairs]
    corrs['color'] = visualize.targeting_guide_color
    corrs.loc[clusterer.guide_library.non_targeting_guides, 'color'] = visualize.nontargeting_guide_color
    corrs.index.name = 'sgRNA'

    corrs_source = bokeh.models.ColumnDataSource(data=corrs)

    data = clusterer.outcome_embedding.copy()

    def bind_hexer(scalar_mappable):
        def to_hex(v):
            return matplotlib.colors.to_hex(scalar_mappable.to_rgba(v))
        return to_hex

    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    cmap = visualize.fold_changes_cmap
    l2fcs_sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    l2fcs_to_hex = bind_hexer(l2fcs_sm)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=30)
    cmap = matplotlib.cm.get_cmap('Purples')
    deletion_length_sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    deletion_length_to_hex = bind_hexer(deletion_length_sm)

    norm = matplotlib.colors.Normalize(vmin=-3, vmax=-1)
    cmap = matplotlib.cm.get_cmap('YlOrBr')
    log10_fraction_sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    log10_fraction_to_hex = bind_hexer(log10_fraction_sm)

    colors = {}

    for guide in clusterer.guide_library.guides:
        values = all_l2fcs[guide]

        hex_values = [matplotlib.colors.to_hex(row) for row in l2fcs_sm.to_rgba(values)]
        colors[f'color_{guide}'] = hex_values

    target_colors, target_to_color = clusterer.sgRNA_colors()
    target_to_hex_color = {target: matplotlib.colors.to_hex(color) for target, color in target_to_color.items()}
    colors['target'] = [matplotlib.colors.to_hex(row) for row in target_colors]
        
    colors = pd.DataFrame(colors, index=data.index)
    relevant_data = clusterer.outcome_embedding[['x', 'y', 'category_colors', 'MH length', 'deletion length', 'log10_fraction']]
        
    all_colors = pd.concat([relevant_data, colors], axis=1)

    all_colors['first_color'] = all_colors[f'color_{initial_guide}']
    all_colors['blank'] = ['#FFFFFF' for _ in all_colors.index]
    all_colors['second_color'] = all_colors['blank']

    color_source = bokeh.models.ColumnDataSource(data=all_colors)

    color_mappers = {
        'none': None,
        'MH length': bokeh.models.LinearColorMapper(low=-0.5, high=3.5, palette=bokeh.palettes.viridis(4)[::-1], low_color='white'),
        'l2fcs': bokeh.models.LinearColorMapper(low=-2, high=2, palette=[l2fcs_to_hex(v) for v in np.linspace(-2, 2, 500)]),
        'deletion length': bokeh.models.LinearColorMapper(low=0, high=30, palette=[deletion_length_to_hex(v) for v in np.linspace(0, 30, 500)]),
        'log10_fraction': bokeh.models.LinearColorMapper(low=-3, high=-1, palette=[log10_fraction_to_hex(v) for v in np.linspace(-3, -1, 500)]),
    }

    figs = {}

    big_frame_size = 600
    small_frame_size = 450

    for name, color_key, transform_key in [('l2fcs', 'first_color', 'none'),
                                           ('category', 'category_colors', 'none'),
                                           ('target', 'target', 'none'),
                                           ('MH length', 'MH length', 'MH length'),
                                           ('deletion length', 'deletion length', 'deletion length'),
                                           ('log10_fraction', 'log10_fraction', 'log10_fraction'),
                                          ]:
        fig = bokeh.plotting.figure(frame_width=big_frame_size, frame_height=big_frame_size, min_border=0)
        
        fig.scatter('x', 'y',
                    source=color_source,
                    fill_color={'field': color_key, 'transform': color_mappers[transform_key]},
                    line_color=None,
                    size=7,
                   )

        fig.grid.visible = False
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.toolbar_location = None
        fig.toolbar.active_drag = None
        fig.outline_line_color = 'black'
        fig.outline_line_alpha = 0.5

        figs[name] = fig

    color_menu = bokeh.models.widgets.Select(title='CRISPRi sgRNA',
                                             options=list(clusterer.guide_library.guides),
                                             value=initial_guide,
                                             width=150,
                                            )

    corrs_fig = bokeh.plotting.figure(plot_width=big_frame_size + small_frame_size + 50 + 20,
                                      plot_height=320,
                                      toolbar_location=None,
                                      active_drag=None,
                                      min_border=0,
                                     )

    corrs_scatter = corrs_fig.scatter(x='x', y='y', source=corrs_source,
                                      fill_color='color',
                                      line_color=None,
                                      size=6,
                                     )

    top_indices = corrs['ranked_indices'].iloc[:10]
    top_guides = corrs.iloc[top_indices].index.values
    filtered_corrs = corrs.loc[top_guides, ['x', 'y', 'color']].copy()
    filtered_corrs_source = bokeh.models.ColumnDataSource(data=filtered_corrs)
    
    labels = bokeh.models.LabelSet(x='x',
                                   y='y',
                                   text='sgRNA',
                                   level='glyph',
                                   x_offset=7,
                                   y_offset=0,
                                   source=filtered_corrs_source,
                                   text_font_size='8pt',
                                   text_baseline='middle',
                                  )
    corrs_fig.add_layout(labels)

    corrs_fig.xaxis.axis_label = 'CRISPRi sgRNAs (ordered alphabetically)'
    corrs_fig.xaxis.axis_label_text_font_style = 'normal'
    corrs_fig.xaxis.axis_label_text_font_size = '14pt'
    corrs_fig.yaxis.axis_label = f'Correlation with {initial_guide}'
    corrs_fig.yaxis.axis_label_text_font_style = 'normal'
    corrs_fig.yaxis.axis_label_text_font_size = '14pt'

    line_kwargs = dict(
        color='black',
        nonselection_color='black',
        line_width=1,
    ) 

    x_bounds = [-5, max(corrs['x']) + 5]

    corrs_fig.line(x=x_bounds, y=[0, 0], **line_kwargs)

    xs = [
        corrs.loc[initial_guide, 'x'],
        corrs.loc[initial_guide, 'x'],
        100,
        100,
    ]

    ys = [
        corrs.loc[initial_guide, 'y'] + 0.05,
        1.12,
        1.12,
        1.2,
    ]

    left_line_source = bokeh.models.ColumnDataSource(pd.DataFrame({'x': xs, 'y': ys}))

    corrs_fig.line(x='x', y='y', source=left_line_source, line_alpha=0.5, **line_kwargs)

    xs = [
        corrs.loc[initial_guide, 'x'],
        corrs.loc[initial_guide, 'x'],
        300,
        300,
    ]

    ys = [
        corrs.loc[initial_guide, 'y'] + 0.05,
        1.08,
        1.08,
        1.2,
    ]

    right_line_source = bokeh.models.ColumnDataSource(pd.DataFrame({'x': xs, 'y': ys}))

    right_line = corrs_fig.line(x='x', y='y', source=right_line_source, visible=False, line_alpha=0.5, **line_kwargs)

    corrs_fig.y_range = bokeh.models.Range1d(-0.7, 1.2)
    corrs_fig.yaxis.bounds = (-0.6, 1)
    corrs_fig.x_range = bokeh.models.Range1d(*x_bounds)

    corrs_fig.xgrid.visible = False
    corrs_fig.xaxis.ticker = []

    corrs_fig.yaxis.ticker = np.linspace(-0.6, 1, 9)
    corrs_fig.outline_line_color = None

    corrs_fig.xaxis.ticker = []

    corrs_fig.xaxis.axis_line_color = None

    nt_text = pd.DataFrame([
        {'x': max(corrs['x']) - len(clusterer.guide_library.non_targeting_guides) / 2,
         'y': -0.4,
         'text': 'non-\ntargeting',
         'color': visualize.nontargeting_guide_color,
        }
    ])

    corrs_fig.text(x='x', y='y', text='text', source=nt_text, color='color', text_align='center', text_font_style='bold', text_font_size={'value': '12px'})

    second_scatter = bokeh.plotting.figure(frame_width=small_frame_size,
                                           frame_height=small_frame_size,
                                           toolbar_location=None,
                                           min_border=0,
                                           outline_line_color=None,
                                          )

    second_scatter.scatter('x', 'y', source=color_source, fill_color='second_color', line_color=None, size=6)
    second_scatter.grid.visible = False
    second_scatter.xaxis.visible = False
    second_scatter.yaxis.visible = False

    common_colorbar_kwargs = dict(
        orientation='horizontal',
        width=200,
        location=(0.1 * big_frame_size, 0.6 * big_frame_size),
        major_tick_out=5,
        major_tick_in=0,
        margin=100,
        padding=10,
        title_text_font_style='normal',
        background_fill_color=None,
    )

    color_bars = {
        'l2fcs': bokeh.models.ColorBar(
            color_mapper=color_mappers['l2fcs'],
            major_tick_line_color='black',
            ticker=bokeh.models.FixedTicker(ticks=[-2, -1, 0, 1, 2]),
            major_label_overrides={2: '≥2', -2: '≤-2'},
            title=f'Log₂ fold change in outcome frequency\nfor the selected CRISPRi sgRNA',
            **common_colorbar_kwargs,
        ),
        'MH length': bokeh.models.ColorBar(
            color_mapper=color_mappers['MH length'],
            major_tick_line_color=None,
            ticker=bokeh.models.FixedTicker(ticks=[0, 1, 2, 3]),
            major_label_overrides={3: '≥3'},
            title='Flanking microhomology (nts)',
            **common_colorbar_kwargs,
        ),
        'deletion length': bokeh.models.ColorBar(
            color_mapper=color_mappers['deletion length'],
            major_tick_line_color='black',
            ticker=bokeh.models.FixedTicker(ticks=[0, 10, 20, 30]),
            major_label_overrides={30: '≥30'},
            title='Deletion length (nts)',
            **common_colorbar_kwargs,
        ),
        'log10_fraction': bokeh.models.ColorBar(
            color_mapper=color_mappers['log10_fraction'],
            major_tick_line_color='black',
            ticker=bokeh.models.FixedTicker(ticks=[-3, -2, -1]),
            major_label_overrides={-3: '≤0.1%', -2: '1%', -1: '≥10%'},
            title='Baseline frequency of outcome',
            **common_colorbar_kwargs,
        ),
    }

    for key in color_bars:
        figs[key].add_layout(color_bars[key])

    # Draw title and various text labels on big scatter plots.

    common_text_kwargs = dict(
        x='x',
        y='y',
        text='text',
        text_align='center',
        color='color',
    )

    category_text = pd.DataFrame([
        {'x': -4, 'y': -3.5, 'text': 'bidirectional\ndeletions', 'color': visualize.category_colors['SpCas9']['deletion, bidirectional']},
        {'x': 3, 'y': 5.5, 'text': 'insertions\nI', 'color': visualize.category_colors['SpCas9']['insertion']},
        {'x': 4.5, 'y': 3.6, 'text': 'insertions\nII', 'color': visualize.category_colors['SpCas9']['insertion']},
        {'x': -7, 'y': 1.7, 'text': 'insertions\nIII', 'color': visualize.category_colors['SpCas9']['insertion']},
        {'x': -7, 'y': -2.2, 'text': 'insertions\nwith\ndeletions', 'color': visualize.category_colors['SpCas9']['insertion with deletion']},
        {'x': 6.4, 'y': 2, 'text': 'unedited', 'color': visualize.category_colors['SpCas9']['wild type']},
        {'x': 9.9, 'y': 5.2, 'text': 'capture of\ngenomic\nsequence\nat break', 'color': visualize.category_colors['SpCas9']['genomic insertion']},
        {'x': 9.9, 'y': 3.8, 'text': '≤75 nts', 'color': visualize.category_colors['SpCas9']['genomic insertion']},
        {'x': 10.4, 'y': 1.9, 'text': '>75 nts', 'color': visualize.category_colors['SpCas9']['genomic insertion']},
        {'x': 10, 'y': -0.5, 'text': 'deletions\nconsistent\nwith either\nside', 'color': visualize.category_colors['SpCas9']['deletion, ambiguous']},
        {'x': 2.5, 'y': -4, 'text': 'deletions\non only\nPAM-distal\nside', 'color': visualize.category_colors['SpCas9']['deletion, PAM-distal']},
        {'x': 1, 'y': 1.2, 'text': 'deletions\non only\nPAM-proximal\nside', 'color': visualize.category_colors['SpCas9']['deletion, PAM-proximal']},
    ])

    figs['category'].text(source=category_text, text_font_size={'value': '14px'}, **common_text_kwargs)

    title_text = pd.DataFrame([{'x': -3, 'y': 5, 'text': 'Cas9 outcome\nembedding', 'color': 'black'}])

    for fig in figs.values():
        fig.text(source=title_text, text_font_size={'value': '26px'}, **common_text_kwargs)

    target_text = pd.DataFrame([
        {'x': -3, 'y': 4.5 - 0.5 * i, 'text': f'Cas9 target {i}', 'color': target_to_hex_color[f'SpCas9 target {i}']}
        for i in [1, 2, 3, 4]
    ])

    figs['target'].text(source=target_text, text_font_size={'value': '18px'}, **common_text_kwargs)

    # Layout the components of the dashboard.

    def row(*children):
        return bokeh.models.layouts.Row(children=list(children))

    def col(*children):
        return bokeh.models.layouts.Column(children=list(children))

    tabs = [
        bokeh.models.Panel(child=figs['category'], title='Category'),
        bokeh.models.Panel(child=figs['target'], title='Target site'),
        bokeh.models.Panel(child=figs['MH length'], title='MH length'),
        bokeh.models.Panel(child=figs['deletion length'], title='Deletion length'),
        bokeh.models.Panel(child=figs['log10_fraction'], title='Baseline frequency'),
        bokeh.models.Panel(child=figs['l2fcs'], title='Log₂ fold changes'),
    ]
    tabs = bokeh.models.Tabs(tabs=tabs, active=5)
    color_by = bokeh.models.Div(text='Color by:')

    def v_gap(height):
        return bokeh.layouts.Spacer(height=height)

    def h_gap(width):
        return bokeh.layouts.Spacer(width=width)

    #final_layout = row([color_by, tabs, col([row([color_menu, h_gap(20), second_scatter]), v_gap(20), corrs_fig])])

    gap_above = 40
    menu_height = 59 # Empirically determined
    tab_menu_height = 29 # Empirically determined
    gap_between = big_frame_size  + tab_menu_height - small_frame_size - gap_above - menu_height
    
    final_layout = col(row(col(v_gap(2), color_by), tabs, h_gap(20), col(v_gap(gap_above), color_menu, v_gap(gap_between), second_scatter)),
                       row(corrs_fig),
                      )

    # Build and attach callbacks at the end, so all necessary args are available.

    menu_callback = callback.build_js_callback(__file__, 'UMAP_menu', 
                                              args=dict(color_source=color_source,
                                                        corrs_source=corrs_source,
                                                        filtered_corrs_source=filtered_corrs_source,
                                                        left_line_source=left_line_source,
                                                        color_menu=color_menu,
                                                        corrs_y_axis=corrs_fig.yaxis,
                                                        tabs=tabs,
                                                       ),
                                               )
    color_menu.js_on_change('value', menu_callback)

    hover_callback = callback.build_js_callback(__file__, 'UMAP_hover',
                                                args=dict(color_source=color_source,
                                                          corrs_source=corrs_source,
                                                          filtered_corrs_source=filtered_corrs_source,
                                                          right_line_source=right_line_source,
                                                          second_scatter=second_scatter,
                                                          right_line=right_line,
                                                         ),
                                                )
    
    hover = bokeh.models.HoverTool(renderers=[corrs_scatter],
                                   callback=hover_callback,
                                  )
    #hover.tooltips = [
    #    (corrs.index.name, '@{{{0}}}'.format(corrs.index.name)),
    #]
    hover.tooltips = None
    corrs_fig.add_tools(hover)

    return final_layout