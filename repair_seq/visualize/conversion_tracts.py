import bokeh.palettes
import numpy as np
import matplotlib.pyplot as plt

import hits.utilities

import repair_seq.visualize
import repair_seq.visualize.outcome_diagrams

def draw_ssODN_configurations(pools=None,
                              tis=None,
                              draw_SNVs_on_target=True,
                              flip_target=False,
                             ):
    if pools is None:
        common_ti = tis[0]
    else:
        common_ti = pools[0].target_info
        tis = [pool.target_info for pool in pools]

    rect_height = 0.25

    fig, ax = plt.subplots(figsize=(25, 4))

    def draw_rect(x0, x1, y0, y1, alpha, color='black', fill=True):
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

    def mark_5_and_3(y, x_start, x_end, color):
        if y > 0:
            left_string = '5\''
            right_string = '3\''
        else:
            right_string = '5\''
            left_string = '3\''

        ax.annotate(left_string, (x_start, y),
                    xytext=(-5 if not flip_target else 5, 0),
                    textcoords='offset points',
                    ha='right' if not flip_target else 'left',
                    va='center',
                    color=color,
                   )

        ax.annotate(right_string, (x_end, y),
                    xytext=(5 if not flip_target else -5, 0),
                    textcoords='offset points',
                    ha='left' if not flip_target else 'right',
                    va='center',
                    color=color,
                   )

    kwargs = dict(ha='center', va='center', fontfamily='monospace',)

    offset_at_cut = 0

    ys = {
        'target_+': 0.15,
        'target_-': -0.15,
        'donor_+': 0.75,
        'donor_-': -0.75,
    }

    colors = bokeh.palettes.Set2[8]

    donor_x_min = -1
    donor_x_max = 1

    for ti, color in zip(tis, colors):
        _, offset, is_reverse_complement = ti.best_donor_target_alignment

        # For ss donors, ti.donor_sequence is the actual stranded sequence
        # supplied. There is no need to complement anything, since it is drawn
        # on the bottom with the - strand if it aligned to the reverse complement
        # of the target strandedness.

        if is_reverse_complement:
            aligned_donor_seq = ti.donor_sequence[::-1]
            donor_y = ys['donor_-']
        else:
            aligned_donor_seq = ti.donor_sequence
            donor_y = ys['donor_+']

        donor_cut_after = ti.cut_after - offset

        donor_before_cut = aligned_donor_seq[:donor_cut_after + 1]
        donor_after_cut = aligned_donor_seq[donor_cut_after + 1:]

        offset_at_this_cut = ti.cut_after - common_ti.cut_after + offset_at_cut

        for b, x in zip(donor_before_cut[::-1], np.arange(len(donor_before_cut))):
            final_x = -x + offset_at_this_cut - 0.5

            donor_x_min = min(donor_x_min, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        for b, x in zip(donor_after_cut, np.arange(len(donor_after_cut))):
            final_x = x + offset_at_this_cut + 0.5

            donor_x_max = max(donor_x_max, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        if is_reverse_complement:
            y = ys['donor_-']
            ys['donor_-'] -= 0.3
        else:
            y = ys['donor_+']
            ys['donor_+'] += 0.3

        draw_rect(-len(donor_before_cut) + offset_at_this_cut,
                  len(donor_after_cut) + offset_at_this_cut,
                  donor_y + rect_height / 2,
                  donor_y - rect_height / 2,
                  alpha=0.5,
                  fill=True,
                  color=color,
                 )

        mark_5_and_3(donor_y, -len(donor_before_cut), len(donor_after_cut), color)

        for name, info in ti.donor_SNVs['target'].items():
            x = info['position'] - ti.cut_after

            if x >= 0:
                x = -0.5 + offset_at_this_cut + x
            else:
                x = -0.5 + offset_at_this_cut + x

            draw_rect(x - 0.5, x + 0.5, donor_y - rect_height / 2, donor_y + rect_height / 2, 0.2)

    # Draw resected target.

    resect_before = int(np.abs(np.floor(donor_x_min))) + 1
    resect_after = int(np.abs(np.ceil(donor_x_max))) + 1

    x_min = -resect_before - 5
    x_max = resect_after + 5

    before_cut = common_ti.target_sequence[:common_ti.cut_after + 1][x_min:]
    after_cut = common_ti.target_sequence[common_ti.cut_after + 1:][:x_max]

    for b, x in zip(before_cut[::-1], np.arange(len(before_cut))):
        final_x = -x - offset_at_cut - 0.5

        ax.annotate(b,
                    (final_x, ys['target_+']),
                    **kwargs,
                   )

        if x < resect_before:
            alpha = 0.3
        else:
            alpha = 1

        ax.annotate(hits.utilities.complement(b),
                    (final_x, ys['target_-']),
                    alpha=alpha,
                    **kwargs,
                   )

    for b, x in zip(after_cut, np.arange(len(after_cut))):
        final_x = x + offset_at_cut + 0.5

        if x < resect_after:
            alpha = 0.3
        else:
            alpha = 1

        ax.annotate(b,
                    (final_x, ys['target_+']),
                    alpha=alpha,
                    **kwargs,
                   )

        ax.annotate(hits.utilities.complement(b),
                    (final_x, ys['target_-']),
                    **kwargs,
                   )

    alpha = 0.1
    draw_rect(offset_at_cut + resect_after, x_max, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(0, x_max, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

    draw_rect(0, x_min, ys['target_+'] - rect_height / 2, ys['target_+'] + rect_height / 2, alpha)
    draw_rect(-offset_at_cut - resect_before, x_min, ys['target_-'] - rect_height / 2, ys['target_-'] + rect_height / 2, alpha)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-2, 2)

    if flip_target:
        ax.invert_xaxis()
        ax.invert_yaxis()

    ax.plot([0, 0], [ys['target_-'] - rect_height, ys['target_+'] + rect_height], color='black', linestyle='--', alpha=0.5)

    mark_5_and_3(ys['target_+'], x_min, x_max, 'black')
    mark_5_and_3(ys['target_-'], x_min, x_max, 'black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if draw_SNVs_on_target:
        for name, info in common_ti.donor_SNVs['target'].items():
            x = info['position'] - ti.cut_after

            if x >= 0:
                x = -0.5 + offset_at_cut + x
            else:
                x = -0.5 - offset_at_cut + x

            for y in [ys['target_+'], ys['target_-']]:
                draw_rect(x - 0.5, x + 0.5, y - rect_height / 2, y + rect_height / 2, 0.2)

    for name, PAM_slice in common_ti.PAM_slices.items():
        sgRNA = common_ti.sgRNA_features[name]
        if sgRNA.strand == '+':
            y_start = ys['target_+'] + rect_height / 4
            y_end = y_start + rect_height / 3
        else:
            y_start = ys['target_-'] - rect_height / 4
            y_end = y_start - rect_height / 3

        x_start = PAM_slice.start - common_ti.cut_after - 1 + 0.2 # offset empirically determined, confused by it
        x_end = PAM_slice.stop - common_ti.cut_after - 1 - 0.2

        PAM_feature = common_ti.PAM_features[common_ti.target, f'{name}_PAM']
        PAM_color = PAM_feature.attribute['color']

        #draw_rect(x_start, x_end, ys['target_-'] - rect_height / 2, ys['target_+'] + rect_height / 2, 0.9, color=PAM_color, fill=False)
        ax.plot([x_start, x_start, x_end, x_end], [y_start, y_end, y_end, y_start], color=PAM_color, linewidth=2)
        
        x_start = sgRNA.start - common_ti.cut_after - 1 + 0.1
        x_end = sgRNA.end - common_ti.cut_after - 0.1
        ax.plot([x_start, x_start, x_end, x_end], [y_start, y_end, y_end, y_start], color=sgRNA.attribute['color'], linewidth=2)
        #draw_rect(x_start, x_end, ys['target_-'] - rect_height / 2, ys['target_+'] + rect_height / 2, 0.5, color=sgRNA.attribute['color'], fill=False)

            
    return fig, ax

def conversion_tracts(pool,
                      heatmap_genes=None,
                      plot_genes=None,
                      guides=None,
                      fc_ylims=None,
                      fc_xlims=(-4, 2),
                      frequency_threshold=0.002,
                      outcomes=None,
                      gene_to_sort_by='MLH1',
                      x_lims=None,
                      just_heatmaps=False,
                      draw_labels=True,
                      draw_conversion_plots=True,
                      gene_to_color=None,
                      diagram_kwargs=None,
                      pools_for_diagram=None,
                      flip_target=False,
                      ax_on_bottom=False,
                      **kwargs,
                     ):
    fracs = pool.non_targeting_fractions()
    if outcomes is not None:
        fracs = fracs.reindex(outcomes, fill_value=0)

    if pools_for_diagram is None:
        pools_for_diagram = [pool]

    outcomes = [(c, s, d) for (c, s, d), f in fracs.items() if c == 'donor' and f > frequency_threshold]

    fig, configuration_ax = draw_ssODN_configurations(pools_for_diagram, flip_target=flip_target, draw_SNVs_on_target=False)

    xs = pool.SNV_name_to_position - pool.target_info.cut_after - 0.5

    # overall incorporation frequency plot

    c_ax_x_min, c_ax_x_max = configuration_ax.get_xlim()
    data_width = c_ax_x_max - c_ax_x_min

    c_ax_p = configuration_ax.get_position()

    if x_lims is not None:
        x_min, x_max = x_lims
    else:
        x_min = int(np.floor(min(xs))) - 2
        x_max = int(np.ceil(max(xs))) + 2

    left = c_ax_p.x0 + ((x_min - c_ax_x_min) / data_width * c_ax_p.width)
    width = abs((x_max - x_min) / data_width * c_ax_p.width)
    height = 0.5 * c_ax_p.height

    gene_guides_by_activity = pool.gene_guides_by_activity()

    if gene_to_color is None:
        gene_to_color = {gene: repair_seq.visualize.good_colors[i] for i, gene in enumerate(heatmap_genes)}

    if draw_conversion_plots:
        frequency_ax = fig.add_axes([left, c_ax_p.y0 - height, width, height])

        ys = pool.conversion_fractions['all_non_targeting']
        frequency_ax.plot(xs, ys * 100, 'o-', color='black', linewidth=2)

        frequency_ax.axvline(0, linestyle='--', color='black', alpha=0.5)
        
        frequency_ax.set_ylim(0, max(ys * 100) * 1.1)

        frequency_ax.set_ylabel('overall\nconversion\npercentage', size=12)

        frequency_ax.set_xlim(x_min, x_max)

        plt.setp(frequency_ax.get_xticklabels(), visible=False)

        # log2 fold changes plot

        f_ax_p = frequency_ax.get_position()
        height = 0.75 * c_ax_p.height
        gap = height * 0.1
        fold_change_ax = fig.add_axes([f_ax_p.x0, f_ax_p.y0 - gap - height, f_ax_p.width, height], sharex=frequency_ax)

        guide_sets = [
            ('negative_control',
            pool.variable_guide_library.non_targeting_guides,
            'non-targeting',
            dict(color='black', alpha=0.2),
            ),
        ]

        for gene_i, gene in enumerate(plot_genes):
            guide_sets.append((gene,
                            gene_guides_by_activity[gene][:1],
                            None,
                            dict(color=gene_to_color[gene], alpha=0.8, linewidth=2.5, markersize=10),
                            ),
                            )

        max_y = 1
        min_y = -2
        
        for gene_i, (gene, gene_guides, label, gene_kwargs) in enumerate(guide_sets):

            fold_change_ax.annotate(gene,
                                    xy=(1, 1),
                                    xycoords='axes fraction',
                                    xytext=(5, -10 - 16 * gene_i),
                                    textcoords='offset points',
                                    color=gene_kwargs['color'],
                                    size=14,
                                )

            for i, guide in enumerate(gene_guides):
                ys = pool.conversion_log2_fold_changes[guide]

                max_y = np.ceil(max(max_y, max(ys)))
                min_y = np.floor(min(min_y, min(ys)))

                label_to_use = None

                if i == 0:
                    if label is None:
                        label_to_use = gene
                    else:
                        label_to_use = label
                else:
                    label_to_use = ''

                fold_change_ax.plot(xs, ys, '.-', label=label_to_use, **gene_kwargs)

        plt.setp(fold_change_ax.get_xticklabels(), visible=False)

        if fc_ylims is None:
            fc_ylims = (max(-6, min_y), min(5, max_y))
        
        fold_change_ax.set_ylim(*fc_ylims)

        #fold_change_ax.set_yticks(np.arange(-3, 2))
        fold_change_ax.grid(alpha=0.5, axis='y')

        fold_change_ax.axhline(0, color='black', alpha=0.2)
        fold_change_ax.axvline(0, linestyle='--', color='black', alpha=0.5)

        fold_change_ax.set_ylabel('log2 fold-change\nin conversion\nfrom non-targeting', size=12)

        fc_ax_p = fold_change_ax.get_position()

    # Make height such that nts are roughly square, with a slop factor for spacing between rows.
    fig_width_inches, fig_height_inches = fig.get_size_inches()

    #width_inches = fc_ax_p.width * fig_width_inches
    width_inches = width * fig_width_inches
    height_inches = width_inches * len(outcomes) / abs(x_max - x_min)

    #diagram_width = fc_ax_p.width
    diagram_width = width
    diagram_height = height_inches / fig_height_inches * 2

    #diagram_left = fc_ax_p.x0
    diagram_left = left

    if draw_conversion_plots:
        diagram_bottom = fc_ax_p.y0 - 0.4 * fc_ax_p.height - diagram_height
    else:
        diagram_bottom = 0
    #diagram_bottom = c_ax_p.y0 - 0.4 * c_ax_p.height - diagram_height

    diagram_rect = [diagram_left, diagram_bottom, diagram_width, diagram_height]

    diagram_ax = fig.add_axes(diagram_rect, sharex=frequency_ax if draw_conversion_plots else None)

    if gene_to_sort_by is None:
        sorted_outcomes = outcomes[::-1]
    else:
        sorted_outcomes = pool.sort_outcomes_by_gene_phenotype(outcomes, gene_to_sort_by)[::-1]

    if diagram_kwargs is None:
        diagram_kwargs = {
            'flip_if_reverse': False,
        }
        
    diagram_kwargs.update(dict(
        window=(x_min, x_max),
        preserve_x_lims=True,
        shift_x=-0 if just_heatmaps else 0.5,
        draw_donor_on_top=True,
        draw_wild_type_on_top=True,
    ))

    # Note: plot does a weird flip of outcomes
    diagram_grid = repair_seq.visualize.outcome_diagrams.DiagramGrid(sorted_outcomes[::-1],
                                                              pool.target_info,
                                                              diagram_ax=diagram_ax,
                                                              ax_on_bottom=ax_on_bottom,
                                                              **diagram_kwargs,
                                                             )

    diagram_grid.add_ax('log10 frequency',
                        side=kwargs.get('frequency_side', 'left'),
                        width_multiple=4,
                        gap_multiple=2.5,
                        title='Percentage of\noutcomes for\n non-targeting\nsgRNAs' if draw_labels else '',
                        title_size=12,
                       )
    
    diagram_grid.add_ax('log2 fold change',
                        side='right',
                        width_multiple=7,
                        gap_multiple=1.5,
                        title='Log$_2$ fold change\nfrom non-targeting' if draw_labels else '',
                        title_size=12,
                       )

    log10_frequencies = np.log10(pool.non_targeting_fractions().loc[sorted_outcomes])
    diagram_grid.plot_on_ax('log10 frequency', log10_frequencies, marker='o', markersize=2.5, linewidth=1, line_alpha=0.9, marker_alpha=0.9, color='black', clip_on=False)

    x_min, x_max = np.log10(0.95 * frequency_threshold), np.log10(0.21)
    diagram_grid.axs_by_name['log10 frequency'].set_xlim(x_min, x_max)
    diagram_grid.style_log10_frequency_ax('log10 frequency')
    if kwargs.get('frequency_side', 'left') == 'left':
        diagram_grid.axs_by_name['log10 frequency'].invert_xaxis()

    fcs = pool.log2_fold_changes().loc[sorted_outcomes]

    for gene in plot_genes:
        if gene == 'DNA2':
            # override since growth phenotype leads to low UMI counts for strong guides
            guide = 'DNA2_1'
        elif gene == 'MLH1':
            guide = 'MLH1_1'
        elif gene == 'PMS2':
            guide = 'PMS2_1'
        elif gene == 'MSH6':
            guide = 'MSH6_1'
        elif gene == 'RBBP8':
            guide = 'RBBP8_1'
        elif gene == 'NBN':
            guide = 'NBN_1',
        elif gene == 'MRE11':
            guide = 'MRE11_1'
        else:
            guide = gene_guides_by_activity[gene][0]
            print(guide)

        diagram_grid.plot_on_ax('log2 fold change', fcs[guide], marker='o', markersize=2.5, marker_alpha=0.9, line_alpha=0.9, linewidth=1, color=gene_to_color[gene], clip_on=False)

    diagram_grid.style_fold_change_ax('log2 fold change')
    diagram_grid.axs_by_name['log2 fold change'].set_xlim(*fc_xlims)

    for gene_i, gene in enumerate(heatmap_genes):
        if gene == 'DNA2':
            # override since growth phenotype leads to low UMI counts for strong guides
            guides = ['DNA2_1', 'DNA2_3']
        elif gene == 'MLH1':
            guides = ['MLH1_1', 'MLH1_2']
        elif gene == 'PMS2':
            guides = ['PMS2_1', 'PMS2_2']
        elif gene == 'MSH6':
            guides = ['MSH6_1', 'MSH6_2']
        elif gene == 'RBBP8':
            guides = ['RBBP8_1', 'RBBP8_2']
        elif gene == 'NBN':
            guides = ['NBN_1', 'NBN_2']
        elif gene == 'MRE11':
            guides = ['MRE11_1', 'MRE11_3']
        else:
            guides = gene_guides_by_activity[gene][:2]
        vals = fcs[guides]

        if gene_i == 0:
            gap_multiple = 1
        else:
            gap_multiple = 0.25

        heatmap_ax = diagram_grid.add_heatmap(vals, f'heatmap {gene}', gap_multiple=gap_multiple, color=gene_to_color[gene])

        if not draw_labels:
            heatmap_ax.set_xticklabels([])

    if just_heatmaps:
        fig.delaxes(configuration_ax)
        if draw_conversion_plots:
            fig.delaxes(frequency_ax)
            fig.delaxes(fold_change_ax)

    return fig, diagram_grid