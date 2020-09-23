import bokeh.palettes
import numpy as np
import matplotlib.pyplot as plt

import hits.utilities

import ddr.visualize
import ddr.visualize.outcome_diagrams

def draw_ssODN_configurations(pools=None, tis=None):
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
                    xytext=(-5, 0),
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    color=color,
                   )

        ax.annotate(right_string, (x_end, y),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
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

        for b, x in zip(donor_before_cut[::-1], np.arange(len(donor_before_cut))):
            final_x = -x - offset_at_cut - 0.5

            donor_x_min = min(donor_x_min, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        for b, x in zip(donor_after_cut, np.arange(len(donor_after_cut))):
            final_x = x + offset_at_cut + 0.5

            donor_x_max = max(donor_x_max, final_x)

            ax.annotate(b,
                        (final_x, donor_y),
                        **kwargs,
                       )

        if is_reverse_complement:
            y = ys['donor_-']
        else:
            y = ys['donor_+']

        draw_rect(-len(donor_before_cut),
                  len(donor_after_cut),
                  donor_y + rect_height / 2,
                  donor_y - rect_height / 2,
                  alpha=0.5,
                  fill=True,
                  color=color,
                 )

        mark_5_and_3(donor_y, -len(donor_before_cut), len(donor_after_cut), color)

        for name, info in common_ti.donor_SNVs['target'].items():
            x = info['position'] - ti.cut_after

            if x >= 0:
                x = -0.5 + offset_at_cut + x
            else:
                x = -0.5 - offset_at_cut + x

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

    ax.plot([0, 0], [ys['target_-'] - rect_height, ys['target_+'] + rect_height], color='black', linestyle='--', alpha=0.5)

    mark_5_and_3(ys['target_+'], x_min, x_max, 'black')
    mark_5_and_3(ys['target_-'], x_min, x_max, 'black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

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
            y_start = ys['target_+'] - rect_height / 4
            y_end = y_start - rect_height / 4
        else:
            y_start = ys['target_-'] + rect_height / 4
            y_end = y_start + rect_height / 4

        x_start = PAM_slice.start - common_ti.cut_after - 1 # offset empirically determined, confused by it
        x_end = PAM_slice.stop - common_ti.cut_after - 1

        ax.plot([x_start, x_start, x_end, x_end], [y_start, y_end, y_end, y_start])
            
    return fig, ax

def conversion_tracts(pool, genes=None, guides=None, fc_ylims=None, outcomes=None):
    fig, configuration_ax = draw_ssODN_configurations([pool])

    xs = pool.SNV_name_to_position - pool.target_info.cut_after - 0.5

    # overall incorporation frequency plot

    c_ax_x_min, c_ax_x_max = configuration_ax.get_xlim()
    data_width = c_ax_x_max - c_ax_x_min

    c_ax_p = configuration_ax.get_position()

    x_min = int(np.floor(min(xs))) - 2
    x_max = int(np.ceil(max(xs))) + 2

    left = c_ax_p.x0 + ((x_min - c_ax_x_min) / data_width * c_ax_p.width)
    width = (x_max - x_min) / data_width * c_ax_p.width

    frequency_ax = fig.add_axes([left, c_ax_p.y0 - 0.5 * c_ax_p.height, width, 0.5 * c_ax_p.height])

    ys = pool.conversion_fractions['all_non_targeting']
    frequency_ax.plot(xs, ys * 100, 'o-', color='black', linewidth=2)

    frequency_ax.axvline(0, linestyle='--', color='black', alpha=0.5)
    
    frequency_ax.set_ylim(0, max(ys * 100) * 1.1)

    frequency_ax.set_ylabel('overall\nconversion\npercentage')

    frequency_ax.set_xlim(x_min, x_max)

    # log2 fold changes plot

    f_ax_p = frequency_ax.get_position()
    fold_change_ax = fig.add_axes([f_ax_p.x0, f_ax_p.y0 - 2.2 * f_ax_p.height, f_ax_p.width, 2 * f_ax_p.height], sharex=frequency_ax)

    guide_sets = [
        ('negative_control', pool.variable_guide_library.non_targeting_guides, 'non-targeting', dict(color='black', alpha=0.2)),
    ]

    gene_to_color = {}

    if genes is not None:
        for gene_i, gene in enumerate(genes):
            gene_to_color[gene] = ddr.visualize.good_colors[gene_i]

            guide_sets.append((gene,
                               pool.variable_guide_library.gene_guides(gene),
                               None,
                               dict(color=gene_to_color[gene], alpha=0.8, linewidth=1.5),
                              )
                             )
    elif guides is not None:
        for gene_i, guide in enumerate(guides):
            guide_sets.append((pool.variable_guide_library.guide_to_gene[guide],
                               [guide],
                               None,
                               dict(color=ddr.visualize.good_colors[gene_i], alpha=0.8, linewidth=1.5),
                              )
                             )

    max_y = 1
    min_y = -2
    
    for gene, gene_guides, label, kwargs in guide_sets:
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

            fold_change_ax.plot(xs, ys, '.-', label=label_to_use, **kwargs)

    fold_change_ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.setp(fold_change_ax.get_xticklabels(), visible=False)

    if fc_ylims is None:
        fc_ylims = (max(-6, min_y), min(5, max_y))
    
    fold_change_ax.set_ylim(*fc_ylims)

    fold_change_ax.axhline(0, color='black', alpha=0.2)
    fold_change_ax.axvline(0, linestyle='--', color='black', alpha=0.5)

    fold_change_ax.set_ylabel('log2 fold-change in conversion from non-targeting')

    fc_ax_p = fold_change_ax.get_position()

    # Make height such that nts are roughly square, with a slop factor for spacing between rows.
    fig_width_inches, fig_height_inches = fig.get_size_inches()

    width_inches = fc_ax_p.width * fig_width_inches
    height_inches = width_inches * len(outcomes) / (x_max - x_min)

    diagram_width = fc_ax_p.width
    diagram_height = height_inches / fig_height_inches * 2

    diagram_left = fc_ax_p.x0
    diagram_bottom = fc_ax_p.y0 - 0.1 * fc_ax_p.height - diagram_height

    diagram_rect = [diagram_left, diagram_bottom, diagram_width, diagram_height]

    diagram_ax = fig.add_axes(diagram_rect, sharex=frequency_ax)

    sorted_outcomes = pool.sort_outcomes_by_gene_phenotype(outcomes, 'MLH1')[::-1]

    diagram_kwargs = dict(window=(x_min, x_max), preserve_x_lims=True, shift_x=-0.5)
    # Note: plot does a weird flip of outcomes
    diagram_grid = ddr.visualize.outcome_diagrams.DiagramGrid(sorted_outcomes[::-1], pool.target_info, diagram_ax=diagram_ax, **diagram_kwargs)

    diagram_grid.plot_diagrams()

    diagram_grid.add_ax('log10 frequency')

    log10_frequencies = np.log10(pool.non_targeting_fractions('perfect', 'none').loc[sorted_outcomes])
    diagram_grid.plot_on_ax('log10 frequency', log10_frequencies, marker='o')
    diagram_grid.style_log10_frequency_ax('log10 frequency')

    diagram_grid.add_ax('log2 fold change')

    fcs = pool.log2_fold_changes('perfect', 'none')['none'].loc[sorted_outcomes]

    for gene in genes:
        diagram_grid.plot_on_ax('log2 fold change', fcs[f'{gene}_1'], marker='o', markersize=2, color=gene_to_color[gene])

    for gene_i, gene in enumerate(genes):
        guides = pool.variable_guide_library.gene_guides(gene)
        vals = fcs[guides]

        if gene_i == 0:
            gap_multiple = 1
        else:
            gap_multiple = 0.5

        diagram_grid.add_heatmap(vals, gap_multiple, color=gene_to_color[gene])

    return fig