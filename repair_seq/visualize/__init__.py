import copy
from collections import defaultdict

import bokeh.palettes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import knock_knock.outcome

from knock_knock.visualize import fold_changes_cmap

targeting_guide_color = 'silver'
nontargeting_guide_color = bokeh.palettes.Greys9[2]

correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
correlation_cmap.set_bad('white')

cell_cycle_cmap = copy.copy(plt.get_cmap('PiYG_r'))

gamma_cmap = copy.copy(plt.get_cmap('magma'))

category10 = bokeh.palettes.Category10[10]
accent = bokeh.palettes.Accent[8]
good_colors = (category10[2:7] + category10[8:] + accent[:3] + accent[4:])*100

colors_list = [[c]*10 for c in good_colors]

def make_guide_to_color(guide_to_gene, individual_genes, gene_sets, only_special=False, palette=None):
    guide_to_color = {}
    gene_to_color = {}

    if palette is None:
        palette = [f'C{i}' for i in range(10)]
    
    gene_set_colors = {name: color for name, color in zip(gene_sets, palette)}
    
    for name, genes in gene_sets.items():
        for gene in genes:
            gene_to_color[gene] = gene_set_colors[name]
            
    for guide, gene in guide_to_gene.items():
        if gene in individual_genes or gene in gene_to_color:
            if gene not in gene_to_color:
                gene_to_color[gene] = 'black'
            
        else:
            if not only_special:
                gene_to_color[gene] = 'silver'
            
        if gene in gene_to_color:
            guide_to_color[guide] = gene_to_color[gene]
        
    return pd.Series(gene_to_color), pd.Series(guide_to_color)

Cas9_category_color_order = [
    'deletion, ambiguous',
    'insertion with deletion',
    'deletion, PAM-distal',
    'deletion, PAM-proximal',
    'deletion, bidirectional',
    'genomic insertion',
    'insertion',
    'wild type',
]

Cas9_category_palette = bokeh.palettes.Dark2[8]
Cas9_category_colors = dict(zip(Cas9_category_color_order, Cas9_category_palette))

Cas9_category_aliases = {n: n for n in Cas9_category_color_order}
Cas9_category_aliases.update({
    'wild type': 'unedited',
    'deletion, bidirectional': 'bidirectional deletions',
    'deletion, PAM-distal': 'deletions on only PAM-distal side',
    'deletion, PAM-proximal': 'deletions on only PAM-proximal side',
    'genomic insertion': 'capture of genomic sequence at break',
    'deletion, ambiguous': 'deletions consistent with either side',
})

Cas9_category_display_order = [
    'wild type',
    'deletion, bidirectional',
    'deletion, PAM-distal',
    'deletion, PAM-proximal',
    'deletion, ambiguous',
    'insertion',
    'insertion with deletion',
    'genomic insertion',
]

Cas9_category_alias_colors = copy.copy(Cas9_category_colors)
for name, alias in Cas9_category_aliases.items():
    color = Cas9_category_alias_colors.pop(name)
    Cas9_category_alias_colors[alias] = color

Cpf1_category_display_order = [
    'wild type',
    'deletion, spans both nicks',
    'deletion, spans PAM-distal nick',
    'deletion, spans PAM-proximal nick',
    'insertion',
    'insertion with deletion',
    'genomic insertion',
]

Cpf1_category_colors = {n: c for n, c in Cas9_category_colors.items() if n in Cpf1_category_display_order}
Cpf1_category_colors.update({
    'deletion, spans both nicks': bokeh.palettes.Set1[8][2],
    'deletion, spans PAM-distal nick': bokeh.palettes.Set1[8][0],
    'deletion, spans PAM-proximal nick': bokeh.palettes.Set1[8][3],
    'deletion, spans neither': 'gray',
})

Cpf1_category_aliases = {n: n for n in Cpf1_category_display_order}
Cpf1_category_aliases.update({
    'wild type': 'unedited',
    'deletion, spans both nicks': 'deletions spanning both nicks',
    'deletion, spans PAM-distal nick': 'deletions spanning only PAM-distal nick',
    'deletion, spans PAM-proximal nick': 'deletions spanning only PAM-proximal nick',
    'deletion, spans neither': 'deletions spanning neither nick',
    'genomic insertion': 'capture of genomic sequence at break',
})

Cpf1_category_alias_colors = copy.copy(Cpf1_category_colors)
for name, alias in Cpf1_category_aliases.items():
    color = Cpf1_category_alias_colors.pop(name)
    Cpf1_category_alias_colors[alias] = color

category_aliases = {
    'SpCas9': Cas9_category_aliases,
    'Cpf1': Cpf1_category_aliases,
    'AsCas12a': Cpf1_category_aliases,
}

category_colors = {
    'SpCas9': Cas9_category_colors,
    'Cpf1': Cpf1_category_colors,
    'AsCas12a': Cpf1_category_colors,
}

category_alias_colors = {
    'SpCas9': Cas9_category_alias_colors,
    'Cpf1': Cpf1_category_alias_colors,
    'AsCas12a': Cpf1_category_alias_colors,
}

category_display_order = {
    'SpCas9': Cas9_category_display_order,
    'Cpf1': Cpf1_category_display_order,
    'AsCas12a': Cpf1_category_display_order,
}
def assign_category_colors(outcomes, target_info):
    combined_categories = knock_knock.outcome.add_directionalities_to_deletions(outcomes, target_info)
    category_to_color = category_colors[target_info.effector.name]
    colors = [category_to_color[c] for c in combined_categories]
    return colors

def add_alphabetical_x_ticks(ax, df):
    ''' Given a DataFrame df whose index is guides in alphabetical order 
    and an ax that has a column from df plotted on its y-axis, add ticks
    to partition the x-axis into spans of guides starting with each letter.
    '''
    letter_xs = defaultdict(list)

    for x, guide in enumerate(df.index):
        first_letter = guide[0]
        letter_xs[first_letter].append(x)

    first_letter_xs = [min(xs) for letter, xs in sorted(letter_xs.items())]
    boundaries = list(np.array(first_letter_xs) - 0.5) + [len(df) - 1 + 0.5]
    mean_xs = {letter: np.mean(xs) for letter, xs in letter_xs.items()}

    ax.set_xticks(boundaries[1:-1])
    ax.set_xticklabels([])

    for letter, x in mean_xs.items():
        if letter == 'n':
            letter = 'non-\ntargeting'

        ax.annotate(letter,
                    xy=(x, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -18),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                   )

    ax.set_xlim(-0.001 * len(df), 1.001 * len(df))