import copy

import bokeh.palettes
import matplotlib.pyplot as plt
import pandas as pd

correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
correlation_cmap.set_bad('white')

fold_changes_cmap = copy.copy(plt.get_cmap('RdBu_r'))

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