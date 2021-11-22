p_vals = scatter_source.data['gene_p_{direction}']

cutoff = Math.pow(10, cutoff_slider.value)

indices = (i for p, i in p_vals when p <= cutoff)

scatter_source.selected.indices = indices
scatter_source.change.emit()