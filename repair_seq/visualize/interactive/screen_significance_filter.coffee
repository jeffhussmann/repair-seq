models = cb_obj.origin.document._all_models_by_name._dict

data = models['scatter_source'].data

if cb_obj.origin.name == 'filter_down'
    p_vals = data['gene_p_down']
else
    p_vals = data['gene_p_up']

cutoff = Math.pow(10, models['cutoff_slider'].value)

indices = (i for p, i in p_vals when p <= cutoff)

console.log(models)

models['scatter_source'].selected.indices = indices
models['scatter_source'].change.emit()
