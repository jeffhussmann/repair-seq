var cutoff, data, i, indices, models, p, p_vals;

models = cb_obj.origin.document._all_models_by_name._dict;

data = models['scatter_source'].data;

if (cb_obj.origin.name === 'filter_down') {
  p_vals = data['gene_p_down'];
} else {
  p_vals = data['gene_p_up'];
}

cutoff = Math.pow(10, models['cutoff_slider'].value);

indices = (function() {
  var j, len, results;
  results = [];
  for (i = j = 0, len = p_vals.length; j < len; i = ++j) {
    p = p_vals[i];
    if (p <= cutoff) {
      results.push(i);
    }
  }
  return results;
})();

console.log(models);

models['scatter_source'].selected.indices = indices;

models['scatter_source'].change.emit();
