var cutoff, i, indices, p, p_vals;

p_vals = scatter_source.data['gene_p_{direction}'];

cutoff = Math.pow(10, cutoff_slider.value);

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

scatter_source.selected.indices = indices;

scatter_source.change.emit();
