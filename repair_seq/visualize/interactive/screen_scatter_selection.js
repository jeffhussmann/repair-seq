var filtered_data, full_data, i, indices, key, models, values;

models = cb_obj.document._all_models_by_name._dict;

indices = cb_obj.indices;

full_data = models['scatter_source'].data;

filtered_data = models['filtered_source'].data;

for (key in full_data) {
  values = full_data[key];
  filtered_data[key] = (function() {
    var j, len, results;
    results = [];
    for (j = 0, len = indices.length; j < len; j++) {
      i = indices[j];
      results.push(values[i]);
    }
    return results;
  })();
}

models['filtered_source'].change.emit();
