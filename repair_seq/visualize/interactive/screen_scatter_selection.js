var i, indices, key, ref, values;

indices = cb_obj.indices;

ref = scatter_source.data;
for (key in ref) {
  values = ref[key];
  filtered_source.data[key] = (function() {
    var j, len, results;
    results = [];
    for (j = 0, len = indices.length; j < len; j++) {
      i = indices[j];
      results.push(values[i]);
    }
    return results;
  })();
}

filtered_source.change.emit();
