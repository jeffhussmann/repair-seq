var i, index, j, key, len, name, range, ref, ref1, squeeze, top_indices, values, y_vals;

squeeze = function(possibly_array) {
  var squeezed;
  if (Array.isArray(possibly_array)) {
    squeezed = possibly_array[0];
  } else {
    squeezed = possibly_array;
  }
  return squeezed;
};

name = squeeze(color_menu.value);

color_source.data['first_color'] = color_source.data['color_' + name];

y_vals = corrs_source.data[name];

corrs_source.data['y'] = y_vals;

range = (function() {
  var results = [];
  for (var j = 0, ref = y_vals.shape[0]; 0 <= ref ? j <= ref : j >= ref; 0 <= ref ? j++ : j--){ results.push(j); }
  return results;
}).apply(this);

corrs_source.data['ranked_indices'] = range.sort((a, b) => y_vals[b] - y_vals[a]);

top_indices = corrs_source.data['ranked_indices'].slice(0, 10);

ref1 = ['x', 'y', 'sgRNA', 'color'];
for (j = 0, len = ref1.length; j < len; j++) {
  key = ref1[j];
  values = corrs_source.data[key];
  filtered_corrs_source.data[key] = (function() {
    var k, len1, results;
    results = [];
    for (k = 0, len1 = top_indices.length; k < len1; k++) {
      i = top_indices[k];
      results.push(values[i]);
    }
    return results;
  })();
}

corrs_y_axis[0].axis_label = 'Correlation with ' + name;

tabs.active = 5;

index = corrs_source.data['sgRNA'].indexOf(name);

left_line_source.data['x'][0] = index;

left_line_source.data['x'][1] = index;

color_source.change.emit();

corrs_source.change.emit();

filtered_corrs_source.change.emit();

left_line_source.change.emit();
