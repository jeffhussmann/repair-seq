var i, index, indices, j, k, key, len, len1, ref, ref1, sgRNA, values,
  indexOf = [].indexOf;

if (cb_data.index.indices.length > 0) {
  index = cb_data.index.indices[0];
  sgRNA = corrs_source.data['sgRNA'][index];
  color_source.data['second_color'] = color_source.data['color_' + sgRNA];
  if (indexOf.call(filtered_corrs_source.data['sgRNA'], sgRNA) < 0) {
    indices = corrs_source.data['ranked_indices'].slice(0, 10);
    indices.push(index);
    ref = ['x', 'y', 'sgRNA', 'color'];
    for (j = 0, len = ref.length; j < len; j++) {
      key = ref[j];
      values = corrs_source.data[key];
      filtered_corrs_source.data[key] = (function() {
        var k, len1, results;
        results = [];
        for (k = 0, len1 = indices.length; k < len1; k++) {
          i = indices[k];
          results.push(values[i]);
        }
        return results;
      })();
    }
  }
  second_scatter.outline_line_color = 'black';
  second_scatter.outline_line_alpha = 0.75;
  right_line_source.data['x'][0] = index;
  right_line_source.data['x'][1] = index;
  right_line_source.data['y'][0] = corrs_source.data['y'][index] + 0.02;
  right_line.visible = true;
} else {
  color_source.data['second_color'] = color_source.data['blank'];
  ref1 = ['x', 'y', 'sgRNA', 'color'];
  for (k = 0, len1 = ref1.length; k < len1; k++) {
    key = ref1[k];
    filtered_corrs_source.data[key] = filtered_corrs_source.data[key].slice(0, 10);
  }
  second_scatter.outline_line_color = null;
  right_line.visible = false;
}

color_source.change.emit();

corrs_source.change.emit();

filtered_corrs_source.change.emit();

right_line_source.change.emit();
