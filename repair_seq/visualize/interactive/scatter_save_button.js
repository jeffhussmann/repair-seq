var column_names, csv_content, encoded, filtered_data, first_name, i, j, length, line, lines, link, models, name, ref;

models = cb_obj.origin.document._all_models_by_name._dict;

column_names = {column_names};

filtered_data = models['filtered_source'].data;

// learned from http://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side
csv_content = "data:text/csv;charset=utf-8,";

lines = [column_names.join('\t')];

first_name = column_names[0];

length = filtered_data[first_name].length;

for (i = j = 0, ref = length; (0 <= ref ? j < ref : j > ref); i = 0 <= ref ? ++j : --j) {
  line = ((function() {
    var k, len, results;
    results = [];
    for (k = 0, len = column_names.length; k < len; k++) {
      name = column_names[k];
      results.push(filtered_data[name][i]);
    }
    return results;
  })()).join('\t');
  lines.push(line);
}

csv_content += lines.join('\n');

encoded = encodeURI(csv_content);

link = document.createElement('a');

link.setAttribute('href', encoded);

link.setAttribute('download', 'table_contents.txt');

link.click();
