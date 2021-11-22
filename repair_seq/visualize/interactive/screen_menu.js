var dataset_name, full_name, i, key, len, max, nt_fraction_values, outcome_name, ref;

dataset_name = dataset_menu.value[0];

outcome_name = outcome_menu.value[0];

console.log('here');

nt_fraction_values = {nt_fractions};

ref = ['frequency', 'percentage', 'ys', 'log2_fold_change', 'total_UMIs', 'gene_p_up', 'gene_p_down'];
// 21.10.23: Note that it is important to update gene_p_* even
// if they aren't showed in table since they are used by filter
// buttons.
for (i = 0, len = ref.length; i < len; i++) {
  key = ref[i];
  full_name = dataset_name + '_' + outcome_name + '_' + key;
  scatter_source.data[key] = scatter_source.data[full_name];
  filtered_source.data[key] = filtered_source.data[full_name];
}

full_name = dataset_name + '_' + outcome_name;

nt_fraction.location = nt_fraction_values[full_name];

title.text = 'Screen condition: ' + dataset_name;

subtitle.text = 'Outcome category: ' + outcome_name;

max = Math.max(...scatter_source.data['percentage']);

y_range.start = 0;

y_range.end = max * 1.2;

y_range.change.emit();

title.change.emit();

subtitle.change.emit();

nt_fraction.change.emit();

scatter_source.change.emit();

filtered_source.change.emit();
