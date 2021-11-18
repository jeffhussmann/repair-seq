var dataset_name, full_name, i, key, label_data, len, max, models, nt_fraction_values, outcome_name, ref, scatter_data;

models = cb_obj.document._all_models_by_name._dict;

scatter_data = models['scatter_source'].data;

label_data = models['filtered_source'].data;

dataset_name = models['dataset_menu'].value[0];

outcome_name = models['outcome_menu'].value[0];

nt_fraction_values = {nt_fractions};

ref = ['frequency', 'percentage', 'ys', 'log2_fold_change', 'total_UMIs', 'gene_p_up', 'gene_p_down'];
// 21.10.23: Note that it is important to update gene_p_* even
// if they aren't showed in table since they are used by filter
// buttons.
for (i = 0, len = ref.length; i < len; i++) {
  key = ref[i];
  full_name = dataset_name + '_' + outcome_name + '_' + key;
  scatter_data[key] = scatter_data[full_name];
  label_data[key] = label_data[full_name];
}

full_name = dataset_name + '_' + outcome_name;

models['nt_fraction'].location = nt_fraction_values[full_name];

models['title'].text = 'Screen condition: ' + dataset_name;

models['subtitle'].text = 'Outcome category: ' + outcome_name;

max = Math.max(...scatter_data['percentage']);

models['y_range'].start = 0;

models['y_range'].end = max * 1.2;

models['y_range'].change.emit();

models['title'].change.emit();

models['subtitle'].change.emit();

models['nt_fraction'].change.emit();

models['scatter_source'].change.emit();

models['filtered_source'].change.emit();
