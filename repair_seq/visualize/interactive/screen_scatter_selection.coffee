models = cb_obj.document._all_models_by_name._dict

indices = cb_obj.indices

full_data = models['scatter_source'].data
filtered_data = models['filtered_source'].data

for key, values of full_data
    filtered_data[key] = (values[i] for i in indices)

models['filtered_source'].change.emit()
