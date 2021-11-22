indices = cb_obj.indices

for key, values of scatter_source.data
    filtered_source.data[key] = (values[i] for i in indices)

filtered_source.change.emit()