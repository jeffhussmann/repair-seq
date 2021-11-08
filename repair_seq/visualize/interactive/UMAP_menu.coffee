squeeze = (possibly_array) ->
    if Array.isArray(possibly_array)
        squeezed = possibly_array[0]
    else
        squeezed = possibly_array
    return squeezed

name = squeeze color_menu.value

color_source.data['first_color'] = color_source.data['color_' + name]
y_vals = corrs_source.data[name]
corrs_source.data['y'] = y_vals

range = [0..y_vals.shape[0]]
`corrs_source.data['ranked_indices'] = range.sort((a, b) => y_vals[b] - y_vals[a])`

top_indices = corrs_source.data['ranked_indices'].slice(0, 10)

for key in ['x', 'y', 'sgRNA', 'color']
    values = corrs_source.data[key]
    filtered_corrs_source.data[key] = (values[i] for i in top_indices)

corrs_y_axis[0].axis_label = 'Correlation with ' + name

tabs.active = 5

index = corrs_source.data['sgRNA'].indexOf(name)
left_line_source.data['x'][0] = index
left_line_source.data['x'][1] = index

color_source.change.emit()
corrs_source.change.emit()
filtered_corrs_source.change.emit()
left_line_source.change.emit()