if cb_data.index.indices.length > 0
    index = cb_data.index.indices[0]
    
    sgRNA = corrs_source.data['sgRNA'][index]
    
    color_source.data['second_color'] = color_source.data['color_' + sgRNA]

    if sgRNA not in filtered_corrs_source.data['sgRNA']
        indices = corrs_source.data['ranked_indices'].slice(0, 10)
        indices.push(index)

        for key in ['x', 'y', 'sgRNA', 'color']
            values = corrs_source.data[key]
            filtered_corrs_source.data[key] = (values[i] for i in indices)

    second_scatter.outline_line_color = 'black'
    second_scatter.outline_line_alpha = 0.5

    right_line_source.data['x'][0] = index
    right_line_source.data['x'][1] = index

    right_line_source.data['y'][0] = corrs_source.data['y'][index] + 0.05

    right_line.visible = true

else
    color_source.data['second_color'] = color_source.data['blank']

    for key in ['x', 'y', 'sgRNA', 'color']
        filtered_corrs_source.data[key] = filtered_corrs_source.data[key].slice(0, 10)

    second_scatter.outline_line_color = null

    right_line.visible = false

color_source.change.emit()
corrs_source.change.emit()
filtered_corrs_source.change.emit()
right_line_source.change.emit()