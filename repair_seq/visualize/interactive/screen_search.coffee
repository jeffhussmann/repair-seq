column_names = ['gene']

query = cb_obj.value

all_matches = []
if query != ''
    for column in column_names
        targets = scatter_source.data[column]
        matches = (i for t, i in targets when t.indexOf(query) > -1 and i not in all_matches)
        all_matches.push matches...

scatter_source.selected.indices = all_matches
scatter_source.change.emit()
