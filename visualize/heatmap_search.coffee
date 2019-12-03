models = cb_obj.document._all_models_by_name._dict

data = models['quad_source'].data

guides = {guides}

query = cb_obj.value

if query != ''
    matches = (i for g, i in guides when g.indexOf(query) > -1)
    alphas = (0.3 for g in guides)
else
    matches = []
    alphas = (1 for g in guides)

lowers = ({lower_bound} for i in matches)
uppers = ({upper_bound} for i in matches)

data['left'] = [({lower_bound} + i for i in matches)..., lowers...]
data['right'] = [({lower_bound} + i + 1 for i in matches)..., uppers...]
data['bottom'] = [lowers..., ({upper_bound} - i - 1 for i in matches)...]
data['top'] = [uppers..., ({upper_bound} - i for i in matches)...]

alphas[i] = 1 for i in matches

models['label_source'].data['alpha'] = alphas

models['quad_source'].change.emit()
models['label_source'].change.emit()
console.log matches
console.log models['quad_source']
