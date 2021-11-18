models = cb_obj.document._all_models_by_name._dict

models['confidence_intervals'].visible = cb_obj.active
models['confidence_intervals'].change.emit()
