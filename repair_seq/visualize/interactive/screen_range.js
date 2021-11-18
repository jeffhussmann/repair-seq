var models;

models = cb_obj.document._all_models_by_name._dict;

if (cb_obj.start < {lower_bound}) {
  cb_obj.start = {lower_bound};
}

if (cb_obj.end > {upper_bound}) {
  cb_obj.end = {upper_bound};
}
