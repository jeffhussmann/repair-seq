models = cb_obj.document._all_models_by_name._dict
x_range = models['x_range']
y_range = models['y_range']

if x_range.tags[0] != 'locked'
    previous_size = x_range.tags[0]
    console.log previous_size

    x_range.tags[0] = 'locked'
    
    x_size = x_range.end - x_range.start
    y_size = y_range.end - y_range.start

    x_change = Math.abs(x_size - previous_size)
    y_change = Math.abs(y_size - previous_size)

    if x_change > y_change
        new_size = x_size
        console.log 'x', new_size
    else
        new_size = y_size
        console.log 'y', new_size

    max_size = {upper_bound} - {lower_bound}
    new_size = Math.min(new_size, max_size)

    if new_size == max_size
        mid = x_range.start + x_range.end
        x_mid = mid
        y_mid = mid
    else
        x_mid = (x_range.start + x_range.end) / 2
        y_mid = (y_range.start + y_range.end) / 2

        min_mid = {lower_bound} + (new_size / 2) 
        max_mid = {upper_bound} - (new_size / 2) 

        if x_mid < min_mid
            x_mid = min_mid
        if x_mid > max_mid
            x_mid = max_mid
        
        if y_mid < min_mid
            y_mid = min_mid
        if y_mid > max_mid
            y_mid = max_mid

    x_range.start = x_mid - (new_size / 2)
    x_range.end = x_mid + (new_size / 2)

    y_range.start = y_mid - (new_size / 2)
    y_range.end = y_mid + (new_size / 2)

    x_range.tags[0] = new_size
