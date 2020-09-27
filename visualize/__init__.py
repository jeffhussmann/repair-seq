import bokeh.palettes

category10 = bokeh.palettes.Category10[10]
accent = bokeh.palettes.Accent[8]
good_colors = (category10[2:7] + category10[8:] + accent[:3] + accent[4:])*100

colors_list = [[c]*10 for c in good_colors]