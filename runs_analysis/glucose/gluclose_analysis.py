import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_FIG = True

LEGEND_FONTIZE = 18

XLABLE_FONTIZE = 18
YLABLE_FONTIZE = 18

XTICKS_FONTIZE = 18
YTICKS_FONTIZE = 18

SUBSTRACTION = 0

q = 0.2
data = pd.read_csv('Glucose_final_exp.csv')
groups = data['group'].unique()

for group in groups:
    group_data = data[data['group'] == group]
    groupped_data = group_data.groupby(['episode'])[['Cost/True trajectory 0']]
    costs_means = groupped_data.mean().values.reshape(-1) - SUBSTRACTION
    costs_std = groupped_data.std().values.reshape(-1, ) - SUBSTRACTION
    print('Group:', group)
    print('Means:', 10 * costs_means[-1])
    print('Std:', 10 * costs_std[-1])
