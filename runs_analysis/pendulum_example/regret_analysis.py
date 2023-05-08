import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_FIG = False
LEGEND_FONTIZE = 12

XLABLE_FONTIZE = 14
YLABLE_FONTIZE = 14

XTICKS_FONTIZE = 12
YTICKS_FONTIZE = 12

q = 0.2
data = pd.read_csv("pendulum_small_input_control.csv")
groups = data['group'].unique()

for group in groups:
    group_data = data[data['group'] == group]

    cur_regrets = []
    for name in group_data['name'].unique():
        name_data = group_data[group_data['name'] == name]
        cur_regrets.append(np.cumsum(name_data['Cost/True trajectory 0'] - 20.16).to_numpy())
    regrets = np.stack(cur_regrets, axis=0)

    regret_means = np.mean(regrets, axis=0)
    regret_lower_quantile = np.quantile(regrets, q, axis=0)
    regret_upper_quantile = np.quantile(regrets, 1 - q, axis=0)

    plt.plot(np.arange(len(regret_means)), regret_means, label=group.split(".", 1)[1])
    plt.fill_between(np.arange(len(regret_means)), regret_lower_quantile, regret_upper_quantile, alpha=0.3)

plt.ylabel("Regret", fontsize=YLABLE_FONTIZE)
plt.xlabel("Episode", fontsize=XLABLE_FONTIZE)
plt.xticks(fontsize=XTICKS_FONTIZE)
plt.yticks(fontsize=YTICKS_FONTIZE)

plt.legend(fontsize=LEGEND_FONTIZE)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("pendulum_regret.pdf")
plt.show()
