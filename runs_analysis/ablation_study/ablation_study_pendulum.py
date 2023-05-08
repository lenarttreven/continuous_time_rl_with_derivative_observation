import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_FIG = True

LEGEND_FONTIZE = 16
TITLE_FONTIZE = 20

XLABLE_FONTIZE = 20
YLABLE_FONTIZE = 20

XTICKS_FONTIZE = 20
YTICKS_FONTIZE = 20

q = 0.2
data = pd.read_csv("Ablation_study_pendulum.csv")
groups = data['group'].unique()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axs = np.array(axes).reshape(1, 3)

MSSs = ['EQUIDISTANT', 'MAX_KERNEL_DISTANCE', 'MAX_DETERMINANT']
EXPLORATION = ['MEAN', 'OPTIMISTIC']

SUBSRTACTION = 20.1

for index, mss in enumerate(MSSs):
    ax = axs[0, index]
    data_mss = data[data['group'].str.contains(mss)]
    data_mean = data_mss[data_mss['group'].str.contains('MEAN')]
    data_optimistic = data_mss[data_mss['group'].str.contains('OPTIMISTIC')]

    groupped_mean = data_mean.groupby(['episode'])[["Cost/True trajectory 0"]]
    groupped_optimistic = data_optimistic.groupby(['episode'])[["Cost/True trajectory 0"]]

    MEAN_costs_means = groupped_mean.median().values.reshape(-1) - SUBSRTACTION
    MEAN_costs_lower_quantile = groupped_mean.quantile(q).values.reshape(-1, ) - SUBSRTACTION
    MEAN_costs_upper_quantile = groupped_mean.quantile(1 - q).values.reshape(-1, ) - SUBSRTACTION

    ax.plot(np.arange(len(MEAN_costs_means)), MEAN_costs_means, label="GREEDY")
    ax.fill_between(np.arange(len(MEAN_costs_means)), MEAN_costs_lower_quantile, MEAN_costs_upper_quantile, alpha=0.3)

    OPTIMISTIC_costs_means = groupped_optimistic.median().values.reshape(-1) - SUBSRTACTION
    OPTIMISTIC_costs_lower_quantile = groupped_optimistic.quantile(q).values.reshape(-1, ) - SUBSRTACTION
    OPTIMISTIC_costs_upper_quantile = groupped_optimistic.quantile(1 - q).values.reshape(-1, ) - SUBSRTACTION

    ax.plot(np.arange(len(OPTIMISTIC_costs_means)), OPTIMISTIC_costs_means, label="OPTIMISM")
    ax.fill_between(np.arange(len(OPTIMISTIC_costs_means)), OPTIMISTIC_costs_lower_quantile,
                    OPTIMISTIC_costs_upper_quantile, alpha=0.3)

    ax.axhline(20.17 - SUBSRTACTION, xmin=0, xmax=20, color="black", linestyle="--", label='Best continuous',
               linewidth=4)
    ax.axhline(20.656 - SUBSRTACTION, xmin=0, xmax=20, color="black", linestyle='dotted',
               label='Best discrete', linewidth=4)

    ax.set_title(mss, fontsize=TITLE_FONTIZE)
    ax.tick_params(axis='x', labelsize=XTICKS_FONTIZE)
    ax.tick_params(axis='y', labelsize=YTICKS_FONTIZE)
    ax.set_xlabel('Episode', fontsize=XLABLE_FONTIZE)

    ax.set_yscale('log')
    if index == 0:
        ax.set_ylabel('Cost', fontsize=YLABLE_FONTIZE)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=LEGEND_FONTIZE)

# fig.text(0.03, 0.85, '20+', fontsize=YTICKS_FONTIZE, ha='center', va='center')
plt.tight_layout()
if SAVE_FIG:
    plt.savefig('ablation_study_pendulum.pdf')
plt.show()
