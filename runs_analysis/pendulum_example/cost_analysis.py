import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_FIG = True

LEGEND_FONTIZE = 18

XLABLE_FONTIZE = 18
YLABLE_FONTIZE = 18

XTICKS_FONTIZE = 18
YTICKS_FONTIZE = 18

SUBSTRACTION = 20.1

q = 0.2
data = pd.read_csv("pendulum_small_input_control.csv")
groups = data["group"].unique()

for group in groups:
    group_data = data[data["group"] == group]
    groupped_data = group_data.groupby(["episode"])[["Cost/True trajectory 0"]]
    costs_means = groupped_data.median().values.reshape(-1) - SUBSTRACTION
    costs_lower_quantile = (
        groupped_data.quantile(q).values.reshape(
            -1,
        )
        - SUBSTRACTION
    )
    costs_upper_quantile = (
        groupped_data.quantile(1 - q).values.reshape(
            -1,
        )
        - SUBSTRACTION
    )
    plt.plot(
        np.arange(len(costs_means)),
        costs_means,
        label=group.split(".", 1)[1].replace("_GREEDY", ""),
    )
    plt.fill_between(
        np.arange(len(costs_means)),
        costs_lower_quantile,
        costs_upper_quantile,
        alpha=0.3,
    )

plt.axhline(
    20.16 - SUBSTRACTION,
    xmin=0,
    xmax=20,
    color="black",
    linestyle="--",
    label="Best continuous",
    linewidth=4,
)
plt.axhline(
    20.593 - SUBSTRACTION,
    xmin=0,
    xmax=20,
    color="black",
    linestyle="dotted",
    label="Best discrete",
    linewidth=4,
)
plt.ylabel("Cost", fontsize=YLABLE_FONTIZE)
plt.xlabel("Episode", fontsize=XLABLE_FONTIZE)
plt.xticks(fontsize=XTICKS_FONTIZE)
plt.yticks(fontsize=YTICKS_FONTIZE)

plt.xlim(3)
plt.yscale("log")
plt.legend(fontsize=LEGEND_FONTIZE)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("pendulum_cost.pdf")
plt.show()
