import pickle

import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm

from cucrl.utils.classes import MeasurementSelection

SAVE_FIG = True
LEGEND_FONTIZE = 18
TITLE_FONTIZE = 20

XLABLE_FONTIZE = 20
YLABLE_FONTIZE = 20

XTICKS_FONTIZE = 20
YTICKS_FONTIZE = 20

state_dim = 2

SUBSTRACTION = 20.1

q = 0.2
data = pd.read_csv("pendulum_small_input_control.csv")
groups = data["group"].unique()


def multicolored_lines(fig, ax, x, y, z):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    lc = colorline(x, y, z=z, cmap="viridis_r")

    def fmt(x, pos):
        a, b = "{:.2e}".format(x).split("e")
        b = int(b)
        return r"${} \times 10^{{{}}}$".format(a, b)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))
    formatter.set_scientific(True)
    # label = r'$\sum_{i}\sigma_i^2(x(t), u(t))$'
    cbar = fig.colorbar(lc, ax=ax, format=formatter)
    cbar.set_label(label="Variance", size=YLABLE_FONTIZE)
    cbar.ax.tick_params(labelsize=XTICKS_FONTIZE)
    cbar.ax.yaxis.get_offset_text().set(size=XTICKS_FONTIZE)

    return fig, ax


def colorline(x, y, z=None, cmap="autumn", linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    # lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha, label='Trajectory')
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha
    )
    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def add_measurement_selection_on_plot_phase(
    fig, ax, measurement_selection: MeasurementSelection
):
    # measurement_selection elements have the following shape:
    # (num_measurements, element_dim)
    assert measurement_selection.potential_xs.ndim == 2
    ax.grid()

    fig, ax = multicolored_lines(
        fig,
        ax,
        measurement_selection.potential_xs[:, 0],
        measurement_selection.potential_xs[:, 1],
        jnp.sum(measurement_selection.vars_before_collection, axis=1),
    )

    number_of_equidistant_measurements = measurement_selection.proposed_ts.shape[0]
    repeat_num = int(
        measurement_selection.potential_xs.shape[0] / number_of_equidistant_measurements
    )

    ax.scatter(
        measurement_selection.potential_xs[:, 0][::repeat_num],
        measurement_selection.potential_xs[:, 1][::repeat_num],
        color="blue",
        marker="+",
        s=300,
        linewidth=2,
        label="Equidistant MSS",
    )

    proposed_indices = measurement_selection.proposed_indices
    ax.scatter(
        measurement_selection.potential_xs[proposed_indices, 0],
        measurement_selection.potential_xs[proposed_indices, 1],
        color="green",
        marker="x",
        s=300,
        linewidth=2,
        label="Adaptive MSS",
    )

    number_of_arrows = 20
    repeat_num_arrows = int(
        measurement_selection.potential_xs.shape[0] / number_of_arrows
    )

    dx_0 = jnp.diff(measurement_selection.potential_xs[:, 0])
    dx_1 = jnp.diff(measurement_selection.potential_xs[:, 1])

    ax.quiver(
        measurement_selection.potential_xs[:-1, 0][::repeat_num_arrows],
        measurement_selection.potential_xs[:-1, 1][::repeat_num_arrows],
        dx_0[::repeat_num_arrows],
        dx_1[::repeat_num_arrows],
        angles="xy",
        scale=2.0,
        color="black",
        alpha=0.3,
        linewidth=0,
    )
    ax.tick_params(axis="x", labelsize=XTICKS_FONTIZE)
    ax.tick_params(axis="y", labelsize=YTICKS_FONTIZE)
    ax.legend(fontsize=LEGEND_FONTIZE, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    return fig


def add_measurement_selection_on_plot(ax, measurement_selection: MeasurementSelection):
    # measurement_selection elements have the following shape:
    # (num_measurements, element_dim)
    assert measurement_selection.potential_xs.ndim == 2
    ax.grid()
    # Plot the selected times
    ax.plot(
        measurement_selection.potential_ts.reshape(-1),
        jnp.zeros_like(measurement_selection.potential_ts.reshape(-1)),
        color="blue",
        alpha=0.3,
    )
    ax.scatter(
        measurement_selection.proposed_ts.reshape(-1),
        jnp.zeros_like(measurement_selection.proposed_ts.reshape(-1)),
        color="green",
        marker="x",
        s=300,
        linewidth=3,
    )
    ax.scatter(
        np.linspace(0, 10, 10),
        jnp.zeros_like(measurement_selection.proposed_ts.reshape(-1)),
        color="Blue",
        marker="+",
        s=300,
        linewidth=3,
    )
    # Plot the variance based on which we selected times
    color = cm.rainbow(np.linspace(0, 1, state_dim))
    for i, c in zip(range(state_dim), color):
        ax.plot(
            measurement_selection.potential_ts.reshape(-1),
            measurement_selection.vars_before_collection[:, i],
            c=c,
            label=r"$\sigma_{}^2(x(t))$".format(i),
        )
    ax.plot(
        measurement_selection.potential_ts.reshape(-1),
        jnp.sum(measurement_selection.vars_before_collection, axis=1),
        color="black",
        label=r"$\sum_{i=0} \sigma_i^2(x(t))$",
    )
    ax.legend(fontsize=LEGEND_FONTIZE)
    ax.set_xlabel("Time", fontsize=XLABLE_FONTIZE)
    ax.set_ylabel("Variance", fontsize=YLABLE_FONTIZE)
    ax.tick_params(axis="x", labelsize=XTICKS_FONTIZE)
    ax.tick_params(axis="y", labelsize=YTICKS_FONTIZE)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))
    formatter.set_scientific(True)

    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(YTICKS_FONTIZE)


def add_first_plot(ax):
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
        ax.plot(
            np.arange(len(costs_means)),
            costs_means,
            label=group.split(".", 1)[1].replace("_GREEDY", ""),
        )
        ax.fill_between(
            np.arange(len(costs_means)),
            costs_lower_quantile,
            costs_upper_quantile,
            alpha=0.3,
        )

    ax.axhline(
        20.16 - SUBSTRACTION,
        xmin=0,
        xmax=20,
        color="black",
        linestyle="--",
        label="Best continuous",
        linewidth=4,
    )
    ax.axhline(
        20.593 - SUBSTRACTION,
        xmin=0,
        xmax=20,
        color="black",
        linestyle="dotted",
        label="Best discrete",
        linewidth=4,
    )
    ax.set_ylabel("Cost", fontsize=YLABLE_FONTIZE)
    ax.set_xlabel("Episode", fontsize=XLABLE_FONTIZE)

    ax.tick_params(axis="x", labelsize=XTICKS_FONTIZE)
    ax.tick_params(axis="y", labelsize=YTICKS_FONTIZE)

    ax.set_xlim(3)
    ax.set_yscale("log")
    ax.legend(fontsize=LEGEND_FONTIZE)


with open("measurement_selection_19.pkl", "rb") as f:
    mss_data = pickle.load(f)

mss = jtu.tree_map(lambda x: x[0, 0], mss_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axs = np.array(axes).reshape(1, 3)

# Plot the phase plot
ax = axs[0, 0]
add_first_plot(ax)

# # Plot comparison of different MSS
add_measurement_selection_on_plot(axs[0, 1], mss)

# Plot the phase plot
ax = axs[0, 2]
add_measurement_selection_on_plot_phase(fig, ax, mss)
ax.set_xlabel(r"$x_1$", fontsize=XLABLE_FONTIZE)
ax.set_ylabel(r"$x_2$", fontsize=YLABLE_FONTIZE)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig("measurement_selection_strategies.pdf")
plt.show()
