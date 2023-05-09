from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from termcolor import colored

from cucrl.utils.classes import PlotData, PlotOpenLoop, MeasurementSelection
from cucrl.utils.helper_functions import norm_difference
from cucrl.utils.representatives import Space
from cucrl.utils.representatives import Statistics


class Plotter:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @staticmethod
    def _add_traj_opt_prediction(ax, nodes_times: jnp.array, nodes_values: jnp.array, visualization_times: jnp.ndarray,
                                 node_predictions: jnp.array):
        ax.scatter(nodes_times, nodes_values, color='red', label='Prediction from NLP solver')
        ax.plot(visualization_times, node_predictions, color='green', label='Spline interpolation')

    def plot_open_loop_learning(self, data: PlotOpenLoop):
        num_traj = len(data.times)
        # Plot control learning
        figure_control_learinng, ax_trajectory_optimization = plt.subplots(
            num_traj, self.state_dim + self.action_dim,
            figsize=(5 * (self.state_dim + self.action_dim) + 4, 4 * num_traj))
        ax_trajectory_optimization = np.array(ax_trajectory_optimization)
        ax_trajectory_optimization = ax_trajectory_optimization.reshape(num_traj, self.state_dim + self.action_dim)
        print(colored('Plotting control learning', 'green'))

        for trajectory in range(num_traj):
            print('Starting with trajectory {}'.format(trajectory))
            print('State dimension: ', end=' ')
            for dimension in range(self.state_dim):
                print(dimension, end=' ')
                ax = ax_trajectory_optimization[trajectory, dimension]
                ax.grid()
                self._add_traj_opt_prediction(ax=ax, nodes_times=data.times[trajectory].reshape(-1),
                                              nodes_values=data.states[trajectory][:, dimension],
                                              visualization_times=data.visualization_times[trajectory].reshape(-1),
                                              node_predictions=data.state_prediction[trajectory][:, dimension])
                ax.set_xlabel(r"t")
                ax.set_ylabel(r"$x_{}(t, x_0)$".format('{' + str(dimension) + '}'))
                if trajectory == 0 and dimension == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
            print('\nAction dimension: ', end=' ')
            for dimension in range(self.action_dim):
                print(dimension, end=' ')
                ax = ax_trajectory_optimization[trajectory, self.state_dim + dimension]
                ax.grid()
                self._add_traj_opt_prediction(ax=ax, nodes_times=data.times[trajectory].reshape(-1),
                                              nodes_values=data.controls[trajectory][:, dimension],
                                              visualization_times=data.visualization_times[trajectory].reshape(-1),
                                              node_predictions=data.controls_prediction[trajectory][:, dimension])
                ax.set_xlabel(r"t")
                ax.set_ylabel(r"$u_{}(t, x_0)$".format('{' + str(dimension) + '}'))
                if trajectory == 0 and dimension == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
        print()
        # Add description for which dimension and trajectory we are on
        cols = ["State dimension {}".format(col) for col in range(self.state_dim)] + ["Control dimension {}".format(col)
                                                                                      for col in range(self.state_dim)]
        rows = ["Trajectory {}".format(row) for row in range(num_traj)]
        pad = 5
        if self.state_dim + self.action_dim >= 2:
            for ax, col in zip(ax_trajectory_optimization[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords="axes fraction", textcoords="offset points",
                            size="large", ha="center", va="baseline", )
        if num_traj >= 2:
            for ax, row in zip(ax_trajectory_optimization[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                            textcoords="offset points", size="large", ha="right", va="center", )
        return figure_control_learinng

    def plot(self, data: PlotData):
        num_traj = len(data.visualization_times)
        # Plot smoother states
        figure_smoother_states, ax_states = plt.subplots(num_traj, self.state_dim + self.action_dim, figsize=(
            5 * (self.state_dim + self.action_dim) + 4, 4 * num_traj))
        ax_states = ax_states.reshape(num_traj, self.state_dim + self.action_dim)
        print(colored('Plot smoother states', 'green'))
        self._create_plot(axs=ax_states, plot_times=data.visualization_times,
                          plot_values=None, plot_values_lower=None,
                          plot_values_upper=None, q=95, statistics_type=Statistics.MEAN,
                          observation_times=None, observations=None,
                          gt_times=data.visualization_times, gt_values=data.gt_states_vis, space=Space.STATE,
                          prediction_states=data.prediction_states, actual_actions=data.actual_actions,
                          predicted_actions=data.predicted_actions)

        # Plot dynamics derivatives
        figure_dynamics_der, ax_dynamics = plt.subplots(num_traj, self.state_dim,
                                                        figsize=(5 * self.state_dim + 4, 4 * num_traj))
        ax_dynamics = np.array(ax_dynamics)
        ax_dynamics = ax_dynamics.reshape(num_traj, self.state_dim)

        dynamics_stds = jnp.sqrt(jnp.clip(data.dynamics_der_vars, a_min=0))
        dynamics_values_lower = data.dynamics_der_means - 1.960 * dynamics_stds
        dynamics_values_upper = data.dynamics_der_means + 1.960 * dynamics_stds
        print(colored('Plot dynamics derivatives', 'green'))
        self._create_plot(axs=ax_dynamics, plot_times=data.visualization_times,
                          plot_values=data.dynamics_der_means, plot_values_lower=dynamics_values_lower,
                          plot_values_upper=dynamics_values_upper, q=95, statistics_type=Statistics.MEAN,
                          observation_times=data.observation_times, observations=data.observations,
                          gt_times=data.visualization_times, gt_values=data.gt_der_vis, space=Space.DERIVATIVE)
        return figure_dynamics_der, figure_smoother_states

    def _create_plot(self, axs, plot_times: jax.Array | None, plot_values: jax.Array | None,
                     plot_values_lower: jax.Array | None, plot_values_upper: jax.Array | None, q: float,
                     statistics_type: Statistics, observation_times: Optional[List[jnp.array]],
                     observations: Optional[List[jnp.array]], gt_times: jax.Array, gt_values: jax.Array,
                     space: Space = Space.STATE, prediction_states: jax.Array | None = None,
                     actual_actions: jax.Array | None = None, predicted_actions: jax.Array | None = None):

        num_trajectories = len(plot_times)

        for trajectory in range(num_trajectories):
            print('Starting with trajectory {}'.format(trajectory))
            print('State dimension: ', end=' ')
            for dimension in range(self.state_dim):
                print(dimension, end=' ')
                ax = axs[trajectory, dimension]
                ax.grid()
                obs_times = None if observation_times is None else observation_times[trajectory]
                obs = None if observations is None else observations[trajectory][:, dimension]
                pred = None if prediction_states is None else prediction_states[trajectory][:, dimension]
                cur_plot_times = None if plot_times is None else plot_times[trajectory]
                cur_plot_values = None if plot_values is None else plot_values[trajectory][:, dimension]
                cur_plot_values_lower = None if plot_values_lower is None else plot_values_lower[trajectory][:,
                                                                               dimension]
                cur_plot_values_upper = None if plot_values_upper is None else plot_values_upper[trajectory][:,
                                                                               dimension]
                self._add_prediction_on_plot(ax=ax, plot_times=cur_plot_times, plot_values=cur_plot_values,
                                             plot_values_lower=cur_plot_values_lower,
                                             plot_values_upper=cur_plot_values_upper,
                                             q=q, statistics_type=statistics_type, observation_times=obs_times,
                                             observations=obs, prediction_states=pred)
                ax.set_xlabel(r"t")
                ax.set_ylabel(r"$x_{}(t)$".format(
                    '{' + str(dimension) + '}') if space == Space.STATE else r"$\dot x_{}(t)$".format(
                    '{' + str(dimension) + '}'))
                ax.plot(gt_times[trajectory], gt_values[trajectory][:, dimension], label=r"True $x_{}$".format(
                    '{' + str(dimension) + '}') if space == Space.STATE else r"True $\dot x_{}$".format(
                    '{' + str(dimension) + '}'), color="black", )
                if trajectory == 0 and dimension == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
            if actual_actions is not None:
                print('\nControl dimension: ', end=' ')
                for dimension in range(self.action_dim):
                    print(dimension, end=' ')
                    ax = axs[trajectory, self.state_dim + dimension]
                    ax.grid()
                    pred_actions = None if predicted_actions is None else predicted_actions[trajectory][:, dimension]
                    self._add_prediction_on_plot(ax=ax, plot_times=plot_times[trajectory],
                                                 plot_values=None, plot_values_lower=None, plot_values_upper=None,
                                                 q=q, statistics_type=statistics_type, observation_times=None,
                                                 observations=None, prediction_states=None,
                                                 actual_actions=actual_actions[trajectory][:, dimension],
                                                 predicted_actions=pred_actions)
                    ax.set_xlabel(r"t")
                    ax.set_ylabel(r"$u_{}(t)$".format('{' + str(dimension) + '}'))
                    if trajectory == 0 and dimension == 0:
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys())
        # Add description for which dimension and trajectory we are on
        cols = ["State dimension {}".format(col) for col in range(self.state_dim)] + ["Control dimension {}".format(col)
                                                                                      for col in range(self.state_dim)]
        rows = ["Trajectory {}".format(row) for row in range(num_trajectories)]
        pad = 5
        if self.state_dim >= 2 or (actual_actions is not None and (self.state_dim + self.action_dim >= 2)):
            for ax, col in zip(axs[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords="axes fraction", textcoords="offset points",
                            size="large", ha="center", va="baseline", )
        if num_trajectories >= 2:
            for ax, row in zip(axs[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                            textcoords="offset points", size="large", ha="right", va="center", )

    @staticmethod
    def _add_prediction_on_plot(ax, plot_times: jax.Array, plot_values: jax.Array | None,
                                plot_values_lower: jax.Array | None, plot_values_upper: jax.Array | None, q: float,
                                statistics_type: Statistics, observation_times: jax.Array | None,
                                observations: jax.Array | None, prediction_states: jax.Array | None,
                                actual_actions: jax.Array | None = None, predicted_actions: jax.Array | None = None):
        if actual_actions is not None:
            ax.plot(plot_times, actual_actions, color='blue', label='Actual control')
        if predicted_actions is not None:
            ax.plot(plot_times, predicted_actions, color='green', label='Predicted control')
        if observation_times is not None:
            ax.plot(observation_times, observations, "r.", markersize=10, label="Observations")
        if prediction_states is not None:
            ax.plot(plot_times, prediction_states, color='green', label='Hallucination')
        label = ''
        if statistics_type == Statistics.MEDIAN:
            label = "Between {:.2f} and {:.2f} quantile".format(1 - q, q)
        elif statistics_type == Statistics.MEAN:
            label = "{}% confidence interval".format(q)
        if plot_values is not None:
            ax.plot(plot_times, plot_values, "b-", label="Prediction")
            ax.fill_between(plot_times.reshape(-1), plot_values_lower, plot_values_upper, alpha=0.5, fc="b", ec="None",
                            label=label)

    def plot_measurement_selection(self, measurement_selection: MeasurementSelection):
        # measurement_selection elements have the following shape:
        # (num_traj, num_hallucinations, num_measurements, element_dim)
        assert measurement_selection.potential_xs.ndim == 4
        num_traj = measurement_selection.potential_xs.shape[0]
        num_hallucinations = measurement_selection.potential_xs.shape[1]
        fig, axs = plt.subplots(num_traj, num_hallucinations, figsize=(5 * num_hallucinations + 4, 4 * num_traj))
        axs = np.array(axs)
        axs = axs.reshape(num_traj, num_hallucinations)
        for trajectory in range(num_traj):
            for hallucination in range(num_hallucinations):
                ax = axs[trajectory, hallucination]
                cur_measurement_selection = jtu.tree_map(lambda x: x[trajectory, hallucination], measurement_selection)
                self._add_measurement_selection_on_plot(ax, cur_measurement_selection)
                ax.set_xlabel(r't')
                ax.set_ylabel("Uncertainty in variance")
                if trajectory == 0 and hallucination == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
        # Add description for which dimension and trajectory we are on
        cols = ["Hallucination {}".format(col) for col in range(num_hallucinations)]
        rows = ["Trajectory {}".format(row) for row in range(num_traj)]
        pad = 5
        if num_hallucinations >= 1:
            for ax, col in zip(axs[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords="axes fraction", textcoords="offset points",
                            size="large", ha="center", va="baseline", )
        if num_traj >= 2:
            for ax, row in zip(axs[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                            textcoords="offset points", size="large", ha="right", va="center", )
        ############################################################################
        ############################################################################
        ############################################################################
        fig_space, axs_space = plt.subplots(num_traj, num_hallucinations,
                                            figsize=(5 * num_hallucinations + 4, 4 * num_traj))
        axs_space = np.array(axs_space)
        axs_space = axs_space.reshape(num_traj, num_hallucinations)
        for trajectory in range(num_traj):
            for hallucination in range(num_hallucinations):
                ax = axs_space[trajectory, hallucination]
                cur_measurement_selection = jtu.tree_map(lambda x: x[trajectory, hallucination], measurement_selection)
                self._add_measurement_selection_on_plot_space(ax, cur_measurement_selection)
                ax.set_xlabel(r'$\Delta(x, u)$')
                ax.set_ylabel("Uncertainty in variance")
                if trajectory == 0 and hallucination == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
        # Add description for which dimension and trajectory we are on
        cols = ["Hallucination {}".format(col) for col in range(num_hallucinations)]
        rows = ["Trajectory {}".format(row) for row in range(num_traj)]
        pad = 5
        if num_hallucinations >= 1:
            for ax, col in zip(axs[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords="axes fraction", textcoords="offset points",
                            size="large", ha="center", va="baseline", )
        if num_traj >= 2:
            for ax, row in zip(axs[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                            textcoords="offset points", size="large", ha="right", va="center", )
        ############################################################################
        ############################################################################
        ############################################################################
        if self.state_dim == 2:
            fig_phase, axs_phase = plt.subplots(num_traj, num_hallucinations,
                                                figsize=(5 * num_hallucinations + 4, 4 * num_traj))
            axs_phase = np.array(axs_phase)
            axs_phase = axs_phase.reshape(num_traj, num_hallucinations)
            for trajectory in range(num_traj):
                for hallucination in range(num_hallucinations):
                    ax = axs_phase[trajectory, hallucination]
                    cur_measurement_selection = jtu.tree_map(lambda x: x[trajectory, hallucination],
                                                             measurement_selection)
                    fig_phase = self._add_measurement_selection_on_plot_phase(fig_phase, ax, cur_measurement_selection)
                    ax.set_xlabel(r'$x_0$')
                    ax.set_ylabel(r'$x_1$')
                    if trajectory == 0 and hallucination == 0:
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys())
                # Add description for which dimension and trajectory we are on
            cols = ["Hallucination {}".format(col) for col in range(num_hallucinations)]
            rows = ["Trajectory {}".format(row) for row in range(num_traj)]
            pad = 5
            if num_hallucinations >= 1:
                for ax, col in zip(axs[0], cols):
                    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords="axes fraction",
                                textcoords="offset points",
                                size="large", ha="center", va="baseline", )
            if num_traj >= 2:
                for ax, row in zip(axs[:, 0], rows):
                    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                                textcoords="offset points", size="large", ha="right", va="center", )
            return fig, fig_space, fig_phase
        else:
            return fig, fig_space, None

    def _add_measurement_selection_on_plot_phase(self, fig, ax, measurement_selection: MeasurementSelection):
        # measurement_selection elements have the following shape:
        # (num_measurements, element_dim)
        assert measurement_selection.potential_xs.ndim == 2
        ax.grid()

        fig, ax = multicolored_lines(fig, ax,
                                     measurement_selection.potential_xs[:, 0],
                                     measurement_selection.potential_xs[:, 1],
                                     jnp.sum(measurement_selection.vars_before_collection, axis=1),
                                     )

        number_of_equidistant_measurements = measurement_selection.proposed_ts.shape[0]
        repeat_num = int(measurement_selection.potential_xs.shape[0] / number_of_equidistant_measurements)

        ax.scatter(measurement_selection.potential_xs[:, 0][::repeat_num],
                   measurement_selection.potential_xs[:, 1][::repeat_num],
                   color='blue', marker='+', s=100, linewidth=2, label='Equidistant measurements')

        proposed_indices = measurement_selection.proposed_indices
        ax.scatter(measurement_selection.potential_xs[proposed_indices, 0],
                   measurement_selection.potential_xs[proposed_indices, 1],
                   color='red', marker='x', s=100, linewidth=2, label='Proposed measurements')

        number_of_arrows = 20
        repeat_num_arrows = int(measurement_selection.potential_xs.shape[0] / number_of_arrows)

        dx_0 = jnp.diff(measurement_selection.potential_xs[:, 0])
        dx_1 = jnp.diff(measurement_selection.potential_xs[:, 1])

        ax.quiver(measurement_selection.potential_xs[:-1, 0][::repeat_num_arrows],
                  measurement_selection.potential_xs[:-1, 1][::repeat_num_arrows],
                  dx_0[::repeat_num_arrows], dx_1[::repeat_num_arrows],
                  angles='xy', scale=2.0, color='black', alpha=0.3, linewidth=0)
        return fig

    def _add_measurement_selection_on_plot(self, ax, measurement_selection: MeasurementSelection):
        # measurement_selection elements have the following shape:
        # (num_measurements, element_dim)
        assert measurement_selection.potential_xs.ndim == 2
        ax.grid()
        # Plot the selected times
        ax.plot(measurement_selection.potential_ts.reshape(-1),
                jnp.zeros_like(measurement_selection.potential_ts.reshape(-1)),
                color='blue', alpha=0.3)
        ax.scatter(measurement_selection.proposed_ts.reshape(-1),
                   jnp.zeros_like(measurement_selection.proposed_ts.reshape(-1)),
                   color='red', marker='x', label='Proposed measurements')
        # Plot the variance based on which we selected times
        color = cm.rainbow(np.linspace(0, 1, self.state_dim))
        for i, c in zip(range(self.state_dim), color):
            ax.plot(measurement_selection.potential_ts.reshape(-1),
                    measurement_selection.vars_before_collection[:, i],
                    c=c, label=r'$\sigma_{}^2(x(t))$'.format(i))
        ax.plot(measurement_selection.potential_ts.reshape(-1),
                jnp.sum(measurement_selection.vars_before_collection, axis=1),
                color='black', label=r'$\sum_{i=0} \sigma_i^2(x(t))$')

    def _add_measurement_selection_on_plot_space(self, ax, measurement_selection: MeasurementSelection):
        # measurement_selection elements have the following shape:
        # (num_measurements, element_dim)
        assert measurement_selection.potential_xs.ndim == 2
        ax.grid()
        # Plot the selected times
        z = jnp.concatenate((measurement_selection.potential_xs, measurement_selection.potential_us), axis=1)
        input_differences = norm_difference(z)
        assert input_differences.shape == (measurement_selection.potential_xs.shape[0],)
        ax.plot(input_differences, jnp.zeros_like(input_differences), color='blue', alpha=0.3)
        ax.scatter(input_differences[measurement_selection.proposed_indices],
                   jnp.zeros_like(input_differences[measurement_selection.proposed_indices]),
                   color='red', marker='x', label='Proposed measurements')
        # Plot the variance based on which we selected times
        color = cm.rainbow(np.linspace(0, 1, self.state_dim))
        for i, c in zip(range(self.state_dim), color):
            ax.plot(input_differences, measurement_selection.vars_before_collection[:, i],
                    c=c, label=r'$\sigma_{}^2(x(t))$'.format(i))
        ax.plot(input_differences, jnp.sum(measurement_selection.vars_before_collection, axis=1),
                color='black', label=r'$\sum_{i=0} \sigma_i^2(x(t))$')


def multicolored_lines(fig, ax, x, y, z):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    lc = colorline(x, y, z=z, cmap='Greens')
    fig.colorbar(lc, ax=ax, label=r'$\sum_{i}\sigma_i^2(x(t), u(t))$')
    return fig, ax


def colorline(x, y, z=None, cmap='autumn', linewidth=2, alpha=1.0):
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
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha, label='Trajectory')

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
