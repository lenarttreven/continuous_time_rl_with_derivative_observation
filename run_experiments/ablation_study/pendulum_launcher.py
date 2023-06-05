import pendulum_exp
from cucrl.utils.representatives import BatchStrategy, ExplorationStrategy
from run_experiments.util import generate_base_command, generate_run_commands

PROJECT_NAME = 'Ablation_study_pendulum_very_small_input_control'

applicable_configs = {
    'ExplorationStrategy': [ExplorationStrategy.OPTIMISTIC_ETA_TIME, ExplorationStrategy.MEAN],
    'MSS': [BatchStrategy.MAX_KERNEL_DISTANCE_GREEDY, BatchStrategy.MAX_DETERMINANT_GREEDY, BatchStrategy.EQUIDISTANT],
    'data_seed': [i for i in range(10)],
}


def main():
    command_list = []
    for exploration_strategy in applicable_configs['ExplorationStrategy']:
        for mss in applicable_configs['MSS']:
            for data_seed in applicable_configs['data_seed']:
                flags = {
                    'measurement_selection_strategy': mss.name,
                    'exploration_strategy': exploration_strategy.name,
                    'data_seed': data_seed,
                    'project_name': PROJECT_NAME
                }

                cmd = generate_base_command(pendulum_exp, flags=flags)
                command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=0, mode='euler', num_hours=8, promt=True, mem=16000)


if __name__ == '__main__':
    main()
