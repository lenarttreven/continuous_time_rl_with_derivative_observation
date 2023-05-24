import furuta_pendulum_exp
from cucrl.utils.representatives import BatchStrategy
from run_experiments.util import generate_base_command, generate_run_commands

PROJECT_NAME = "Furuta_pendulum_right_physics"

applicable_configs = {
    'MSS': [BatchStrategy.MAX_KERNEL_DISTANCE_GREEDY, BatchStrategy.MAX_DETERMINANT_GREEDY, BatchStrategy.EQUIDISTANT],
    'data_seed': [i for i in range(10)],
}


def main():
    command_list = []
    for mss in applicable_configs['MSS']:
        for data_seed in applicable_configs['data_seed']:
            flags = {
                'measurement_selection_strategy': mss.name,
                'data_seed': data_seed,
                'project_name': PROJECT_NAME
            }

            cmd = generate_base_command(furuta_pendulum_exp, flags=flags)
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', num_hours=24, promt=True, mem=16000)


if __name__ == '__main__':
    main()
