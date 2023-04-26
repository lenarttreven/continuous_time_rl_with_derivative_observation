sbatch --gpus=1 --mem-per-cpu=5120 --wrap="python /cluster/home/trevenl/continuous_time_rl_with_derivative_observation/cucrl/utils/ensembles.py"

sbatch --gpus=1 --wrap="python /cluster/home/trevenl/continuous_time_rl_with_derivative_observation/run_experiments/test_gpu.py"
