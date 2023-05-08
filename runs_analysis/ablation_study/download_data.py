import pandas as pd
import wandb


def get_dataset(run_name):
    api = wandb.Api()
    runs = api.runs(run_name)
    dataset = pd.DataFrame()
    keys = ['Cost/True trajectory 0']
    for run in runs:
        frames = []
        for key in keys:
            frames.append(run.history(keys=[key], x_axis="episode"))
        frames = [df.set_index('episode') for df in frames]
        frames = pd.concat(frames, axis=0)
        frames['group'] = run.group
        frames['name'] = run.name
        dataset = pd.concat([dataset, frames])

    return dataset


out = get_dataset("trevenl/Ablation_study_pendulum_very_small_input_control")
out.to_csv("Ablation_study_pendulum.csv")
