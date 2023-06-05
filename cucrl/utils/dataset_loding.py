from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


@jax.jit
def jax_collate(batch):
    return jtu.tree_map(lambda *x: jnp.stack(x), *batch)


class DataRepr(NamedTuple):
    xs: jnp.ndarray
    ys: jnp.ndarray


class DataTrain(Dataset):
    def __init__(self, input_data, target_data):
        self.my_data = DataRepr(xs=input_data, ys=target_data)

    def __len__(self):
        return len(self.my_data.xs)

    def __getitem__(self, index):
        return DataRepr(
            **{
                dict_key: value[index]
                for dict_key, value in self.my_data._asdict().items()
            }
        )


# Example usage
input_data = jnp.array([[i for _ in range(1)] for i in range(10)], dtype=jnp.float32)
target_data = jnp.array([[i for _ in range(1)] for i in range(10)], dtype=jnp.float32)

dataset = DataTrain(input_data, target_data)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=jax_collate)

for batch in dataloader:
    print("Batch: ")
    print(batch.xs)

print("New cycle")
for batch in dataloader:
    print("Batch: ")
    print(batch.xs)
