import torch

from src import collectives
from tests.utils import run_distributed


def _allreduce_worker(rank: int, world_size: int) -> None:
    # Each rank contributes rank + 1
    x = torch.tensor([rank + 1.0], dtype=torch.float32)
    y = collectives.my_allreduce(x)

    # Sum of 1..world_size
    expected = world_size * (world_size + 1) / 2.0
    assert torch.allclose(y, torch.tensor([expected], dtype=torch.float32))


def test_allreduce_basic_world_size_4() -> None:
    run_distributed(world_size=4, fn=_allreduce_worker)


def test_allreduce_world_size_3() -> None:
    run_distributed(world_size=3, fn=_allreduce_worker)

