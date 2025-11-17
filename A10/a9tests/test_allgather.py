import torch

from src import collectives
from tests.utils import run_distributed


def _allgather_worker(rank: int, world_size: int) -> None:
    # Each rank contributes a vector [r, r+10]
    x = torch.tensor([rank, rank + 10], dtype=torch.float32)
    y = collectives.my_allgather(x)

    # y should have shape [world_size * 2]
    assert y.shape[0] == world_size * 2

    # Build expected 1-D tensor
    expected_list = []
    for r in range(world_size):
        expected_list.extend([float(r), float(r + 10)])
    expected = torch.tensor(expected_list, dtype=torch.float32)

    assert torch.allclose(y.view(-1), expected)


def test_allgather_basic() -> None:
    run_distributed(world_size=4, fn=_allgather_worker)

