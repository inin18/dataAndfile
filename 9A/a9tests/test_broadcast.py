import torch

from src import collectives
from tests.utils import run_distributed


def _broadcast_worker(rank: int, world_size: int) -> None:
    src = 0
    if rank == src:
        x = torch.tensor([42.0], dtype=torch.float32)
    else:
        x = torch.tensor([-1.0], dtype=torch.float32)

    y = collectives.my_broadcast(x, src=src)

    # After broadcast, all ranks must see the same value
    assert torch.allclose(y, torch.tensor([42.0], dtype=torch.float32))


def test_broadcast_basic() -> None:
    run_distributed(world_size=4, fn=_broadcast_worker)

