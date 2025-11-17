import torch
import torch.distributed as dist

from src.collectives import my_reduce_scatter
from tests.utils import run_distributed


def _reduce_scatter_worker(rank: int, world_size: int) -> None:
    """
    Test reduce_scatter with simple inputs.
    Each rank contributes a tensor [rank*world_size, rank*world_size+1, ..., rank*world_size+world_size-1].
    After reduce_scatter, each rank r should have sum of chunk r from all ranks.
    """
    # Each rank creates input: [rank*world_size, rank*world_size+1, ..., rank*world_size+world_size-1]
    x = torch.arange(
        rank * world_size, (rank + 1) * world_size, dtype=torch.float32
    )

    # After reduce_scatter, rank r receives sum of all ranks' r-th element
    # i.e., rank r receives: sum_{i=0}^{world_size-1} (i*world_size + r)
    #                      = r*world_size + r*world_size + ... + r*world_size + (0 + 1 + ... + world_size-1)
    #                      = r * world_size * world_size + sum(0..world_size-1)
    #                      = r * world_size^2 + world_size*(world_size-1)/2

    # Actually, let's think more carefully:
    # Rank 0 contributes: [0, 1, 2, 3] for world_size=4
    # Rank 1 contributes: [4, 5, 6, 7]
    # Rank 2 contributes: [8, 9, 10, 11]
    # Rank 3 contributes: [12, 13, 14, 15]
    #
    # After reduce_scatter:
    # Rank 0 receives sum of [0, 4, 8, 12] = 24
    # Rank 1 receives sum of [1, 5, 9, 13] = 28
    # Rank 2 receives sum of [2, 6, 10, 14] = 32
    # Rank 3 receives sum of [3, 7, 11, 15] = 36
    #
    # General formula for rank r: sum_{i=0}^{world_size-1} (i*world_size + r)
    #                           = r * world_size + sum_{i=0}^{world_size-1} i * world_size
    #                           = r * world_size + world_size * (world_size-1)*world_size/2
    #                           = r * world_size + world_size^2 * (world_size-1)/2

    y = my_reduce_scatter(x)

    # Verify shape
    assert y.shape == (1,), f"Expected shape (1,), got {y.shape}"

    # Verify value
    expected = rank * world_size + world_size * world_size * (world_size - 1) // 2
    assert torch.allclose(
        y, torch.tensor([expected], dtype=torch.float32)
    ), f"Rank {rank}: expected {expected}, got {y.item()}"


def test_reduce_scatter_basic_world_size_4() -> None:
    run_distributed(world_size=4, fn=_reduce_scatter_worker)


def _reduce_scatter_2d_worker(rank: int, world_size: int) -> None:
    """
    Test reduce_scatter with 2D tensors.
    """
    d = 3  # feature dimension
    # Each rank creates a [world_size, d] tensor
    # All values are (rank + 1.0) for simplicity
    x = torch.full((world_size, d), fill_value=rank + 1.0, dtype=torch.float32)

    y = my_reduce_scatter(x)

    # After reduce_scatter, rank r receives sum of chunk r from all ranks
    # Chunk r from rank i has value (i + 1.0)
    # Sum = 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
    expected_value = world_size * (world_size + 1) / 2

    # y should have shape [1, d] (one chunk)
    assert y.shape == (1, d), f"Expected shape (1, {d}), got {y.shape}"

    expected = torch.full((1, d), fill_value=expected_value, dtype=torch.float32)
    assert torch.allclose(
        y, expected
    ), f"Rank {rank}: expected all {expected_value}, got {y}"


def test_reduce_scatter_2d_world_size_4() -> None:
    run_distributed(world_size=4, fn=_reduce_scatter_2d_worker)


def _reduce_scatter_inverse_allgather_worker(rank: int, world_size: int) -> None:
    """
    Test that reduce_scatter is the inverse of allgather (in the sense that
    reduce_scatter(allgather(x)) should give us back a reduced version).
    """
    from src.collectives import my_allgather

    # Each rank starts with a single value
    x = torch.tensor([rank + 1.0], dtype=torch.float32)

    # Allgather: [1, 2, 3, 4] on all ranks (for world_size=4)
    gathered = my_allgather(x)

    # Now if we do reduce_scatter on gathered, each rank should receive
    # the sum of the corresponding element from all ranks.
    # But since gathered is the same on all ranks, reduce_scatter will give
    # each rank the same value * world_size.
    # Rank r receives (r+1) * world_size
    result = my_reduce_scatter(gathered)

    expected = torch.tensor([(rank + 1.0) * world_size], dtype=torch.float32)
    assert torch.allclose(
        result, expected
    ), f"Rank {rank}: expected {expected.item()}, got {result.item()}"


def test_reduce_scatter_inverse_allgather_world_size_4() -> None:
    run_distributed(world_size=4, fn=_reduce_scatter_inverse_allgather_worker)

