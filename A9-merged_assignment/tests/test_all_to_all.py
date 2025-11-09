import torch

from src import collectives
from tests.utils import run_distributed


def _all_to_all_worker(rank: int, world_size: int) -> None:
    """
    Each rank creates a tensor [0, 1, 2, ..., world_size*2 - 1] + rank * 100
    So rank 0: [0, 1, 2, ..., world_size*2-1]
    Rank 1: [100, 101, 102, ..., 100 + world_size*2-1]
    etc.

    After all_to_all, each rank should receive chunk r from each rank.
    """
    chunk_size = 2
    total_size = world_size * chunk_size

    # Create input tensor: [rank*100, rank*100+1, rank*100+2, ...]
    x = torch.arange(total_size, dtype=torch.float32) + rank * 100

    # Perform all_to_all
    y = collectives.my_all_to_all(x, scatter_dim=0, gather_dim=0)

    # Verify output shape
    assert y.shape[0] == total_size, f"Expected shape[0]={total_size}, got {y.shape[0]}"

    # Build expected output:
    # Rank r should receive:
    # - chunk r from rank 0: [r*chunk_size, r*chunk_size+1, ..., (r+1)*chunk_size-1]
    # - chunk r from rank 1: [100 + r*chunk_size, ...]
    # - chunk r from rank i: [i*100 + r*chunk_size, ...]
    expected_list = []
    for sender_rank in range(world_size):
        chunk_start = rank * chunk_size
        chunk_end = (rank + 1) * chunk_size
        for i in range(chunk_start, chunk_end):
            expected_list.append(float(sender_rank * 100 + i))

    expected = torch.tensor(expected_list, dtype=torch.float32)

    assert torch.allclose(y, expected), (
        f"Rank {rank}: Expected {expected.tolist()}, got {y.tolist()}"
    )


def test_all_to_all_basic() -> None:
    run_distributed(world_size=4, fn=_all_to_all_worker)


def _all_to_all_2d_worker(rank: int, world_size: int) -> None:
    """
    Test all_to_all with 2D tensors.
    Each rank creates a tensor of shape [world_size * 2, 3].
    """
    chunk_size = 2
    total_size = world_size * chunk_size
    feature_dim = 3

    # Create input: each element is rank*1000 + row*10 + col
    x = torch.zeros(total_size, feature_dim, dtype=torch.float32)
    for i in range(total_size):
        for j in range(feature_dim):
            x[i, j] = rank * 1000 + i * 10 + j

    # Perform all_to_all
    y = collectives.my_all_to_all(x, scatter_dim=0, gather_dim=0)

    # Verify shape
    assert y.shape == (total_size, feature_dim)

    # Build expected output
    expected = torch.zeros(total_size, feature_dim, dtype=torch.float32)
    for sender_rank in range(world_size):
        chunk_start_idx = sender_rank * chunk_size
        chunk_end_idx = (sender_rank + 1) * chunk_size
        
        # In the output, chunk from sender_rank goes to position [sender_rank*chunk_size : (sender_rank+1)*chunk_size]
        for local_idx, global_idx in enumerate(range(chunk_start_idx, chunk_end_idx)):
            # This is the chunk that rank 'rank' receives from 'sender_rank'
            # The chunk corresponds to rows [rank*chunk_size : (rank+1)*chunk_size] from sender_rank's input
            orig_row_idx = rank * chunk_size + local_idx
            for j in range(feature_dim):
                expected[global_idx, j] = sender_rank * 1000 + orig_row_idx * 10 + j

    assert torch.allclose(y, expected), (
        f"Rank {rank}: Mismatch in 2D all_to_all"
    )


def test_all_to_all_2d() -> None:
    run_distributed(world_size=4, fn=_all_to_all_2d_worker)

