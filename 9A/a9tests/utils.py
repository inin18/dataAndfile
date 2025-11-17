import os
from typing import Callable

import torch.distributed as dist
import torch.multiprocessing as mp
import socket

from src import dist_init


def _find_free_port() -> int:
    """Find a free TCP port for initializing the process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, fn: Callable, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Disable libuv for Windows compatibility
    os.environ["USE_LIBUV"] = "0"

    dist_init.init_process_group(backend="gloo")
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def run_distributed(world_size: int, fn: Callable) -> None:
    """
    Helper used by tests to run `fn(rank, world_size)` on `world_size` ranks.

    It will:
    * Find a free port.
    * Spawn `world_size` worker processes with torch.multiprocessing.spawn.
    * Initialize a Gloo process group on CPU in each worker.
    """
    port = _find_free_port()
    mp.spawn(
        _worker,
        args=(world_size, fn, port),
        nprocs=world_size,
        join=True,
    )

