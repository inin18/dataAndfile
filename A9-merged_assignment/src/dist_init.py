import os
from typing import Tuple

import torch.distributed as dist


def init_process_group(backend: str = "gloo") -> Tuple[int, int]:
    """
    Initialize a torch.distributed process group using environment variables
    set by torchrun.

    Returns
    -------
    (rank, world_size)
    """
    if dist.is_initialized():
        # Already initialized; just return
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # MASTER_ADDR and MASTER_PORT are also set by torchrun; we rely on them.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size

