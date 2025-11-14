from typing import Tuple

import torch
import torch.distributed as dist

def _check_initialized() -> Tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError(
            "Process group is not initialized. "
            "Did you call dist_init.init_process_group() before using collectives?"
        )
    return dist.get_rank(), dist.get_world_size()


def my_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast `tensor` from rank `src` to all ranks.

    Constraints:
    -----------
    * You MUST NOT call dist.broadcast or any other built-in collective.
    * You MUST implement this operation ONLY with dist.send and dist.recv.
    * This function must be called on ALL ranks of the process group.

    Semantics:
    ----------
    * On `src` rank, `tensor` contains the source data.
    * On all other ranks, `tensor` is an allocated tensor with correct shape/dtype
      (contents are ignored).
    * After the call, all ranks must hold the same data as the original `src` tensor.
    
    Args:
        tensor: The tensor to broadcast (in-place).
        src: The source rank (default: 0).
    
    Returns:
        The tensor after broadcasting (same reference as input).
    """
    rank, world_size = _check_initialized()
    
    # TODO: Implement broadcast using only dist.send and dist.recv
    if rank == src:
        for each in range(world_size):
            if each != rank:
                dist.send(tensor=tensor, dst=each) 
    else:
        dist.recv(tensor=tensor, src=src)

    return tensor

    raise NotImplementedError("TODO: Implement my_broadcast")


def my_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    Allreduce (sum) implemented using only send/recv.

    Semantics:
    ----------
    Let x_r be the input tensor on rank r.
    After the call, ALL ranks hold the element-wise sum: sum_r x_r.

    Constraints:
    ------------
    * No built-in collectives (all_reduce, reduce, gather, etc.) may be used.
    * Only send/recv and local CPU tensor ops are allowed.
    * Must be called on ALL ranks.
    * Operates in-place on `tensor` and also returns it.
    
    Args:
        tensor: The tensor to reduce (in-place).
    
    Returns:
        The tensor after allreduce (same reference as input).
    """
    rank, world_size = _check_initialized()
    
    # TODO: Implement allreduce using only dist.send and dist.recv
    # Hint: A simple algorithm is to gather to a root rank, sum, then broadcast
    root = 0
    if rank == root:
        tensor_tmp = torch.empty_like(tensor) 
        for each in range(world_size):
            if each != root:
                dist.recv(tensor=tensor_tmp, src=each)
                tensor.add_(tensor_tmp)
    else:
        dist.send(tensor=tensor, dst=root)

    if rank == root:
        for each in range(world_size):
            if each!=root:
                dist.send(tensor=tensor, dst=each)
    else:
        dist.recv(tensor=tensor, src=root)

    return tensor

    raise NotImplementedError("TODO: Implement my_allreduce")


def my_allgather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Allgather implemented using only send/recv.

    Semantics:
    ----------
    * Each rank r starts with a local tensor of shape [N, ...].
    * After the call, all ranks hold a tensor of shape [world_size * N, ...]
      which is the concatenation of all ranks tensors in rank order:
      [tensor_0; tensor_1; ...; tensor_{world_size-1}].

    Constraints:
    ------------
    * No built-in all_gather or similar collectives.
    * Only send/recv and CPU tensor ops.
    * Must be called on ALL ranks.
    
    Args:
        tensor: The local tensor to gather.
    
    Returns:
        The gathered tensor (new tensor, not in-place).
    """
    rank, world_size = _check_initialized()
    
    # TODO: Implement allgather using only dist.send and dist.recv
    root = 0
    chunk = tensor.size(0)
    gathered_size = chunk*world_size
    gathered_tensor = torch.empty([gathered_size]+list(tensor.size())[1:], dtype=tensor.dtype, device=tensor.device)
    if rank != root:
        dist.send(tensor=tensor, dst=root)
    else:
        gathered_tensor[chunk*root:chunk*(root+1)].copy_(tensor)
        tensor_tmp = torch.empty_like(tensor)
        for each in range(world_size):
            if each!=root:
                dist.recv(tensor=tensor_tmp, src=each)
                gathered_tensor[chunk*each:chunk*(each+1)].copy_(tensor_tmp)

    if rank==root:
        for each in range(world_size):
            if each!=root:
                dist.send(tensor=gathered_tensor, dst=each)
    else:
        dist.recv(tensor=gathered_tensor, src=root)

    return gathered_tensor
    raise NotImplementedError("TODO: Implement my_allgather")


def my_reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce-scatter (sum) implemented using only send/recv.

    Semantics:
    ----------
    * Each rank r starts with a tensor of shape [world_size * N, ...].
    * The input tensor is conceptually divided into world_size equal chunks along dim 0.
    * Chunk i from all ranks are summed element-wise.
    * Rank i receives the summed result of chunk i.
    * The output on rank r has shape [N, ...] and contains sum of chunk r from all ranks.

    Constraints:
    ------------
    * No built-in reduce_scatter or similar collectives.
    * Only send/recv and CPU tensor ops.
    * Must be called on ALL ranks.
    * Assumes tensor.shape[0] is divisible by world_size.
    
    Args:
        tensor: The tensor to reduce and scatter.
    
    Returns:
        The reduced chunk for this rank (new tensor).
    """
    rank, world_size = _check_initialized()
    
    # TODO: Implement reduce_scatter using only dist.send and dist.recv
    root = 0
    tensor_tmp = torch.zeros(tensor.size())
    tensor_buffer = torch.empty_like(tensor)

    if rank == root:
        tensor_tmp.add_(tensor)
        for each in range(world_size):
            if each != root:
                dist.recv(tensor=tensor_buffer, src=each)
                tensor_tmp.add_(tensor_buffer)
    else:
        dist.send(tensor=tensor, dst=root)

    chunk_size = tensor_tmp.size()[0]//world_size
    chunk_tensor = torch.zeros([chunk_size]+list(tensor.size())[1:], dtype=tensor.dtype,device=tensor.device)
    out_tensor = torch.empty_like(chunk_tensor)
    if rank==root:
        for each in range(world_size):
            chunk_tensor=tensor_tmp[each*chunk_size:(each+1)*chunk_size]
            if each != root:
                dist.send(tensor=chunk_tensor, dst=each)
            else:
                out_tensor = chunk_tensor
    else:
        dist.recv(tensor=out_tensor, src=root)

    return out_tensor

    raise NotImplementedError("TODO: Implement my_reduce_scatter")


def my_all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    """
    All-to-all with flexible scatter and gather dimensions.
    
    This is used for Ulysses-style sequence parallelism where we need to
    redistribute data across different dimensions.
    
    Args:
        tensor: Input tensor to redistribute
        scatter_dim: Dimension to split and scatter across ranks (must be divisible by world_size)
        gather_dim: Dimension to gather received chunks along
    
    Returns:
        Redistributed tensor with scatter_dim scattered and gather_dim gathered
        
    Semantics:
    ----------
    * The input is split into `world_size` chunks along `scatter_dim`
    * Rank `r` sends chunk `i` (along scatter_dim) to rank `i`
    * Rank `r` receives chunk `r` (along scatter_dim) from each rank and gathers them along `gather_dim`
    * When `scatter_dim != gather_dim`: scatter_dim shrinks by factor of world_size, gather_dim grows by factor of world_size
    * When `scatter_dim == gather_dim`: output shape equals input shape (redistribution only)
    
    Example:
        Input on each rank:  [batch, seq_len/N, n_heads, head_dim] where N=world_size
        scatter_dim=2 (heads), gather_dim=1 (sequence)
        Output on each rank: [batch, seq_len, n_heads/N, head_dim]
        
    Constraints:
    ------------
    * No built-in all_to_all or similar collectives.
    * Only send/recv (including isend/irecv) and CPU tensor ops.
    * Must be called on ALL ranks.
    """
    rank, world_size = _check_initialized()
    
    # TODO: Implement all_to_all using only dist.send and dist.recv
    # Hint: Use isend/irecv to avoid deadlocks
    tensor_size = list(tensor.size())
    chunk_size =tensor_size[scatter_dim]//world_size
    chunks = tensor.split(chunk_size, dim=scatter_dim)

    tensor_buffers = [torch.empty_like(chunks[0]) for _ in range(world_size)]
    recv_req = []
    for each in range(world_size):
        if each != rank:
            tag = rank + world_size*each # tag//world_size, tag%world_size
            recv_req.append(dist.irecv(tensor=tensor_buffers[each], src=each,tag=tag))
        else:
            tensor_buffers[rank] = chunks[rank]
    
    send_req = []
    for each in range(world_size):
        if each != rank:
            send_req.append(dist.isend(tensor=chunks[each], dst=each, tag=each+world_size*rank))

    for req in recv_req:
        req.wait()
    for req in send_req:
        req.wait()
    tensor_gathered = torch.cat(tensor_buffers, dim=gather_dim)

    return tensor_gathered


    raise NotImplementedError("TODO: Implement my_all_to_all")

