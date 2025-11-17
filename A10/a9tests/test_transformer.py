import torch

from src.transformer import BasicTransformerBlock, TransformerConfig
from tests.utils import run_distributed


def _tp_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    cfg = TransformerConfig(d_model=32, n_heads=4, dim_ff=64, dropout_p=0.0)
    block = BasicTransformerBlock(cfg)

    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, cfg.d_model)

    # Reference single-process semantics
    y_ref = block.forward(x.clone())

    # Initialize tensor parallel shards
    block.init_tensor_parallel_shards()

    # Tensor-parallel forward
    y_tp = block.forward_tensor_parallel(x.clone())

    assert y_tp.shape == y_ref.shape
    assert torch.allclose(y_tp, y_ref, atol=1e-5, rtol=1e-5)


def _ulysses_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    cfg = TransformerConfig(d_model=32, n_heads=4, dim_ff=64, dropout_p=0.0)
    block = BasicTransformerBlock(cfg)

    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, cfg.d_model)

    # Reference single-process semantics
    y_ref = block.forward(x.clone())

    # Split sequence across ranks (Ulysses expects a shard as input)
    local_seq_len = seq_len // world_size
    x_shard = x[:, rank * local_seq_len:(rank + 1) * local_seq_len, :].clone()
    
    # Ulysses-style sequence parallel (expects sharded input)
    y_shard = block.forward_ulysses(x_shard)
    
    # Gather full output from all ranks for comparison
    y_ulysses = torch.cat([
        torch.zeros(batch, local_seq_len, cfg.d_model) for _ in range(world_size)
    ], dim=1)
    y_ulysses[:, rank * local_seq_len:(rank + 1) * local_seq_len, :] = y_shard
    
    # AllGather to get full result
    import torch.distributed as dist
    gathered = [torch.zeros_like(y_shard) for _ in range(world_size)]
    dist.all_gather(gathered, y_shard)
    y_ulysses = torch.cat(gathered, dim=1)

    assert y_ulysses.shape == y_ref.shape
    assert torch.allclose(y_ulysses, y_ref, atol=1e-5, rtol=1e-5)


def test_transformer_tensor_parallel_matches_reference() -> None:
    # Will fail with NotImplementedError until student implements TP
    run_distributed(world_size=2, fn=_tp_worker)


def test_transformer_ulysses_matches_reference() -> None:
    # Will fail with NotImplementedError until student implements Ulysses SP
    run_distributed(world_size=2, fn=_ulysses_worker)

