from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Import custom my_all_to_all for Ulysses attention
from .collectives import my_all_to_all


@dataclass
class TransformerConfig:
    d_model: int = 32
    n_heads: int = 4
    dim_ff: int = 64
    dropout_p: float = 0.0


class BasicTransformerBlock(nn.Module):
    """
    A small Transformer encoder block for teaching purposes.

    Shape:
        Input:  [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        dim_ff = config.dim_ff

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # Self-attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

        # LayerNorms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(config.dropout_p)

    # ---- plain single-process forward (reference) ----

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        # [B, S, D] -> [B, n_heads, S, head_dim]
        n_heads = self.config.n_heads
        return (
            x.view(bsz, seq_len, n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference single-process encoder block."""
        bsz, seq_len, d_model = x.shape
        h = self.ln1(x)

        # Self-attention
        q = self._shape(self.q_proj(h), bsz, seq_len)
        k = self._shape(self.k_proj(h), bsz, seq_len)
        v = self._shape(self.v_proj(h), bsz, seq_len)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, S, head_dim]

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, n_heads, head_dim]
        attn_output = attn_output.view(bsz, seq_len, d_model)
        attn_output = self.o_proj(attn_output)
        x = x + self.dropout(attn_output)

        # Feed-forward
        h2 = self.ln2(x)
        ff = self.fc2(F.relu(self.fc1(h2)))
        x = x + self.dropout(ff)
        return x

    # ---- distributed parallel forwards to be implemented by students ----

    def init_tensor_parallel_shards(self):
        """
        Initialize sharded weights for tensor parallelism.
        Each rank extracts its portion of the model weights.
        
        Requirements:
        -------------
        * Store sharded weights in the following attributes:
          - self.q_proj_shard: Shard of Q projection weight (column-parallel)
          - self.k_proj_shard: Shard of K projection weight (column-parallel)
          - self.v_proj_shard: Shard of V projection weight (column-parallel)
          - self.o_proj_shard: Shard of O projection weight (row-parallel)
          - self.fc1_shard: Shard of FC1 weight (column-parallel)
          - self.fc1_bias_shard: Shard of FC1 bias (if present)
          - self.fc2_shard: Shard of FC2 weight (row-parallel)
        * Column-parallel means sharding along the output dimension (rows of weight matrix).
        * Row-parallel means sharding along the input dimension (columns of weight matrix).
        * Each rank should extract its portion based on rank and world_size.
        """
        # TODO: Implement this method
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.q_proj_shard = torch.chunk(self.q_proj.weight.data, world_size, dim=0)[rank]
        self.k_proj_shard = torch.chunk(self.k_proj.weight.data, world_size, dim=0)[rank]
        self.v_proj_shard = torch.chunk(self.v_proj.weight.data, world_size, dim=0)[rank]
        self.o_proj_shard = torch.chunk(self.o_proj.weight.data, world_size, dim=1)[rank]
        self.fc1_shard = torch.chunk(self.fc1.weight.data, world_size, dim=0)[rank]
        if self.fc1.bias != None:
            self.fc1_bias_shard = torch.chunk(self.fc1.bias.data, world_size, dim=0)[rank]
        self.fc2_shard = torch.chunk(self.fc2.weight.data, world_size, dim=1)[rank]
        
        #raise NotImplementedError("TODO: Implement init_tensor_parallel_shards")

    def forward_tensor_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tensor-parallel forward pass using sharded weights.

        Requirements:
        -------------
        * Assume init_tensor_parallel_shards() has been called first.
        * Use the sharded weights stored in self.*_shard attributes.
        * Assume a process group has been initialized (Gloo, CPU).
        * Use dist.get_rank() and dist.get_world_size() as your model-parallel
          rank and world size.
        * Implement 1D tensor parallelism over the model dimension d_model.
        * You may use standard PyTorch distributed collectives (dist.all_reduce, dist.all_gather).
        * You may also use your custom collectives from Task 1 if you prefer.
        * On every rank, the returned tensor MUST match self.forward(x)
          (up to numerical tolerance).
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
        
        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        # TODO: Implement this method
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        bsz, seq_len, d_model = x.shape
        h = self.ln1(x)

        # Self-attention
        q_shard = F.linear(h, self.q_proj_shard)
        q = [torch.empty_like(q_shard, dtype=q_shard.dtype) for _ in range(world_size)]
        dist.all_gather(q, q_shard)
        q = torch.cat(q, dim=-1)
        q = self._shape(q, bsz, seq_len)

        k_shard = F.linear(h,self.k_proj_shard)
        k = [torch.empty_like(k_shard, dtype=k_shard.dtype) for _ in range(world_size)]
        dist.all_gather(k, k_shard)
        k = torch.cat(k, dim=-1)
        k = self._shape(k, bsz, seq_len)

        v_shard = F.linear(h, self.v_proj_shard)
        v = [torch.empty_like(v_shard, dtype=v_shard.dtype) for _ in range(world_size)]
        dist.all_gather(v, v_shard)
        v = torch.cat(v, dim=-1)
        v = self._shape(v, bsz, seq_len)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, S, head_dim]

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, n_heads, head_dim]
        attn_output = attn_output.view(bsz, seq_len, d_model)
        attn_output_shared = F.linear(attn_output, self.o_proj_shard.transpose(0,1))
        
        dist.all_reduce(attn_output_shared)
        attn_output_gathered = torch.empty_like(x)
        attn_output_gathered_list = list(torch.chunk(attn_output_gathered, world_size, dim=-1))
        dist.all_gather(attn_output_gathered_list, attn_output_shared)

        x = x + self.dropout(attn_output_gathered)

        # Feed-forward
        h2 = self.ln2(x)
        fc1_output_shard = F.linear(h2, self.fc1_shard, bias = self.fc1_bias_shard if self.fc1_bias_shard is not None else False)
        fc1_output = [torch.empty_like(fc1_output_shard, dtype=fc1_output_shard.dtype) for _ in range(world_size)]
        dist.all_gather(fc1_output, fc1_output_shard)
        fc1_output = torch.cat(fc1_output,dim=-1)

        fc1_fc2_shard = torch.chunk(fc1_output, world_size, dim=-1)[rank]

        ff = F.linear(F.relu(fc1_fc2_shard), self.fc2_shard)
#        ff = torch.matmul(F.relu(fc1_fc2_shard), self.fc2_shard.transpose(0,1))
        dist.all_reduce(ff)
        x = x + self.dropout(ff)

        return x
        raise NotImplementedError("TODO: Implement forward_tensor_parallel")

    def forward_ulysses(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ulysses-style sequence-parallel forward pass.

        Requirements:
        -------------
        * Input x is a shard of the sequence: shape [batch, local_seq_len, d_model]
          where local_seq_len = seq_len / world_size.
        * Each rank processes a different subsequence.
        * Use your custom my_all_to_all() function from Task 1 for communication.
          This is required because the Gloo backend does not support dist.all_to_all().
        * You may also use dist.all_gather() or dist.all_reduce() if needed.
        * Self-attention requires global attention (tokens can attend to all positions).
          Use all-to-all to exchange information between ranks.
        * Returns the local shard of the output: shape [batch, local_seq_len, d_model].
        * Parallelism is determined by world_size.
        * The output must match the corresponding slice of self.forward(x_full)
          where x_full is the full sequence.
        
        Hint:
        -----
        Ulysses sequence parallelism approach:
        1. Start with sequence shard: (batch, seq_len/N, d_model)
        2. Compute Q, K, V for local shard: (batch, seq_len/N, n_heads, head_dim)
        3. All-to-all to redistribute (scatter heads, gather sequence): 
           (batch, seq_len, n_heads/N, head_dim)
        4. Compute attention on full sequence for subset of heads
        5. All-to-all reverse (scatter sequence, gather heads):
           (batch, seq_len/N, n_heads, head_dim)
        6. Output projection and FFN on local sequence shard
        7. Return local shard
        
        Args:
            x: Input tensor shard of shape [batch, local_seq_len, d_model].
        
        Returns:
            Output tensor shard of shape [batch, local_seq_len, d_model].
        """
        # TODO: Implement this method using my_all_to_all from collectives.py
        raise NotImplementedError("TODO: Implement forward_ulysses")
