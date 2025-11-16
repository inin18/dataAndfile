from typing import Optional

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import silu as swish

from autoencoder_2d import (
    AutoEncoderConfig,
    DiagonalGaussianDistribution
)


class ParallelConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        # Assume dist is initialized
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0 
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # raise NotImplementedError("Implement ParallelConv2d")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Extends `nn.Conv2d` to work with sequence-parallel inputs
        Inputs are split along the width dimension (W), so each process gets (B, C, H, W_local)
        Handle input/output tensor shapes correctly when width dimension is split
        Communication at boundaries**: Since convolutions require neighboring pixels, you need to communicate boundary regions between devices to ensure correct convolution results at the edges of each device's width chunk
        """
        x_shard = torch.chunk(x, self.world_size, dim=-1)[self.rank]
        B, C, H, W_local = x_shard.size()

        halo = self.kernel_size // 2
        right = torch.zeros(B, C, H, halo, device=x_shard.device)
        left = torch.zeros(B, C, H, halo, device=x_shard.device)

        req = []

        if self.rank < self.world_size - 1:
            req.append(dist.irecv(right, src = self.rank+1, group=self.process_group))
        if self.rank > 0:
            req.append(dist.irecv(left, src=self.rank - 1, group=self.process_group))
            
        if self.rank < self.world_size - 1:
            req.append(dist.isend(x_shard[:, :, :, -halo:].contiguous(), dst=self.rank + 1, group=self.process_group))
        if self.rank > 0:
            req.append(dist.isend(x_shard[:, :, :, :halo].contiguous(), dst=self.rank - 1, group=self.process_group))

        for r in req:
            r.wait()

        x_shard = torch.cat([left, x_shard, right], dim=-1)
        if self.rank==0 and self.padding>0:
            x_shard = torch.nn.functional.pad(x_shard, (self.padding, 0))
        if self.rank==self.world_size-1 and self.padding>0:
            x_shard = torch.nn.functional.pad(x_shard, (0, self.padding))
        x_shard = torch.nn.functional.pad(x_shard, (0, 0,self.padding, self.padding))
        
        out_shard = self.conv(x_shard)
        w_out = out_shard.size(-1)

        import math
        left_crop = int(math.ceil((halo if self.rank>0 else self.padding)/self.stride))
        right_crop = w_out-int(math.ceil((halo if self.rank<self.world_size-1 else self.padding)/self.stride))
        out_shard = out_shard[..., left_crop:right_crop]
        
        out_gathered = [torch.empty_like(out_shard, device=x.device) for _ in range(self.world_size)]
        dist.all_gather(out_gathered, out_shard, group=self.process_group)
        out = torch.cat(out_gathered, dim=-1)
        return out
        raise NotImplementedError("Implement ParallelConv2d.forward")


class ParallelGroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.norm = nn.GroupNorm(num_groups, num_channels, eps, affine)
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # raise NotImplementedError("Implement ParallelGroupNorm")
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        x_shard = torch.chunk(x, self.world_size, dim=-1)[self.rank]
        out_shard = self.norm(x_shard)
        out = [torch.empty_like(out_shard) for _ in range(self.world_size)]
        dist.all_gather(out, out_shard, self.process_group)
        out = torch.cat(out, dim=-1)
        return out
        raise NotImplementedError("Implement ParallelGroupNorm.forward")
    
    


class ParallelAttnBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.in_channels = in_channels
        
        raise NotImplementedError("Implement ParallelAttnBlock")
    
    def attention(self, h_: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelAttnBlock.attention")
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelAttnBlock.forward")


class ParallelUpsample(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        
        raise NotImplementedError("Implement ParallelUpsample")
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelUpsample.forward")


class ParallelResnetBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        raise NotImplementedError("Implement ParallelResnetBlock")
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelResnetBlock.forward")


class ParallelDecoder(nn.Module):
    
    def __init__(
        self,
        config: AutoEncoderConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config.z_channels, curr_res, curr_res)
        
        raise NotImplementedError("Implement ParallelDecoder")
    
    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelDecoder.forward")


class ParallelAutoEncoder(nn.Module):
    
    def __init__(
        self,
        config: AutoEncoderConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        
        from autoencoder_2d import Encoder
        
        raise NotImplementedError("Implement ParallelAutoEncoder")
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError("Implement ParallelAutoEncoder.decode")
    
    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, DiagonalGaussianDistribution, Tensor]:
        raise NotImplementedError("Implement ParallelAutoEncoder.forward")
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

