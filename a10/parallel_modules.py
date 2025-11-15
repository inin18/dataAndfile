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
        
        raise NotImplementedError("Implement ParallelConv2d")
    
    def forward(self, x: Tensor) -> Tensor:
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
        
        raise NotImplementedError("Implement ParallelGroupNorm")
    
    def forward(self, x: Tensor) -> Tensor:
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

