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
        x_shard = x #torch.chunk(x, self.world_size, dim=-1)[self.rank]
        B, C, H, W_local = x_shard.size()

        halo = self.kernel_size // 2
        if halo == 0:
            right = torch.tensor(0) 
            left = torch.tensor(0)
        else:
            right = torch.zeros(B, C, H, halo, device=x_shard.device)
            left = torch.zeros(B, C, H, halo, device=x_shard.device)

            req = []
            if self.rank%2==0:
                if self.rank < self.world_size - 1:
                    req.append(dist.irecv(right, src = self.rank+1, group=self.process_group))
                if self.rank > 0:
                    req.append(dist.irecv(left, src=self.rank - 1, group=self.process_group))
            
                if self.rank < self.world_size - 1:
                    req.append(dist.isend(x_shard[:, :, :, -halo:].contiguous(), dst=self.rank + 1, group=self.process_group))
                if self.rank > 0:
                    req.append(dist.isend(x_shard[:, :, :, :halo].contiguous(), dst=self.rank - 1, group=self.process_group))
            else:
                if self.rank < self.world_size - 1:
                    req.append(dist.isend(x_shard[:, :, :, -halo:].contiguous(), dst=self.rank + 1, group=self.process_group))
                if self.rank > 0:
                    req.append(dist.isend(x_shard[:, :, :, :halo].contiguous(), dst=self.rank - 1, group=self.process_group))

                if self.rank < self.world_size - 1:
                    req.append(dist.irecv(right, src = self.rank+1, group=self.process_group))
                if self.rank > 0:
                    req.append(dist.irecv(left, src=self.rank - 1, group=self.process_group))
            

            for i,r in enumerate(req):
                r.wait()
            if self.rank==0:
                x_shard = torch.cat([x_shard, right], dim=-1)
            elif self.rank==self.world_size-1:
                x_shard = torch.cat([left, x_shard], dim=-1)
            else:
                x_shard = torch.cat([left, x_shard, right], dim=-1)

        if self.rank==0 and self.padding>0:
            x_shard = torch.nn.functional.pad(x_shard, (self.padding, 0))
        if self.rank==self.world_size-1 and self.padding>0:
            x_shard = torch.nn.functional.pad(x_shard, (0, self.padding))
        x_shard = torch.nn.functional.pad(x_shard, (0, 0,self.padding, self.padding))
        
        out_shard = self.conv(x_shard)
        
        return out_shard
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
        self.eps = eps
        self.affine = affine
        self.num_groups = num_groups
        self.num_channels = num_channels
        self._has_broadcast = False
        self.weight=self.norm.weight
        self.bias=self.norm.bias

        # raise NotImplementedError("Implement ParallelGroupNorm")
    def forward(self, x: Tensor) -> Tensor:
        '''
        if self.norm.affine and not self._has_broadcast:
            if self.norm.weight is not None:
                dist.broadcast(self.norm.weight, src=0, group=self.process_group)
            if self.norm.bias is not None:
                dist.broadcast(self.norm.bias, src=0, group=self.process_group)
            self._has_broadcast = True
            '''
        B, C, H, W = x.size()
        G = self.num_groups
        x_shard = x.to(torch.float64)
        x_grouped = x_shard.view(B,G,C//G, H, x_shard.size(-1))
        local_sum = x_grouped.sum(dim=(2,3,4))
        local_sumsq = (x_grouped**2).sum(dim=(2,3,4))
        cnt = C//G * H * x_grouped.size(-1)

        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.process_group)
        dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM,group=self.process_group)
        cnt = cnt *self.world_size
        cnt_tensor = torch.tensor(cnt, dtype=torch.float64,device=x.device).view(1,1).expand(B, G)

        mean = (local_sum/cnt_tensor).view(B, G, 1, 1, 1)
        var = (local_sumsq/cnt_tensor-mean.squeeze()**2).view(B, G, 1, 1, 1)
        var = torch.clamp(var, min=0.0)
        eps = torch.tensor(self.eps, dtype=torch.float64,device=x.device).view(1,1).expand(B, G).view(B,G,1,1,1)
        invstd = 1.0/torch.sqrt(var+eps)

        x_norm = (x_grouped-mean)*invstd
        x_norm = x_norm.view(B, C, H, x_shard.size(-1))

        if self.norm.affine:
            w = self.norm.weight.view(1, C, 1, 1).to(device=x_shard.device, dtype=self.norm.weight.dtype)
            b = self.norm.bias.view(1, C, 1, 1).to(device=x_shard.device, dtype=self.norm.weight.dtype)
            x_norm = x_norm * w + b
        out = x_norm.to(x.dtype)
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

