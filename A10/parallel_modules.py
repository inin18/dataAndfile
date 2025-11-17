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
        self.weight = self.conv.weight
        self.bias = self.conv.bias
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
        if halo == 0 or self.world_size==1:
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

        #implement parallel attention by Ring Attention
        self.norm = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = ParallelConv2d(in_channels, in_channels, kernel_size=1)
        self.k = ParallelConv2d(in_channels, in_channels, kernel_size=1)
        self.v = ParallelConv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = ParallelConv2d(in_channels, in_channels, kernel_size=1)
        
        #raise NotImplementedError("Implement ParallelAttnBlock")
    
    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w)

        raise NotImplementedError("Implement ParallelAttnBlock.attention")
    
    def forward(self, x: Tensor) -> Tensor:

        return

        raise NotImplementedError("Implement ParallelAttnBlock.forward")


class ParallelUpsample(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD

        self.conv = ParallelConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, process_group=process_group)
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        #raise NotImplementedError("Implement ParallelUpsample")
    
    def forward(self, x: Tensor) -> Tensor:
        # - Handle upsampling in sequence-parallel context
        # - Ensure the upsampled output maintains correct sequence parallelism

        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
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

        self.norm1 = ParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, process_group=self.process_group)
        self.conv1 = ParallelConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=self.process_group)
        self.norm2 = ParallelGroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True, process_group=self.process_group)
        self.conv2 = ParallelConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, process_group=self.process_group)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, process_group=self.process_group)
        
        #raise NotImplementedError("Implement ParallelResnetBlock")
    
    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h
        
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
        #- Replace all components with parallel versions:
        #  - `conv_in` → `ParallelConv2d`
        #  - `norm_out` → `ParallelGroupNorm`
        #  - `conv_out` → `ParallelConv2d`
        #  - `mid.attn_1` → `ParallelAttnBlock`
        #  - All `ResnetBlock` components should use parallel norms and convs
        #  - All `Upsample` components → `ParallelUpsample`
        #  - Any attention blocks in upsampling layers → `ParallelAttnBlock`
        # z to block_in
        self.conv_in = ParallelConv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1, process_group=self.process_group)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ParallelResnetBlock(in_channels=block_in, out_channels=block_in, process_group=self.process_group)
        self.mid.attn_1 = AttnBlock(block_in) # not implemented yet: ParallelAttnBlock
        self.mid.block_2 = ParallelResnetBlock(in_channels=block_in, out_channels=block_in, process_group=self.process_group)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ParallelResnetBlock(in_channels=block_in, out_channels=block_outi, process_group=self.process_group))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = ParallelUpsample(block_in, process_group=self.process_group)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = ParallelGroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True, process_group=self.process_group)
        self.conv_out = ParallelConv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1, process_group=self.process_group)
        
        #raise NotImplementedError("Implement ParallelDecoder")
    
    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        return self.conv_out(h)
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

        #- Use the parallel decoder while keeping the encoder unchanged
        #- Ensure the interface matches the original `AutoEncoder
        self.encoder = Encoder(config)
        self.decoder = ParallelDecoder(config, process_group=self.process_group)
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor
        self.sample = config.sample
        
        #raise NotImplementedError("Implement ParallelAutoEncoder")
    
    def decode(self, z: Tensor) -> Tensor:
        T = z.shape[2]
        z = rearrange(z, "b c t h w -> (b t) c h w")
        z = z / self.scale_factor + self.shift_factor
        x = self.decoder(z)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        
        return x
        
        raise NotImplementedError("Implement ParallelAutoEncoder.decode")
    
    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, DiagonalGaussianDistribution, Tensor]:

        # encode
        #x.shape[2]
        T = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        params = self.encoder(x)
        params = rearrange(params, "(b t) c h w -> b c t h w", t=T)
        posterior = DiagonalGaussianDistribution(params)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = self.scale_factor * (z - self.shift_factor)

        # decode
        x_rec = self.decode(z)

        return x_rec, posterior, z

        raise NotImplementedError("Implement ParallelAutoEncoder.forward")
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

