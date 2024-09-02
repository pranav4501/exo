# Copyright © 2023 Apple Inc.

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import AutoencoderConfig
from .unet import ResnetBlock2D, upsample_nearest
from dataclasses import dataclass, field
from exo.inference.shard import Shard
from typing import Tuple
import inspect
from ..base import IdentityBlock

class Attention(nn.Module):
    """A single head unmasked attention for use with the VAE."""

    def __init__(self, dims: int, norm_groups: int = 32):
        super().__init__()

        self.group_norm = nn.GroupNorm(norm_groups, dims, pytorch_compatible=True)
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        B, H, W, C = x.shape

        y = self.group_norm(x)

        queries = self.query_proj(y).reshape(B, H * W, C)
        keys = self.key_proj(y).reshape(B, H * W, C)
        values = self.value_proj(y).reshape(B, H * W, C)

        scale = 1 / math.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ values).reshape(B, H, W, C)

        y = self.out_proj(y)
        x = x + y

        return x


class EncoderDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample=True,
        add_upsample=True,
    ):
        super().__init__()

        # Add the resnet blocks
        self.resnets = [
            ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                groups=resnet_groups,
            )
            for i in range(num_layers)
        ]

        # Add an optional downsampling layer
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=0
            )

        # or upsampling layer
        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if "downsample" in self:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.downsample(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))

        return x


class Encoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=i < len(block_out_channels) - 1,
                add_upsample=False,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        for l in self.down_blocks:
            x = l(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """Implements the decoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shard: Shard,
        layer_range: List[int],
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels
        print(f"Decoder layer range: {layer_range}")
        self.layers_range = layer_range
        if 0 in layer_range:
            self.conv_in = nn.Conv2d(
                in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
            )
        
        if 0 in layer_range:
            self.mid_blocks = [
                ResnetBlock2D(
                    in_channels=block_out_channels[-1],
                    out_channels=block_out_channels[-1],
                    groups=resnet_groups,
                ),
                Attention(block_out_channels[-1], resnet_groups),
                ResnetBlock2D(
                    in_channels=block_out_channels[-1],
                    out_channels=block_out_channels[-1],
                    groups=resnet_groups,
                ),
            ]

        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels
        
        self.up_blocks = []
        current_layer = 1

        for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            print(in_channels, out_channels)
            if current_layer in layer_range:
                self.up_blocks.append(
                    EncoderDecoderBlock2D(
                        in_channels,
                        out_channels,
                        num_layers=layers_per_block,
                        resnet_groups=resnet_groups,
                        add_downsample=False,
                        add_upsample=i < len(block_out_channels) - 1,
                    )
                )
            else:
                self.up_blocks.append(IdentityBlock())
            current_layer += 1
        if 4 in layer_range:
            self.conv_norm_out = nn.GroupNorm(
                resnet_groups, block_out_channels[0], pytorch_compatible=True
            )
            self.conv_out = nn.Conv2d(block_out_channels[0], self.out_channels, 3, padding=1)


    def __call__(self, x):
        if 0 in self.layers_range:
            x = self.conv_in(x)
            x = self.mid_blocks[0](x)
            x = self.mid_blocks[1](x)
            x = self.mid_blocks[2](x)
        for l in self.up_blocks:
            x = l(x)
        if 4 in self.layers_range:
            x = self.conv_norm_out(x)
            x = nn.silu(x)
            x = self.conv_out(x)
        return x


class Autoencoder(nn.Module):
    """The autoencoder that allows us to perform diffusion in the latent space."""

    def __init__(self, config: AutoencoderConfig, shard: Shard):
        super().__init__()
        self.shard = shard
        self.start_layer = shard.start_layer
        self.end_layer = shard.end_layer
        self.layers_range = list(range(self.start_layer, self.end_layer+1))
        self.latent_channels = config.latent_channels_in
        self.scaling_factor = config.scaling_factor
        self.decoder_only = True  # stable diffusion text to speech only uses decoder from the autoencoder
        if not self.decoder_only:
            self.encoder = Encoder(
                config.in_channels,
                config.latent_channels_out,
                config.block_out_channels,
                config.layers_per_block,
                resnet_groups=config.norm_num_groups,
            )
            self.quant_proj = nn.Linear(
            config.latent_channels_out, config.latent_channels_out
            )
        self.decoder = Decoder(
            config.latent_channels_in,
            config.out_channels,
            shard,
            self.layers_range,
            config.block_out_channels,
            config.layers_per_block + 1,
            resnet_groups=config.norm_num_groups,
        )

        self.post_quant_proj = nn.Conv2d(
            4,4,1
        )

    def decode(self, z):
        if 0 in self.layers_range:
            z = z / self.scaling_factor
            z=self.post_quant_proj(z)
        return self.decoder(z)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mean, logvar = x.split(2, axis=-1)
        mean = mean * self.scaling_factor
        logvar = logvar + 2 * math.log(self.scaling_factor)

        return mean, logvar

    def __call__(self, x, key=None):
        mean, logvar = self.encode(x)
        z = mx.random.normal(mean.shape, key=key) * mx.exp(0.5 * logvar) + mean
        x_hat = self.decode(z)

        return dict(x_hat=x_hat, z=z, mean=mean, logvar=logvar)

    def sanitize(self, weights):
        layers = self.layers_range
        sanitized_weights = {}
        for key, value in weights.items():
            # Map encoder layers
            flag=0
            if((('encoder' in key) and not self.decoder_only) or 'decoder' in key):
                if "conv_in" in key and 0 in layers:
                    flag=1
                if "mid.block_1" in key and 0 in layers:
                    key = key.replace("mid.block_1", "mid_blocks.0")
                    flag=1
                if "mid.block_2" in key and 0 in layers:
                    key = key.replace("mid.block_2", "mid_blocks.2")
                    flag=1
                if "mid.attn_1.k" in key and 0 in layers:
                    key = key.replace("mid.attn_1.k", "mid_blocks.1.key_proj")
                    flag=1
                if "mid.attn_1.q" in key and 0 in layers:
                    key = key.replace("mid.attn_1.q", "mid_blocks.1.query_proj")
                    flag=1
                if "mid.attn_1.v" in key and 0 in layers:
                    key = key.replace("mid.attn_1.v", "mid_blocks.1.value_proj")
                    flag=1
                if "mid.attn_1.proj_out" in key and 0 in layers:
                    key = key.replace("mid.attn_1.proj_out", "mid_blocks.1.out_proj")
                    flag=1
                if "mid.attn_1.norm" in key and 0 in layers:
                    key = key.replace("mid.attn_1.norm", "mid_blocks.1.group_norm")
                    flag=1
                
                if "up.0.block" in key and 4 in layers:
                    key = key.replace("up.0.block", "up_blocks.3.resnets")
                    flag=1
                if "up.1.block" in key and 3 in layers:
                    key = key.replace("up.1.block", "up_blocks.2.resnets")
                    flag=1
                if "up.2.block" in key and 2 in layers:
                    key = key.replace("up.2.block", "up_blocks.1.resnets")
                    flag=1
                if "up.3.block" in key and 1 in layers:
                    key = key.replace("up.3.block", "up_blocks.0.resnets")
                    flag=1

                if "up.0.upsample.conv" in key and 4 in layers:
                    key = key.replace("up.0.upsample.conv", "up_blocks.3.upsample")
                    flag=1
                if "up.1.upsample.conv" in key and 3 in layers:
                    key = key.replace("up.1.upsample.conv", "up_blocks.2.upsample")
                    flag=1
                if "up.2.upsample.conv" in key and 2 in layers:
                    key = key.replace("up.2.upsample.conv", "up_blocks.1.upsample")
                    flag=1
                if "up.3.upsample.conv" in key and 1 in layers:
                    key = key.replace("up.3.upsample.conv", "up_blocks.0.upsample")
                    flag=1
                if "norm_out" in key and 4 in layers:
                    key = key.replace("norm_out", "conv_norm_out")
                    flag=1
                if 'conv_out' in key and 4 in layers:
                    flag=1

                if "nin_shortcut" in key:
                    key = key.replace("nin_shortcut", "conv_shortcut")
                
                
                if "down.0.block." in key:
                    key = key.replace("down.0.block.", "down_blocks.0.resnets.")                
                if "down.1.block." in key:
                    key = key.replace("down.1.block.", "down_blocks.1.resnets.")
                if "down.2.block." in key:
                    key = key.replace("down.2.block.", "down_blocks.2.resnets.")
                if "down.3.block." in key:
                    key = key.replace("down.3.block.", "down_blocks.3.resnets.")

                if "down.0.downsample.conv" in key:
                    key = key.replace("down.0.downsample.conv", "down_blocks.0.downsample")
                if "down.1.downsample.conv" in key:
                    key = key.replace("down.1.downsample.conv", "down_blocks.1.downsample")
                if "down.2.downsample.conv" in key:
                    key = key.replace("down.2.downsample.conv", "down_blocks.2.downsample")
                
                
                
                if "conv_shortcut.weight" in key:
                    value = value.squeeze()

                if len(value.shape) == 4 and "query_proj" not in key:
                    value = value.transpose(0, 2, 3, 1)
                    value = value.reshape(-1).reshape(value.shape)
                if "query_proj" or "key_proj" or "value_proj" or "out_proj" in key:
                    value = value.squeeze()
                if flag==1:
                    sanitized_weights[key] = value
            if "post_quant_conv" in key or ("quant_conv" in key and not self.decoder_only):
                key = key.replace("quant_conv", "quant_proj")
                if len(value.shape) == 4:
                    value = value.transpose(0, 2, 3, 1)
                    value = value.reshape(-1).reshape(value.shape)
                    
                sanitized_weights[key] = value
        return sanitized_weights

@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


@dataclass
class ModelArgs(AutoencoderConfig):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        if isinstance(self.shard, dict):
            self.shard = Shard(**self.shard)

        if not isinstance(self.shard, Shard):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        if not self.shard.is_first_layer():
            self.vision_config = None
