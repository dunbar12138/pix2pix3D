# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import SynthesisNetwork, FullyConnectedLayer, normalize_2nd_moment, DiscriminatorBlock
from training.triplane import OSGDecoder
from training.volumetric_rendering.renderer import ImportanceRenderer, ImportanceSemanticRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
import dnnlib

from torch_utils import misc

import numpy as np
from einops import repeat, rearrange
import math
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------- #

@persistence.persistent_class
class EqualConv2d(torch.nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        new_scale   = 1.0
        self.weight = torch.nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size) * new_scale
        )
        self.scale   = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride  = stride
        self.padding = padding
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

# ------------------------------------------------------------------------------------------- #

@persistence.persistent_class
class Encoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        bottleneck_factor   = 2,        # By default, the same as discriminator we use 4x4 features
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 1,        # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping
        lowres_head         = None,     # add a low-resolution discriminator head
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        model_kwargs        = {},
        upsample_type       = 'default',
        progressive         = False,
        **unused
    ):
        super().__init__()
        self.img_resolution      = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels        = img_channels
        self.block_resolutions   = [2 ** i for i in range(self.img_resolution_log2, bottleneck_factor, -1)]
        self.architecture        = architecture
        self.lowres_head         = lowres_head
        self.upsample_type       = upsample_type
        self.progressive         = progressive
        self.model_kwargs        = model_kwargs
        self.output_mode         = model_kwargs.get('output_mode', 'styles')
        if self.progressive:
            assert self.architecture == 'skip', "not supporting other types for now."
        self.predict_camera      = model_kwargs.get('predict_camera', False)

        channel_base = int(channel_base * 32768)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)        
        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)    
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels  = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        # this is an encoder
        if self.output_mode in ['W', 'W+', 'None']:
            self.num_ws    = self.model_kwargs.get('num_ws', 0)
            self.n_latents = self.num_ws if self.output_mode == 'W+' else (0 if self.output_mode == 'None' else 1) 
            self.w_dim     = self.model_kwargs.get('w_dim', 512)
            self.add_dim   = self.model_kwargs.get('add_dim', 0) if not self.predict_camera else 9
            self.out_dim   = self.w_dim * self.n_latents + self.add_dim
            assert self.out_dim > 0, 'output dimenstion has to be larger than 0'
            assert self.block_resolutions[-1] // 2 == 4, "make sure the last resolution is 4x4"
            self.projector = EqualConv2d(channels_dict[4], self.out_dim, 4, padding=0, bias=False)
        else:
            raise NotImplementedError
        self.register_buffer("alpha", torch.scalar_tensor(-1))

    def set_alpha(self, alpha):
        if alpha is not None:
            self.alpha.fill_(alpha)
    
    def set_resolution(self, res):
        self.curr_status = res

    def get_block_resolutions(self, input_img):
        block_resolutions = self.block_resolutions
        lowres_head = self.lowres_head
        alpha = self.alpha
        img_res = input_img.size(-1)
        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1):
            if (self.alpha < 1) and (self.alpha > 0): 
                try:
                    n_levels, _, before_res, target_res = self.curr_status
                    alpha, index = math.modf(self.alpha * n_levels)
                    index = int(index)
                except Exception as e:  # TODO: this is a hack, better to save status as buffers.
                    before_res = target_res = img_res 
                if before_res == target_res:
                    # no upsampling was used in generator, do not increase the discriminator
                    alpha = 0    
                block_resolutions = [res for res in self.block_resolutions if res <= target_res]
                lowres_head = before_res
            elif self.alpha == 0:
                block_resolutions = [res for res in self.block_resolutions if res <= lowres_head]
        return block_resolutions, alpha, lowres_head

    def forward(self, inputs, **block_kwargs):
        if isinstance(inputs, dict):
            img = inputs['img']
        else:
            img = inputs

        block_resolutions, alpha, lowres_head = self.get_block_resolutions(img)
        if img.size(-1) > block_resolutions[0]:
            img = downsample(img, block_resolutions[0])

        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
            img0 = downsample(img, img.size(-1) // 2)
           
        x = None if (not self.progressive) or (block_resolutions[0] == self.img_resolution) \
            else getattr(self, f'b{block_resolutions[0]}').fromrgb(img)

        for res in block_resolutions:
            block = getattr(self, f'b{res}')
            if (lowres_head == res) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
                if self.architecture == 'skip':
                    img = img * alpha + img0 * (1 - alpha)
                if self.progressive:
                    x = x * alpha + block.fromrgb(img0) * (1 - alpha)      # combine from img0           
            x, img = block(x, img, **block_kwargs)
        
        outputs = {}
        if self.output_mode in ['W', 'W+', 'None']:
            out = self.projector(x)[:,:,0,0]
            if self.predict_camera:
                out, out_cam_9d = out[:, 9:], out[:, :9]
                outputs['camera'] = camera_9d_to_16d(out_cam_9d)
            
            if self.output_mode == 'W+':
                out = rearrange(out, 'b (n s) -> b n s', n=self.num_ws, s=self.w_dim)
            elif self.output_mode == 'W':
                out = repeat(out, 'b s -> b n s', n=self.num_ws)
            else:
                out = None
            outputs['ws'] = out

        return outputs
    

#----------------------------------------------------------------------------

@persistence.persistent_class
class MaskMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        in_resolution,              # Input resolution.
        in_channels,                # Number of input channels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        one_hot         = True,
        **unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.one_hot = one_hot

        if embed_features is None:
            embed_features = w_dim
        if layer_features is None:
            layer_features = w_dim
        if c_dim == 0:
            features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        else:
            features_list = [z_dim + embed_features * 2] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        self.embed_mask = Encoder(img_resolution=in_resolution, img_channels=in_channels, model_kwargs={'num_ws': 1, 'w_dim': embed_features, 'output_mode': 'W'})
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z=None, c=None, batch=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **unused_kwargs):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))   # normalize z to sphere
            # assert (batch['mask'].squeeze(1).long() >= 0).all()
            if self.one_hot:
                mask_one_hot = torch.nn.functional.one_hot(batch['mask'].squeeze(1).long(), self.in_channels).permute(0,3,1,2)
            else:
                mask_one_hot = batch['mask']
            misc.assert_shape(mask_one_hot, [None, self.in_channels, self.in_resolution, self.in_resolution])
            y = normalize_2nd_moment(self.embed_mask(mask_one_hot.to(torch.float32))['ws'].squeeze(1))
            misc.assert_shape(y, [None, self.w_dim])
            x = torch.cat([x.contiguous(), y.contiguous()], dim=1) if x is not None else y

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                c_embed = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, c_embed], dim=1) if x is not None else c_embed
        
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and update_emas:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MaskMappingNetwork_disentangle(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        in_resolution,              # Input resolution.
        in_channels,                # Number of input channels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        one_hot         = True,
        **unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.one_hot = one_hot

        self.geometry_layer = 7

        if embed_features is None:
            embed_features = w_dim
        if layer_features is None:
            layer_features = w_dim
        if c_dim == 0:
            features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]
        else:
            features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        self.embed_mask = Encoder(img_resolution=in_resolution, img_channels=in_channels, model_kwargs={'num_ws': self.geometry_layer, 'w_dim': w_dim, 'output_mode': 'W+'})
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([num_ws, w_dim]))

    def forward(self, z=None, c=None, batch=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **unused_kwargs):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))   # normalize z to sphere

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                c_embed = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, c_embed], dim=1) if x is not None else c_embed
        
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Geometry Code from Mask
        misc.assert_shape(batch['mask'], [z.shape[0], 1, None, None])
        if self.one_hot:
            mask_one_hot = torch.nn.functional.one_hot(batch['mask'].squeeze(1).long(), self.in_channels).permute(0,3,1,2)
        else:
            mask_one_hot = batch['mask']
        misc.assert_shape(mask_one_hot, [z.shape[0], self.in_channels, self.in_resolution, self.in_resolution])
        y = self.embed_mask(mask_one_hot.to(torch.float32))['ws'] # B x 7 x w_dim
        misc.assert_shape(y, [None, self.geometry_layer, self.w_dim])

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws - self.geometry_layer, 1])
                x = torch.cat([y, x], dim=1)

        # Update moving average of W.
        if self.w_avg_beta is not None and update_emas:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class EdgeMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        in_resolution,              # Input resolution.
        in_channels,                # Number of input channels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        **unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if layer_features is None:
            layer_features = w_dim
        if c_dim == 0:
            features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        else:
            features_list = [z_dim + embed_features * 2] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        self.embed_edge = Encoder(img_resolution=in_resolution, img_channels=in_channels, model_kwargs={'num_ws': 1, 'w_dim': embed_features, 'output_mode': 'W'})
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z=None, c=None, batch=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **unused_kwargs):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))   # normalize z to shpere
            # mask_one_hot = torch.nn.functional.one_hot(batch['mask'].squeeze(1).long(), self.in_channels).permute(0,3,1,2)
            edge = batch['mask'].to(torch.float32)
            misc.assert_shape(edge, [None, self.in_channels, self.in_resolution, self.in_resolution])
            y = normalize_2nd_moment(self.embed_edge(edge)['ws'].squeeze(1))
            misc.assert_shape(y, [None, self.w_dim])
            x = torch.cat([x.contiguous(), y.contiguous()], dim=1) if x is not None else y

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                c_embed = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, c_embed], dim=1) if x is not None else c_embed
        
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and update_emas:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------


@persistence.persistent_class
class EdgeMappingNetwork_disentangle(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        in_resolution,              # Input resolution.
        in_channels,                # Number of input channels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        **unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        self.geometry_layer = 7

        if embed_features is None:
            embed_features = w_dim
        if layer_features is None:
            layer_features = w_dim
        if c_dim == 0:
            features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]
        else:
            features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        self.embed_mask = Encoder(img_resolution=in_resolution, img_channels=in_channels, model_kwargs={'num_ws': self.geometry_layer, 'w_dim': w_dim, 'output_mode': 'W+'})
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([num_ws, w_dim]))

    def forward(self, z=None, c=None, batch=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **unused_kwargs):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))   # normalize z to sphere

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                c_embed = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, c_embed], dim=1) if x is not None else c_embed
        
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Geometry Code from Mask
        misc.assert_shape(batch['mask'], [z.shape[0], 1, None, None])
        edge = batch['mask'].to(torch.float32)
        misc.assert_shape(edge, [z.shape[0], self.in_channels, self.in_resolution, self.in_resolution])
        y = self.embed_mask(edge.to(torch.float32))['ws'] # B x 7 x w_dim
        misc.assert_shape(y, [None, self.geometry_layer, self.w_dim])

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws - self.geometry_layer, 1])
                x = torch.cat([y, x], dim=1)

        # Update moving average of W.
        if self.w_avg_beta is not None and update_emas:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

# ----------------------------------------------------------------------------

@persistence.persistent_class
class Generator_cond(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        # self.mapping = MaskMappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.mapping = dnnlib.util.construct_class_by_name(**mapping_kwargs, z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = Generator_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    def sample(self, coordinates, directions, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)



#----------------------------------------------------------------------------


@persistence.persistent_class
class TriPlaneSemanticGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        semantic_channels,          # Number of semantic channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        data_type = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.semantic_channels=semantic_channels
        self.data_type = data_type
        self.renderer = ImportanceSemanticRenderer()
        self.ray_sampler = RaySampler()

        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # mapping_semantic_kwargs = mapping_kwargs.copy()
        # mapping_semantic_kwargs['z_dim'] = 0
        self.backbone_semantic = Generator_cond(0, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_semantic = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module_semantic'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], semantic_channels=semantic_channels, **sr_kwargs)

        self.decoder = OSGDecoder(64, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True})
        self.decoder_semantic = OSGDecoder_semantic(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True if semantic_channels == 1 else False})

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        ws_texture = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_semantic = self.backbone_semantic.mapping(None, c * self.rendering_kwargs.get('c_scale', 0), batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return torch.cat([ws_texture, ws_semantic], dim=-1)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        # if use_cached_backbone and self._last_planes is not None:
        #     planes = self._last_planes
        # else:
        #     planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        # if cache_backbone:
        #     self._last_planes = planes
        N, M, _ = ray_origins.shape
        assert ws.shape[-1] == self.w_dim * 2
        ws_texture, ws_semantic = ws[..., :self.w_dim], ws[..., self.w_dim:]
        planes_texture = self.backbone.synthesis(ws_texture, update_emas=update_emas, **synthesis_kwargs)
        planes_semantic = self.backbone_semantic.synthesis(ws_semantic, update_emas=update_emas, **synthesis_kwargs)


        # Reshape output into three 32-channel planes
        planes_texture = planes_texture.view(len(planes_texture), 3, 32, planes_texture.shape[-2], planes_texture.shape[-1])
        planes_semantic = planes_semantic.view(len(planes_semantic), 3, 32, planes_semantic.shape[-2], planes_semantic.shape[-1])

        # Perform volume rendering                      
        feature_samples, depth_samples, weights_samples = self.renderer(planes_texture, planes_semantic, self.decoder, self.decoder_semantic, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_feature_image, semantics_feature_image = feature_image[:, :feature_image.shape[1] // 2], feature_image[:, feature_image.shape[1] // 2:]

        # Run superresolution to get final image
        rgb_image = rgb_feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, rgb_feature_image, ws_texture, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        semantic_image = semantics_feature_image[:, :self.semantic_channels]
        sr_semantic_image = self.superresolution_semantic(semantic_image, semantics_feature_image, ws_semantic, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'semantic': sr_semantic_image, 'semantic_raw': semantic_image}
    
    def sample(self, coordinates, directions, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        assert ws.shape[-1] == self.w_dim * 2
        ws_texture, ws_semantic = ws[..., :self.w_dim], ws[..., self.w_dim:]
        planes_texture = self.backbone.synthesis(ws_texture, update_emas=update_emas, **synthesis_kwargs)
        planes_semantic = self.backbone_semantic.synthesis(ws_semantic, update_emas=update_emas, **synthesis_kwargs)


        # Reshape output into three 32-channel planes
        planes_texture = planes_texture.view(len(planes_texture), 3, 32, planes_texture.shape[-2], planes_texture.shape[-1])
        planes_semantic = planes_semantic.view(len(planes_semantic), 3, 32, planes_semantic.shape[-2], planes_semantic.shape[-1])

        return self.renderer.run_model(planes_texture, planes_semantic, self.decoder, self.decoder_semantic, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        assert ws.shape[-1] == self.w_dim * 2
        ws_texture, ws_semantic = ws[..., :self.w_dim], ws[..., self.w_dim:]
        planes_texture = self.backbone.synthesis(ws_texture, update_emas=update_emas, **synthesis_kwargs)
        planes_semantic = self.backbone_semantic.synthesis(ws_semantic, update_emas=update_emas, **synthesis_kwargs)


        # Reshape output into three 32-channel planes
        planes_texture = planes_texture.view(len(planes_texture), 3, 32, planes_texture.shape[-2], planes_texture.shape[-1])
        planes_semantic = planes_semantic.view(len(planes_semantic), 3, 32, planes_semantic.shape[-2], planes_semantic.shape[-1])

        return self.renderer.run_model(planes_texture, planes_semantic, self.decoder, self.decoder_semantic, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


# ----------------------------------------------------------------------------

class OSGDecoder_semantic(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.final_sigmoid = options['sigmoid']
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if self.final_sigmoid:
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        else:
            rgb = x[..., 1:]
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


#----------------------------------------------------------------------------
class OSGDecoder_semantic_entangle(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.feature_sigmoid = options['sigmoid']
        self.semantic_channels = options['semantic_channels']
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if self.feature_sigmoid:
            feature = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        else:
            rgb = torch.sigmoid(x[..., 1:4])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
            semantic = x[..., 4:4+self.semantic_channels]
            feature = torch.sigmoid(x[..., 4+self.semantic_channels:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
            feature = torch.cat((rgb, semantic, feature), dim=-1)
        # feature = x[..., 1:]
        sigma = x[..., 0:1]
        return {'rgb': feature, 'sigma': sigma}

class OSGDecoder_semantic_lateSeparate(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.net_semantic = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

        self.semantic_sigmoid = options['sigmoid']
        # self.semantic_channels = options['semantic_channels']
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        rgb = self.net(x)
        semantic = self.net_semantic(x)
        rgb = rgb.view(N, M, -1)
        semantic = semantic.view(N, M, -1)
        sigma = semantic[..., 0:1]

        rgb = torch.sigmoid(rgb[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        if self.semantic_sigmoid:
            semantic = torch.sigmoid(semantic[..., 1:])*(1 + 2*0.001) - 0.001
        else:
            semantic = semantic[..., 1:]

        feature = torch.cat((rgb, semantic), dim=-1)
    
        # feature = x[..., 1:]
        
        return {'rgb': feature, 'sigma': sigma}


#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneSemanticEntangleGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        semantic_channels,          # Number of semantic channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        data_type = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.semantic_channels=semantic_channels
        self.data_type = data_type
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        self.backbone = Generator_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_semantic = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module_semantic'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], semantic_channels=semantic_channels, **sr_kwargs)

        self.decoder = OSGDecoder_semantic_lateSeparate(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True if semantic_channels == 1 else False, 'semantic_channels': semantic_channels})
        # self.decoder_semantic = OSGDecoder_semantic(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True if semantic_channels == 1 else False})

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering                      
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_feature_image, semantics_feature_image = feature_image[:, :feature_image.shape[1] // 2], feature_image[:, feature_image.shape[1] // 2:]

        # Run superresolution to get final image
        rgb_image = rgb_feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, rgb_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        semantic_image = semantics_feature_image[:, :self.semantic_channels]
        sr_semantic_image = self.superresolution_semantic(semantic_image, semantics_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'semantic': sr_semantic_image, 'semantic_raw': semantic_image}
    
    def sample(self, coordinates, directions, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


# ----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneSemanticEntangleGenerator_withBG(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        semantic_channels,          # Number of semantic channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        data_type = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.semantic_channels=semantic_channels
        self.data_type = data_type
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        self.backbone = Generator_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        mapping_bg_kwargs = mapping_kwargs.copy()
        mapping_bg_kwargs['class_name'] = None
        self.backbone_bg = StyleGAN2Backbone(z_dim, 0, w_dim, img_resolution=256, img_channels=32*2, mapping_kwargs=mapping_bg_kwargs, **synthesis_kwargs)

        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution_semantic = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module_semantic'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], semantic_channels=semantic_channels, **sr_kwargs)

        self.decoder = OSGDecoder_semantic_lateSeparate(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True if semantic_channels == 1 else False, 'semantic_channels': semantic_channels})
        # self.decoder_semantic = OSGDecoder_semantic(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigmoid': True if semantic_channels == 1 else False})

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering                      
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Create background plane
        ws_bg = ws[:,-1,:].unsqueeze(1).repeat([1, ws.shape[1], 1])
        planes_bg = self.backbone_bg.synthesis(ws_bg, update_emas=update_emas, **synthesis_kwargs)
        planes_bg = planes_bg.view(len(planes_bg), 64, planes_bg.shape[-2], planes_bg.shape[-1])

        # Combine foreground and background
        feature_samples, depth_samples = self.combine_fg_bg(feature_samples, depth_samples, weights_samples, planes_bg, ray_origins, ray_directions, self.rendering_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weight_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_feature_image, semantics_feature_image = feature_image[:, :feature_image.shape[1] // 2], feature_image[:, feature_image.shape[1] // 2:]

        # Run superresolution to get final image
        rgb_image = rgb_feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, rgb_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        semantic_image = semantics_feature_image[:, :self.semantic_channels]
        sr_semantic_image = self.superresolution_semantic(semantic_image, semantics_feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'semantic': sr_semantic_image, 'semantic_raw': semantic_image, 'weight': weight_image}
    
    def sample(self, coordinates, directions, z, c, batch, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, batch, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, batch['pose'], batch, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    def combine_fg_bg(self, feature_samples, depth_samples, weights_samples, planes_bg, ray_origins, ray_directions, rendering_kwargs):
        # Combine foreground and background
        # feature_samples: [N, M, 64]
        # depth_samples: [N, M, 1]
        # weights_samples: [N, M, 1]
        # planes_bg: [N, 64, H, W]
        # ray_origins: [N, M, 3]
        # ray_directions: [N, M, 3]
        # rendering_kwargs: dict

        # Convert ray directions to spherical coordinates
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
        theta = torch.atan2(ray_directions[:, :, 1], ray_directions[:, :, 0])
        phi = torch.acos(ray_directions[:, :, 2])

        # Convert spherical coordinates to pixel coordinates (-1 to 1)
        x = theta * 2 / np.pi
        y = phi * 2 / np.pi - 1

        # Sample background planes
        feature_samples_bg = F.grid_sample(planes_bg, torch.stack([x, y], dim=-1).unsqueeze(1), mode='bilinear', padding_mode='border') # [N, 64, 1, M]
        feature_samples_bg = feature_samples_bg.squeeze(2).permute(0, 2, 1) # [N, M, 64]
        assert feature_samples_bg.shape == feature_samples.shape

        
        # Sigmoid the rgb features and bound the semantic features
        feature_samples_bg = torch.sigmoid(feature_samples_bg)*(1 + 2*0.001) - 0.001
        feature_samples_bg = feature_samples_bg * 2 - 1 # [-1, 1]

        feature_samples_bg[:,:,32:] = feature_samples_bg[:,:,32:] * 10 # [-10, 10]

        if self.semantic_channels > 1:
            # Hardcode the background semantic class to 0
            feature_samples_bg[:, :, 32+1:32+self.semantic_channels] = 0
            feature_samples_bg[:, :, 32] = 20

            # feature_samples[:, :, 32] = 0

        
        # Combine foreground and background
        feature_samples = feature_samples + feature_samples_bg * (1 - weights_samples)
        depth_samples_bg = torch.ones_like(depth_samples) * rendering_kwargs['ray_end']
        depth_samples = depth_samples + depth_samples_bg * (1 - weights_samples)

        return feature_samples, depth_samples



# ----------------------------------------------------------------------------