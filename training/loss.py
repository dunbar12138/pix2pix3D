# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

import lpips

import torch.nn.functional as F

from training.loss_utils import cross_entropy2d

# ----------------------------------------------------------------------------


class Loss:
    # to be overridden by subclass
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        raise NotImplementedError()

# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter(
            [1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand(
                (c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64,
                                     device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand(
                    [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(
                    torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(
            ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(
                augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * \
            self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3),
                    1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * \
            self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(
                cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (
                1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(
            real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(
                        torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * \
                self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand(
                (ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + \
                torch.tensor([0, 0, -1], device=ws.device) * (1/256) * \
                self.G.rendering_kwargs['box_warp']  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(
                sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(
                        torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * (1/256) * \
                self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand(
                (ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + \
                torch.tensor([0, 0, -1], device=ws.device) * (1/256) * \
                self.G.rendering_kwargs['box_warp']  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(
                sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(
                        torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * (1/256) * \
                self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(
                    gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image,
                                'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(
                        'Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum(
                            [1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------


class Pix2Pix3DLoss(Loss):
    def __init__(self, device, G, D, D_semantic=None, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 random_c_prob=0, lambda_l1=2, lambda_lpips=10, lambda_D_semantic=1, seg_weight=0, edge_weight=2, only_raw_recons=False, silhouette_loss=False,
                 lambda_cross_view=0):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.D_semantic = D_semantic
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter(
            [1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        self.random_c_prob = random_c_prob

        self.lpips_loss = lpips.LPIPS(net='vgg').to(device=device)

        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_D_semantic = lambda_D_semantic

        if int(seg_weight) == 1:
            self.seg_weight = torch.tensor([0.42768099,  0.45614868,  1.59952169,  4.38863045,  4.85695198,
                                            4.86439145,  3.53563349,  3.57896961,  3.37838867,  3.66981824,
                                            4.17743386,  3.5624441,  2.78190484,  0.40917425,  2.38560636,
                                            4.65813434, 17.17367367,  1.13303585,  1.25281865]).to(self.device)
        elif int(seg_weight) == 2:
            print('Using seg weight 2')
            self.seg_weight = torch.tensor([1.82911031e-01, 2.08071618e-01, 2.55846962e+00, 1.92600773e+01,
                                            2.35899825e+01, 2.36623042e+01, 1.25007042e+01, 1.28090235e+01,
                                            1.14135100e+01, 1.34675659e+01, 1.74509537e+01, 1.26910080e+01,
                                            7.73899453e+00, 1.67423571e-01, 5.69111768e+00, 2.16982155e+01,
                                            2.94935067e+02, 1.28377023e+00, 1.56955458e+00]).to(self.device)
        else:
            self.seg_weight = None

        self.edge_weight = edge_weight
        self.only_raw_recons = only_raw_recons
        self.silhouette_loss = silhouette_loss
        self.lambda_cross_view = lambda_cross_view

    def run_G(self, z, c, batch, swapping_prob, neural_rendering_resolution, update_emas=False, mode='random_z_image_c'):
        # if swapping_prob is not None:
        #     c_swapped = torch.roll(c.clone(), 1, 0)
        #     c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        # else:
        #     c_gen_conditioning = torch.zeros_like(c)
        if mode == 'random_z_image_c':
            ws = self.G.mapping(
                z, batch['pose'], batch, update_emas=update_emas)
            gen_output = self.G.synthesis(
                ws, batch['pose'], neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        elif mode == 'random_z_random_c':
            ws = self.G.mapping(
                z, batch['pose'], batch, update_emas=update_emas)
            gen_output = self.G.synthesis(
                ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        # if self.style_mixing_prob > 0:
        #     with torch.autograd.profiler.record_function('style_mixing'):
        #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        input_img = {'image': img['image'].clone(
        ), 'image_raw': img['image_raw'].clone()}
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=input_img['image'].device).div(
                    blur_sigma).square().neg().exp2()
                input_img['image'] = upfirdn2d.filter2d(
                    input_img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([input_img['image'],
                                                          torch.nn.functional.interpolate(input_img['image_raw'], size=input_img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                         dim=1))
            input_img['image'] = augmented_pair[:,
                                                :input_img['image'].shape[1]]
            input_img['image_raw'] = torch.nn.functional.interpolate(
                augmented_pair[:, input_img['image'].shape[1]:], size=input_img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(input_img, c, update_emas=update_emas)
        return logits

    def run_D_semantic(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        # mask = batch['mask']
        # mask = torch.nn.functional.one_hot(mask, num_classes=self.G.semantic_channels).permute(0, 3, 1, 2).float()
        # mask_raw = torch.nn.functional.interpolate(mask, size=img['image_raw'].shape[2:], mode='nearest')

        # img['image'] = torch.cat([img['image'], mask], dim=1)
        # img['image_raw'] = torch.cat([img['image_raw'], mask_raw], dim=1)
        input_img = {'image': img['image'].clone(
        ), 'image_raw': img['image_raw'].clone()}
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=input_img['image'].device).div(
                    blur_sigma).square().neg().exp2()
                input_img['image'] = upfirdn2d.filter2d(
                    input_img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([input_img['image'],
                                                          torch.nn.functional.interpolate(input_img['image_raw'], size=input_img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                         dim=1))
            input_img['image'] = augmented_pair[:,
                                                :input_img['image'].shape[1]]
            input_img['image_raw'] = torch.nn.functional.interpolate(
                augmented_pair[:, input_img['image'].shape[1]:], size=input_img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D_semantic(input_img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, batch, gen_z, gen_c, gain, cur_nimg, debug=False):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg',
                         'Dboth', 'D_semanticmain', 'D_semanticreg', 'D_semanticboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * \
            self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3),
                    1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * \
            self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if torch.rand(1) < self.random_c_prob:
            generator_mode = 'random_z_random_c'
            c_render = gen_c
        else:
            generator_mode = 'random_z_image_c'
            c_render = batch['pose']

        if self.neural_rendering_resolution_final is not None:
            alpha = min(
                cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (
                1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img, real_c = batch['image'], batch['pose']
        real_img_raw = filtered_resizing(
            real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        real_mask = batch['mask']
        # real_mask_raw = F.interpolate(real_mask, size=neural_rendering_resolution, mode='nearest')
        # real_mask = {'mask': real_mask, 'mask_raw': real_mask_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, mode=generator_mode)
                gen_logits = self.run_D(
                    gen_img, c_render, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                if self.D_semantic is not None:
                    input_img = {}
                    if self.G.data_type == 'seg':
                        mask_softmax = torch.nn.functional.softmax(
                            gen_img['semantic'], dim=1)
                        mask_softmax_raw = torch.nn.functional.softmax(
                            gen_img['semantic_raw'], dim=1)
                        # Detach to avoid backpropagating through the rgb branch.
                        input_img['image'] = torch.cat(
                            [gen_img['image'].detach(), mask_softmax], dim=1)
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'].detach(), mask_softmax_raw], dim=1)
                    else:
                        # Detach to avoid backpropagating through the rgb branch.
                        input_img['image'] = torch.cat(
                            [gen_img['image'].detach(), gen_img['semantic']], dim=1)
                        input_img['image_raw'] = torch.cat(
                            [gen_img['image_raw'].detach(), gen_img['semantic_raw']], dim=1)
                    gen_logits_semantic = self.run_D_semantic(
                        input_img, c_render, blur_sigma=blur_sigma)
                    training_stats.report(
                        'Loss/scores/fake_semantic', gen_logits_semantic)
                    training_stats.report(
                        'Loss/signs/fake_semantic', gen_logits_semantic.sign())
                    loss_Gmain += torch.nn.functional.softplus(
                        -gen_logits_semantic) * self.lambda_D_semantic

                if generator_mode == 'random_z_image_c':
                    loss_G_img_reconstruction = F.smooth_l1_loss(gen_img['image'], real_img['image']) * self.lambda_l1 \
                        + self.lpips_loss(gen_img['image'], real_img['image']) * self.lambda_lpips
                    loss_G_img_reconstruction_raw = F.smooth_l1_loss(gen_img['image_raw'], real_img['image_raw']) * self.lambda_l1 \
                        + self.lpips_loss(gen_img['image_raw'], real_img['image_raw']) * self.lambda_lpips
                    loss_G_img_reconstruction = loss_G_img_reconstruction * \
                        (1 - float(self.only_raw_recons)) + \
                        loss_G_img_reconstruction_raw
                    training_stats.report(
                        'Loss/G/loss_img_reconstruction', loss_G_img_reconstruction)
                    # print(loss_G_img_reconstruction.shape, loss_Gmain.shape)
                    loss_Gmain = loss_Gmain + \
                        loss_G_img_reconstruction.squeeze(-1).squeeze(-1)

                    if 'semantic' in gen_img:
                        if self.G.data_type == 'seg':
                            loss_G_semantic_reconstruction = cross_entropy2d(gen_img['semantic'], real_mask.squeeze(
                                1).long(), weight=self.seg_weight) * (1 - float(self.only_raw_recons))
                            real_mask_raw = F.interpolate(
                                real_mask, size=neural_rendering_resolution, mode='nearest')
                            loss_G_semantic_reconstruction_raw = cross_entropy2d(
                                gen_img['semantic_raw'], real_mask_raw.squeeze(1).long(), weight=self.seg_weight)
                            loss_G_semantic_reconstruction = loss_G_semantic_reconstruction + \
                                loss_G_semantic_reconstruction_raw
                        else:
                            real_mask = batch['mask']
                            # real_mask_raw = filtered_resizing(real_mask, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
                            real_mask_raw = F.interpolate(
                                real_mask, size=neural_rendering_resolution, mode='nearest')
                            loss_G_semantic_reconstruction = F.smooth_l1_loss(gen_img['semantic'], real_mask) * self.edge_weight * (
                                1 - float(self.only_raw_recons)) + F.smooth_l1_loss(gen_img['semantic_raw'], real_mask_raw) * self.edge_weight
                        training_stats.report(
                            'Loss/G/loss_semantic_reconstruction', loss_G_semantic_reconstruction)
                        loss_Gmain = loss_Gmain + \
                            loss_G_semantic_reconstruction.squeeze(
                                -1).squeeze(-1)

                        loss_G_silhouette = 0
                        if self.silhouette_loss and self.G.data_type == 'seg':
                            real_mask_raw = F.interpolate(
                                real_mask, size=neural_rendering_resolution, mode='nearest')
                            loss_G_silhouette = self.calculate_silhouette_loss(
                                gen_img['weight'], real_mask_raw.long())
                            loss_Gmain = loss_Gmain + loss_G_silhouette
                        training_stats.report(
                            'Loss/G/loss_silhouette', loss_G_silhouette)

                    else:
                        training_stats.report(
                            'Loss/G/loss_semantic_reconstruction', 0)
                        training_stats.report('Loss/G/loss_silhouette', 0)

                else:
                    training_stats.report('Loss/G/loss_img_reconstruction', 0)
                    training_stats.report(
                        'Loss/G/loss_semantic_reconstruction', 0)
                    training_stats.report('Loss/G/loss_silhouette', 0)

                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            # Cross-view consistency loss
            with torch.no_grad():
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, mode='random_z_random_c')
            batch_proj = batch.copy()
            batch_proj['mask'] = torch.argmax(
                gen_img['semantic'].detach(), dim=1, keepdim=True)
            gen_img_proj, gen_ws_proj = self.run_G(gen_z, gen_c, batch_proj, swapping_prob=swapping_prob,
                                                   neural_rendering_resolution=neural_rendering_resolution, mode='random_z_image_c')
            with torch.no_grad():
                gen_img_recon, gen_ws_recon = self.run_G(
                    gen_z, gen_c, batch, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, mode='random_z_image_c')

            loss_cross_view = F.smooth_l1_loss(
                gen_img_proj['semantic_raw'], gen_img_recon['semantic_raw']) * self.lambda_cross_view

            training_stats.report('Loss/G/loss_cross_view', loss_cross_view)
            loss_cross_view.mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, batch['pose'], batch, update_emas=False)

            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * \
                self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, batch['pose'], batch, update_emas=False)

            initial_coordinates = torch.rand(
                (ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + \
                torch.tensor([0, 0, -1], device=ws.device) * (1/256) * \
                self.G.rendering_kwargs['box_warp']  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(
                sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, batch['pose'], batch, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(
                        torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * (1/256) * \
                self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, batch['pose'], batch, update_emas=False)

            initial_coordinates = torch.rand(
                (ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + \
                torch.tensor([0, 0, -1], device=ws.device) * (1/256) * \
                self.G.rendering_kwargs['box_warp']  # Behind
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(
                sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand(
                    [], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, batch['pose'], batch, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty(
                        [], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand(
                        [], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(
                        torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand(
                (ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + \
                torch.randn_like(initial_coordinates) * (1/256) * \
                self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat(
                [initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(
                all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(
                sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            if torch.rand(1) < self.random_c_prob:
                generator_mode = 'random_z_random_c'
                c_render = gen_c
            else:
                generator_mode = 'random_z_image_c'
                c_render = batch['pose']
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, update_emas=True, mode=generator_mode)
                gen_logits = self.run_D(
                    gen_img, c_render, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach(
                ).requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image,
                                'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(
                        'Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum(
                            [1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[
                                                           real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # D_semanticmain: Minimize logits for generated images and masks.
        loss_Dgen_semantic = 0
        if phase in ['D_semanticmain', 'D_semanticboth']:
            if torch.rand(1) < self.random_c_prob:
                generator_mode = 'random_z_random_c'
                c_render = gen_c
            else:
                generator_mode = 'random_z_image_c'
                c_render = batch['pose']
            with torch.autograd.profiler.record_function('Dgen_semantic_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, update_emas=True, mode=generator_mode)
                input_img = {}
                if self.G.data_type == 'seg':
                    mask_softmax = torch.nn.functional.softmax(
                        gen_img['semantic'], dim=1)
                    mask_softmax_raw = torch.nn.functional.softmax(
                        gen_img['semantic_raw'], dim=1)
                    input_img['image'] = torch.cat(
                        [gen_img['image'], mask_softmax], dim=1)
                    input_img['image_raw'] = torch.cat(
                        [gen_img['image_raw'], mask_softmax_raw], dim=1)
                else:
                    input_img['image'] = torch.cat(
                        [gen_img['image'], gen_img['semantic']], dim=1)
                    input_img['image_raw'] = torch.cat(
                        [gen_img['image_raw'], gen_img['semantic_raw']], dim=1)
                gen_logits_semantic = self.run_D_semantic(
                    input_img, c_render, blur_sigma=blur_sigma)

                training_stats.report(
                    'Loss/scores/fake_semantic', gen_logits_semantic)
                training_stats.report(
                    'Loss/signs/fake_semantic', gen_logits_semantic.sign())
                loss_Dgen_semantic = torch.nn.functional.softplus(
                    gen_logits_semantic)
            with torch.autograd.profiler.record_function('Dgen_semantic_backward'):
                loss_Dgen_semantic.mean().mul(gain).backward()

        # D_semanticmain: Maximize logits for real images and masks.
        # Dr1: Apply R1 regularization.
        if phase in ['D_semanticmain', 'D_semanticreg', 'D_semanticboth']:
            name = 'Dreal_semantic' if phase == 'D_semanticmain' else 'Dr1_semantic' if phase == 'D_semanticreg' else 'Dreal_Dr1_semantic'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(
                    phase in ['D_semanticreg', 'D_semanticboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(
                    phase in ['D_semanticreg', 'D_semanticboth'])

                if self.G.data_type == 'seg':
                    real_mask = torch.nn.functional.one_hot(batch['mask'].squeeze(
                        1).long(), num_classes=self.G.semantic_channels).permute(0, 3, 1, 2).float()
                    real_mask_raw = filtered_resizing(
                        real_mask, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

                    real_mask_tmp_image = real_mask.detach().requires_grad_(
                        phase in ['D_semanticreg', 'D_semanticboth'])
                    real_mask_tmp_image_raw = real_mask_raw.detach().requires_grad_(
                        phase in ['D_semanticreg', 'D_semanticboth'])
                else:
                    real_mask = batch['mask']
                    real_mask_raw = filtered_resizing(
                        real_mask, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

                    real_mask_tmp_image = real_mask.detach().requires_grad_(
                        phase in ['D_semanticreg', 'D_semanticboth'])
                    real_mask_tmp_image_raw = real_mask_raw.detach().requires_grad_(
                        phase in ['D_semanticreg', 'D_semanticboth'])

                real_img_tmp = {'image': torch.cat([real_img_tmp_image, real_mask_tmp_image], dim=1), 'image_raw': torch.cat([
                    real_img_tmp_image_raw, real_mask_tmp_image_raw], dim=1)}

                real_logits_semantic = self.run_D_semantic(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report(
                    'Loss/scores/real_semantic', real_logits_semantic)
                training_stats.report(
                    'Loss/signs/real_semantic', real_logits_semantic.sign())

                loss_Dreal_semantic = 0
                if phase in ['D_semanticmain', 'D_semanticboth']:
                    loss_Dreal_semantic = torch.nn.functional.softplus(
                        -real_logits_semantic)
                    training_stats.report(
                        'Loss/D/loss_semantic', loss_Dgen_semantic + loss_Dreal_semantic)

                loss_Dr1_semantic = 0
                if phase in ['D_semanticreg', 'D_semanticboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads_semantic'), conv2d_gradfix.no_weight_gradients():
                            r1_grads_semantic = torch.autograd.grad(outputs=[real_logits_semantic.sum()], inputs=[
                                                                    real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image_semantic = r1_grads_semantic[0]
                            r1_grads_image_raw_semantic = r1_grads_semantic[1]
                        r1_penalty_semantic = r1_grads_image_semantic.square().sum(
                            [1, 2, 3]) + r1_grads_image_raw_semantic.square().sum([1, 2, 3])
                    else:
                        with torch.autograd.profiler.record_function('r1_grads_semantic'):
                            r1_grads_semantic = torch.autograd.grad(outputs=[real_logits_semantic.sum(
                            )], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image_semantic = r1_grads_semantic[0]
                        r1_penalty_semantic = r1_grads_image_semantic.square().sum([
                            1, 2, 3])
                    loss_Dr1_semantic = r1_penalty_semantic * r1_gamma * 0.5
                    training_stats.report(
                        'Loss/r1_penalty_semantic', r1_penalty_semantic)
                    training_stats.report(
                        'Loss/D/reg_semantic', loss_Dr1_semantic)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal_semantic + loss_Dr1_semantic).mean().mul(gain).backward()

    def calculate_silhouette_loss(self, weight_image, mask):
        # weight_image: [B, 1, H, W]
        # mask: [B, 1, H, W]

        # assert weight_image.shape == mask.shape
        # bg_mask = (mask == 0)
        # fg_mask = (mask > 0)
        # bg_weight = weight_image[bg_mask]
        # fg_weight = weight_image[fg_mask]
        # bg_loss = bg_weight.mean()
        # fg_loss = 1 - fg_weight.mean()

        # return bg_loss + fg_loss

        assert weight_image.shape == mask.shape
        ref_silhouette = (mask > 0).float()
        loss = (weight_image - ref_silhouette).pow(2).mean() * 10
        return loss

# ----------------------------------------------------------------------------
