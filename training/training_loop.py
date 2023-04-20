# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section

from training.utils import color_mask

# #----------------------------------------------------------------------------

# def setup_snapshot_image_grid(training_set, random_seed=0):
#     rnd = np.random.RandomState(random_seed)
#     gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
#     gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

#     # No labels => show random subset of training samples.
#     if not training_set.has_labels:
#         all_indices = list(range(len(training_set)))
#         rnd.shuffle(all_indices)
#         grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

#     else:
#         # Group training samples by label.
#         label_groups = dict() # label => [idx, ...]
#         for idx in range(len(training_set)):
#             label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
#             if label not in label_groups:
#                 label_groups[label] = []
#             label_groups[label].append(idx)

#         # Reorder.
#         label_order = list(label_groups.keys())
#         rnd.shuffle(label_order)
#         for label in label_order:
#             rnd.shuffle(label_groups[label])

#         # Organize into grid.
#         grid_indices = []
#         for y in range(gh):
#             label = label_order[y % len(label_order)]
#             indices = label_groups[label]
#             grid_indices += [indices[x % len(indices)] for x in range(gw)]
#             label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

#     # Load data.
#     images, labels = zip(*[training_set[i] for i in grid_indices])
#     return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, test_set=None, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    # gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    # gh = np.clip(4320 // training_set.image_shape[1], 4, 32)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    # assert not training_set.has_labels
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images = torch.stack([torch.Tensor(training_set[i]['image']) for i in grid_indices])
    masks = torch.stack([torch.Tensor(training_set[i]['mask']) for i in grid_indices])
    poses = torch.stack([torch.Tensor(training_set[i]['pose']) for i in grid_indices])
    if test_set is None:
        return (gw, gh), images, masks, poses
    else:
        all_indices = list(range(len(test_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
        images_test = torch.stack([torch.Tensor(test_set[i]['image']) for i in grid_indices])
        masks_test = torch.stack([torch.Tensor(test_set[i]['mask']) for i in grid_indices])
        poses_test = torch.stack([test_set[i]['pose'] for i in grid_indices])
        images = torch.cat([images, images_test], dim=0)
        masks = torch.cat([masks, masks_test], dim=0)
        poses = torch.cat([poses, poses_test], dim=0)
        return (gw, gh*2), images, masks, poses

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def get_image_grid(img, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        return PIL.Image.fromarray(img[:, :, 0], 'L')
    if C == 3:
        return PIL.Image.fromarray(img, 'RGB')

#----------------------------------------------------------------------------

def log_table(G_ema, grid_z, grid_i, grid_c, grid_m, grid_p, mask_type, global_step, device, wandb):
    max_rounds = 16
    
    images_all = []
    images = []

    segs_all = []
    segs = []
    for round_idx in range(max_rounds):
        out = G_ema(grid_z[round_idx].to(device), grid_c[round_idx].to(device), 
                {'image':grid_i[round_idx].to(device), 'pose':grid_p[round_idx].to(device), 'mask':grid_m[round_idx].to(device)}, noise_mode='const')
        images.append(out['image'])
        if 'semantic' in out:
            if mask_type == 'seg':
                segs.append(torch.argmax(out['semantic'], dim=1)) # B x 32 x 32
            else:
                segs.append(out['semantic'])
    images_all.append(torch.cat(images))
    if len(segs) > 0:
        segs_all.append(torch.cat(segs))

    yaw_range = 0.35
    for yaw in [-1, -0.5, 0, 0.5, 1]:
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw * yaw_range, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        pose = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        images = []
        segs = []
        for round_idx in range(max_rounds):
            out = G_ema(grid_z[round_idx].to(device), pose.expand(grid_z[round_idx].shape[0], -1), 
                {'image':grid_i[round_idx].to(device), 'pose':grid_p[round_idx].to(device), 'mask':grid_m[round_idx].to(device)}, noise_mode='const')
            images.append(out['image'])
            if 'semantic' in out:
                if mask_type == 'seg':
                    segs.append(torch.argmax(out['semantic'], dim=1)) # B x 32 x 32
                else:
                    segs.append(out['semantic'])
        images_all.append(torch.cat(images))
        if len(segs) > 0:
            segs_all.append(torch.cat(segs))

    images_all = torch.stack(images_all) # 6 x N x 3 x 512 x 512
    grid_i_all = torch.cat(grid_i)
    grid_m_all = torch.cat(grid_m)

    if len(segs_all) > 0:
        segs_all = torch.stack(segs_all) # 6 x N x 512 x 512

    if len(segs_all) > 0:
        columns = ["Real Image", "Mask", "Generated_ema", "Generated Mask"]
    else:
        columns = ["Real Image", "Mask", "Generated_ema"]

    table = wandb.Table(columns=columns)

    for row in range(images_all.shape[1]):
        g_img_ema = wandb.Image(images_all[:,row].clamp(-1,1))
        r_img = wandb.Image(grid_i_all[row])
        if mask_type == 'seg':
            r_mask = wandb.Image(color_mask(grid_m_all[row].squeeze(0).cpu()))
        else:
            r_mask = wandb.Image(grid_m_all[row].squeeze(0).cpu().numpy())
        if len(segs_all) > 0:
            g_mask = segs_all[:,row] # 6 x 512 x 512
            if mask_type == 'seg':
                g_mask = torch.tensor(color_mask(g_mask.cpu()).transpose(0,3,1,2))
            else:
                g_mask = torch.tensor(g_mask.cpu().numpy().clip(-1, 1))
            g_mask = wandb.Image(g_mask)
            table.add_data(r_img, r_mask, g_img_ema, g_mask)
        else:
            table.add_data(r_img, r_mask, g_img_ema)
    wandb.log({"Visualize": table}, step=global_step)
    del images_all, grid_i_all, grid_m_all, segs_all


#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    wandb_log               = False,    # Log to Weights & Biases.
    exp_name                = 'default',# Experiment name.
    no_eval                 = False,    # Disable evaluation of metrics.
    debug                   = False,    # Enable debug mode.
    D_semantic_kwargs       = None,       # Options for discriminator mask network.
):
    kwargs = locals()
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels, semantic_channels=G_kwargs.mapping_kwargs.in_channels, data_type=training_set.data_type)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    D_semantic = None
    if D_semantic_kwargs is not None:
        D_semantic = dnnlib.util.construct_class_by_name(**D_semantic_kwargs, c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels + G_kwargs.mapping_kwargs.in_channels).train().requires_grad_(False).to(device)

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False, allow_mismatch=True)
        if D_semantic is not None:
            if 'D_semantic' in resume_data:
                misc.copy_params_and_buffers(resume_data['D_semantic'], D_semantic, require_all=False, allow_mismatch=True)
            else:
                misc.copy_params_and_buffers(resume_data['D'], D_semantic, require_all=False, allow_mismatch=True)



    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        batch = {'pose':torch.empty([batch_gpu, G.c_dim], device=device), 'mask':torch.empty([batch_gpu, 1, training_set.resolution, training_set.resolution], device=device)}
        img = misc.print_module_summary(G, [z, c, batch])
        misc.print_module_summary(D, [img, c])
        del z, c, batch, img

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')


    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe, D_semantic]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, D_semantic=D_semantic, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval), ('D_semantic', D_semantic, D_opt_kwargs, D_reg_interval)]:
        if module is None:
            continue
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

        if wandb_log:
            import wandb   # Setup wandb for logging
            wandb.init(project="EG3D-Conditional", name=exp_name)
            wandb.config.update(kwargs)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, masks, poses = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        if training_set_kwargs.data_type == 'seg':
            masks_color = color_mask(masks.squeeze(1)).transpose(0,3,1,2)
        elif training_set_kwargs.data_type == 'edge':
            masks_color = 255 - masks
            masks = -(masks.to(torch.float32) / 127.5 - 1)
        save_image_grid(masks_color, os.path.join(run_dir, 'mask.png'), drange=[0, 255], grid_size=grid_size)

        grid_z = torch.randn([images.shape[0], G.z_dim]).split(batch_gpu)
        grid_i = (images.float() / 127.5 - 1).split(batch_gpu)
        grid_m = masks.split(batch_gpu)
        grid_p = poses.split(batch_gpu)
        grid_c = poses.split(batch_gpu)

        if wandb_log:
            wandb.log({"Real Images": [wandb.Image(get_image_grid(images, drange=[0,255], grid_size=grid_size))]}, step=0)
            wandb.log({"Real Masks": [wandb.Image(get_image_grid(masks_color, drange=[0,255], grid_size=grid_size))]}, step=0)

        # # Fake init
        # out = []
        # for round_idx in range(len(grid_z)):
        #     o = G_ema(grid_z[round_idx].to(device), grid_c[round_idx].to(device), 
        #     {'image':grid_i[round_idx].to(device), 'pose':grid_p[round_idx].to(device), 'mask':grid_m[round_idx].to(device)}, noise_mode='const')
        #     out.append(o)
        # # out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
        # images = torch.cat([o['image'].cpu() for o in out]).numpy()
        # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
        # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
        # save_image_grid(images, os.path.join(run_dir, f'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        # save_image_grid(images_raw, os.path.join(run_dir, f'fakes_init_raw.png'), drange=[-1,1], grid_size=grid_size)
        # save_image_grid(images_depth, os.path.join(run_dir, f'fakes_init_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

        # if 'semantic' in out[0]:
        #     if training_set_kwargs.data_type == 'seg':
        #         images_semantic = torch.cat([torch.argmax(o['semantic'], dim=1).cpu() for o in out]).numpy()
        #         images_semantic_raw = torch.cat([torch.argmax(o['semantic_raw'], dim=1).cpu() for o in out]).numpy()
        #         images_semantic_color = color_mask(images_semantic).transpose(0, 3, 1, 2)
        #         images_semantic_raw_color = color_mask(images_semantic_raw).transpose(0, 3, 1, 2)
        #     elif training_set_kwargs.data_type == 'edge':
        #         images_semantic = torch.cat([o['semantic'].cpu() for o in out]).numpy()
        #         images_semantic_raw = torch.cat([o['semantic_raw'].cpu() for o in out]).numpy()
        #         images_semantic_color = 255 - (images_semantic + 1) * 127.5
        #         images_semantic_raw_color = 255- (images_semantic_raw + 1) * 127.5
            
        #     save_image_grid(images_semantic_color, os.path.join(run_dir, f'fakes_init_semantic.png'), drange=[0, 255], grid_size=grid_size)
        #     save_image_grid(images_semantic_raw_color, os.path.join(run_dir, f'fakes_init_semantic_raw.png'), drange=[0, 255], grid_size=grid_size)
            

        # if wandb_log:
        #     wandb.log({'fakes_init': wandb.Image(get_image_grid(images, drange=[-1,1], grid_size=grid_size))}, step=0)
        #     wandb.log({'fakes_init_raw': wandb.Image(get_image_grid(images_raw, drange=[-1,1], grid_size=grid_size))}, step=0)
        #     wandb.log({'fakes_init_depth': wandb.Image(get_image_grid(images_depth, drange=[images_depth.min(), images_depth.max()], grid_size=grid_size))}, step=0)
        #     if 'semantic' in out[0]:
        #         wandb.log({'fakes_init_semantic': wandb.Image(get_image_grid(images_semantic_color, drange=[0, 255], grid_size=grid_size))}, step=0)
        #         wandb.log({'fakes_init_semantic_raw': wandb.Image(get_image_grid(images_semantic_raw_color, drange=[0, 255], grid_size=grid_size))}, step=0)


    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        torch.cuda.empty_cache()
        # Fetch training data.
        if debug:
            print(f"rank: {rank}, cur_tick: {cur_tick}, Fetch training data.")
        with torch.autograd.profiler.record_function('data_fetch'):
            def load_data(iterator):
                batch = next(iterator)
                # batch['image_raw'] = batch['image'].clone()
                    # batch['mask'] = resize_mask(batch['mask'], curr_res)
                # img = [{'img': img} for img in (img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)]
                # c = c.to(device).split(batch_gpu)
                batch['image'] = (batch['image'].to(torch.float32) / 127.5 - 1) # [-1, 1]
                # if mask_type == 'kp':
                #     batch['mask'] = (batch['mask'].to(torch.float32) * 2 - 1) # [-1, 1]
                if training_set_kwargs.data_type == 'edge':
                    batch['mask'] = -(batch['mask'].to(torch.float32) / 127.5 - 1) # [-1, 1]
                for key in batch:
                    batch[key] = batch[key].to(device).split(batch_gpu)
                return batch

            phase_batch = load_data(training_set_iterator)
            # phase_real_img, phase_real_c = next(training_set_iterator)
            # phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            # phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            # for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
            #     loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            for round_idx in range(len(phase_batch['image'])):
                if debug:
                    print(f"rank: {rank}, cur_tick: {cur_tick}, cur_nimg: {cur_nimg}, phase: {phase.name}, round_idx: {round_idx}")
                round_batch = {key:phase_batch[key][round_idx] for key in phase_batch}
                loss.accumulate_gradients(phase=phase.name, batch=round_batch, gen_z=phase_gen_z[round_idx], gen_c=phase_gen_c[round_idx], gain=phase.interval, cur_nimg=cur_nimg)
                # if debug:
                #     print(f"rank: {rank}, cur_tick: {cur_tick}, phase: {phase.name}, round_idx: {round_idx} Done")
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            out = []
            for round_idx in range(len(grid_z)):
                o = G_ema(grid_z[round_idx].to(device), grid_c[round_idx].to(device), 
                {'image':grid_i[round_idx].to(device), 'pose':grid_p[round_idx].to(device), 'mask':grid_m[round_idx].to(device)}, noise_mode='const')
                out.append(o)
            # out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            if 'semantic' in out[0]:
                if training_set_kwargs.data_type == 'seg':
                    images_semantic = torch.cat([torch.argmax(o['semantic'], dim=1).cpu() for o in out]).numpy()
                    images_semantic_raw = torch.cat([torch.argmax(o['semantic_raw'], dim=1).cpu() for o in out]).numpy()
                    images_semantic_color = color_mask(images_semantic).transpose(0, 3, 1, 2)
                    images_semantic_raw_color = color_mask(images_semantic_raw).transpose(0, 3, 1, 2)
                elif training_set_kwargs.data_type == 'edge':
                    images_semantic = torch.cat([o['semantic'].cpu() for o in out]).numpy()
                    images_semantic_raw = torch.cat([o['semantic_raw'].cpu() for o in out]).numpy()
                    images_semantic_color = (images_semantic + 1) * 127.5
                    images_semantic_raw_color = (images_semantic_raw + 1) * 127.5
                
                save_image_grid(images_semantic_color, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_semantic.png'), drange=[0, 255], grid_size=grid_size)
                save_image_grid(images_semantic_raw_color, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_semantic_raw.png'), drange=[0, 255], grid_size=grid_size)

            if 'weight' in out[0]:
                images_weight = torch.cat([o['weight'].cpu() for o in out]).numpy()
                save_image_grid(images_weight, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_weight.png'), drange=[0, 1], grid_size=grid_size)
                

            if wandb_log:
                wandb.log({'fakes': wandb.Image(get_image_grid(images, drange=[-1,1], grid_size=grid_size))}, step=cur_nimg//1000)
                wandb.log({'fakes_raw': wandb.Image(get_image_grid(images_raw, drange=[-1,1], grid_size=grid_size))}, step=cur_nimg//1000)
                wandb.log({'fakes_depth': wandb.Image(get_image_grid(images_depth, drange=[images_depth.min(), images_depth.max()], grid_size=grid_size))}, step=cur_nimg//1000)
                if 'semantic' in out[0]:
                    wandb.log({'fakes_semantic': wandb.Image(get_image_grid(images_semantic_color, drange=[0, 255], grid_size=grid_size))}, step=cur_nimg//1000)
                    wandb.log({'fakes_semantic_raw': wandb.Image(get_image_grid(images_semantic_raw_color, drange=[0, 255], grid_size=grid_size))}, step=cur_nimg//1000)
                if 'weight' in out[0]:
                    wandb.log({'fakes_weight': wandb.Image(get_image_grid(images_weight, drange=[0, 1], grid_size=grid_size))}, step=cur_nimg//1000)

            #--------------------
            # Log front-view images.
            del out
            forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            forward_pose = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            out = []
            for round_idx in range(len(grid_z)):
                o = G_ema(grid_z[round_idx].to(device), forward_pose.expand(grid_z[round_idx].shape[0], -1), 
                {'image':grid_i[round_idx].to(device), 'pose':grid_p[round_idx].to(device), 'mask':grid_m[round_idx].to(device)}, noise_mode='const')
                out.append(o)
            # out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes_front{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes_front{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes_front{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            if 'semantic' in out[0]:
                if training_set_kwargs.data_type == 'seg':
                    images_semantic = torch.cat([torch.argmax(o['semantic'], dim=1).cpu() for o in out]).numpy()
                    images_semantic_raw = torch.cat([torch.argmax(o['semantic_raw'], dim=1).cpu() for o in out]).numpy()
                    images_semantic_color = color_mask(images_semantic).transpose(0, 3, 1, 2)
                    images_semantic_raw_color = color_mask(images_semantic_raw).transpose(0, 3, 1, 2)
                elif training_set_kwargs.data_type == 'edge':
                    images_semantic = torch.cat([o['semantic'].cpu() for o in out]).numpy()
                    images_semantic_raw = torch.cat([o['semantic_raw'].cpu() for o in out]).numpy()
                    images_semantic_color = (images_semantic + 1) * 127.5
                    images_semantic_raw_color = (images_semantic_raw + 1) * 127.5
                save_image_grid(images_semantic_color, os.path.join(run_dir, f'fakes_front{cur_nimg//1000:06d}_semantic.png'), drange=[0, 255], grid_size=grid_size)
                save_image_grid(images_semantic_raw_color, os.path.join(run_dir, f'fakes_front{cur_nimg//1000:06d}_semantic_raw.png'), drange=[0, 255], grid_size=grid_size)

            if wandb_log:
                wandb.log({'fakes_front': wandb.Image(get_image_grid(images, drange=[-1,1], grid_size=grid_size))}, step=cur_nimg//1000)
                wandb.log({'fakes_front_raw': wandb.Image(get_image_grid(images_raw, drange=[-1,1], grid_size=grid_size))}, step=cur_nimg//1000)
                wandb.log({'fakes_front_depth': wandb.Image(get_image_grid(images_depth, drange=[images_depth.min(), images_depth.max()], grid_size=grid_size))}, step=cur_nimg//1000)
                if 'semantic' in out[0]:
                    wandb.log({'fakes_front_semantic': wandb.Image(get_image_grid(images_semantic_color, drange=[0, 255], grid_size=grid_size))}, step=cur_nimg//1000)
                    wandb.log({'fakes_front_semantic_raw': wandb.Image(get_image_grid(images_semantic_raw_color, drange=[0, 255], grid_size=grid_size))}, step=cur_nimg//1000)

            #--------------------
            # Log Multi-view images.
            log_table(G_ema, grid_z, grid_i, grid_c, grid_m, grid_p, mask_type=training_set_kwargs.data_type, global_step=cur_nimg//1000, device=device, wandb=wandb)



            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe), ('D_semantic', D_semantic)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                else:
                    continue
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0) and not no_eval:
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        if debug:
            print(f"rank: {rank}, cur_tick: {cur_tick}, Collecting statistics.")
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        if debug:
            print(f"rank: {rank}, cur_tick: {cur_tick}, Updating logs.")
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        if wandb_log and rank == 0:
            wandb.log({name: value.mean for name, value in stats_dict.items()} , step=int(cur_nimg / 1e3))
            wandb.log(stats_metrics, step=int(cur_nimg / 1e3))

        # Update state.
        if debug:
            print(f"rank: {rank}, cur_tick: {cur_tick}, Updating state.")
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
