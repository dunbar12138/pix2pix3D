import sys
sys.path.append('./')

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm


import legacy
from camera_utils import LookAtPoseSampler

from matplotlib import pyplot as plt

from pathlib import Path

import json

from training.utils import color_mask, color_list

from tqdm import tqdm

import imageio

import argparse

def init_conditional_dataset_kwargs(data, mask_data, data_type, resolution=None):
    try:
        if data_type =='seg':
            dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageSegFolderDataset', path=data, mask_path=mask_data, data_type=data_type, use_labels=True, max_size=None, xflip=False, resolution=resolution)
            dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
            dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
            dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
            return dataset_kwargs, dataset_obj.name
        elif data_type == 'edge':
            dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageEdgeFolderDataset', path=data, mask_path=mask_data, data_type=data_type, use_labels=True, max_size=None, xflip=False)
            dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
            dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
            dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
            return dataset_kwargs, dataset_obj.name
        else:
            raise click.ClickException(f'Unknown data_type: {data_type}')
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

def render_video(G, ws, intrinsics, num_frames = 120, pitch_range = 0.25, yaw_range = 0.35, neural_rendering_resolution = 128, device='cuda'):
    frames, frames_label = [], []

    for frame_idx in tqdm(range(num_frames)):
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames),
                                                3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames),
                                                torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device), radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        pose = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        with torch.no_grad():
            # out = G(z, pose, {'mask': batch['mask'].unsqueeze(0).to(device), 'pose': torch.tensor(batch['pose']).unsqueeze(0).to(device)})
            out = G.synthesis(ws, pose, noise_mode='const', neural_rendering_resolution=neural_rendering_resolution)
        frames.append(((out['image'].cpu().numpy()[0] + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0))
        frames_label.append(color_mask(torch.argmax(out['semantic'], dim=1).cpu().numpy()[0]).astype(np.uint8))

    return frames, frames_label

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate samples from a trained model')
    parser.add_argument('--network', help='Path to the network pickle file', required=True)
    parser.add_argument('--outdir', help='Directory to save the output', required=True)
    # Define an argument of a list of random seeds
    parser.add_argument('--random_seed', help='Random seed', type=list, default=[0])

    parser.add_argument('--input_id', type=int, default=0, help='Input label map id', required=False)
    parser.add_argument('--data_dir', default='data/', help='Directory to the data', required=False)
    parser.add_argument('--input', help='input label map', required=False)
    parser.add_argument('--cfg', help='Base Configuration: seg2face, seg2cat, edge2car', required=True)
    args = parser.parse_args()
    device = 'cuda'

    # Load the network
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

    if args.cfg == 'seg2cat' or args.cfg == 'seg2face':
        neural_rendering_resolution = 128
        data_type = 'seg'
        # Initialize pose sampler.
        forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device), 
                                                        radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal_length = 4.2647 # shapenet has higher FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        forward_pose = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    elif args.cfg == 'edge2car':
        neural_rendering_resolution = 64
        data_type= 'edge'
    else:
        print('Invalid cfg')
        return

    save_dir = Path(args.outdir)

    # Load the input label map
    if args.input is not None:
        input_label = PIL.Image.open(args.input)
        input_label = np.array(input_label).astype(np.uint8)
        input_label = torch.from_numpy(input_label).unsqueeze(0).unsqueeze(0).to(device)
        input_pose = forward_pose.to(device)
    elif args.input_id is not None:
        # Initialize dataset.
        data_path = Path(args.data_dir) / 'afhq_v2_train_cat_512.zip'
        mask_data = Path(args.data_dir) / 'afhqcat_seg_6c.zip'
        # data_path = '/data2/datasets/AFHQ_eg3d/afhq_v2_train_cat_512.zip'
        # mask_data = '/data2/datasets/AFHQ_eg3d/afhqcat_seg_6c.zip'
        dataset_kwargs, dataset_name = init_conditional_dataset_kwargs(str(data_path), str(mask_data), data_type)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        batch = dataset[args.input_id]


        # Save the input label map
        PIL.Image.fromarray(color_mask(batch['mask'][0]).astype(np.uint8)).save(save_dir / f'{args.cfg}_{args.input_id}_input.png')

        input_pose = torch.tensor(batch['pose']).unsqueeze(0).to(device)
        input_label = torch.tensor(batch['mask']).unsqueeze(0).to(device)

    # Generate videos
    for seed in args.random_seed:
        z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, G.z_dim).astype('float32')).to(device)

        with torch.no_grad():
            ws = G.mapping(z, input_pose, {'mask': input_label, 'pose': input_pose})
        
        # Generate the video
        frames, frames_label = render_video(G, ws, intrinsics, num_frames = 120, pitch_range = 0.25, yaw_range = 0.35, neural_rendering_resolution=neural_rendering_resolution, device=device)

        # Save the video
        imageio.mimsave(save_dir / f'{args.cfg}_{args.input_id}_{seed}.mp4', frames, fps=60)
        imageio.mimsave(save_dir / f'{args.cfg}_{args.input_id}_{seed}_label.mp4', frames_label, fps=60)



if __name__ == '__main__':
    main()