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

from matplotlib import pyplot as plt

from pathlib import Path

import json

from training.utils import color_mask, color_list

from tqdm import tqdm

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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate samples from a trained model')
    parser.add_argument('--network', help='Path to the network pickle file', required=True)
    parser.add_argument('--outdir', help='Directory to save the output', required=True)
    # Define an argument of a list of random seeds
    parser.add_argument('--random_seed', help='Random seed', nargs="+", type=int)

    parser.add_argument('--input_id', type=int, default=0, help='Input label map id', required=True)
    parser.add_argument('--data_dir', default='data/', help='Directory to the data', required=False)
    parser.add_argument('--cfg', help='Base Configuration: seg2face, seg2cat, edge2car', required=True)
    args = parser.parse_args()
    device = 'cuda'

    if args.cfg == 'seg2cat' or args.cfg == 'seg2face':
        neural_rendering_resolution = 128
        data_type = 'seg'
    elif args.cfg == 'edge2car':
        neural_rendering_resolution = 64
        data_type= 'edge'
    else:
        print('Invalid cfg')
        return

    # Load the network
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

    # Load the input label map
    # Initialize dataset.
    # data_path = Path(args.data_dir) / 'afhq_v2_train_cat_512.zip'
    # mask_data = Path(args.data_dir) / 'afhqcat_seg_6c.zip'
    data_path = '/data2/datasets/AFHQ_eg3d/afhq_v2_train_cat_512.zip'
    mask_data = '/data2/datasets/AFHQ_eg3d/afhqcat_seg_6c_no_nose.zip'
    dataset_kwargs, dataset_name = init_conditional_dataset_kwargs(str(data_path), str(mask_data), data_type)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    batch = dataset[args.input_id]

    save_dir = Path(args.outdir)

    # Save the input label map
    PIL.Image.fromarray(color_mask(batch['mask'][0]).astype(np.uint8)).save(save_dir / f'{args.cfg}_{args.input_id}_input.png')

    # Generate samples
    for seed in args.random_seed:
        z = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, G.z_dim).astype('float32')).to(device)
        input_pose = torch.tensor(batch['pose']).unsqueeze(0).to(device)
        input_label = torch.tensor(batch['mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            ws = G.mapping(z, input_pose, {'mask': input_label, 'pose': input_pose})
            out = G.synthesis(ws, input_pose, noise_mode='const', neural_rendering_resolution=neural_rendering_resolution)
            
        image_color = ((out['image'][0].permute(1, 2, 0).cpu().numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8)
        image_seg = color_mask(torch.argmax(out['semantic'][0], dim=0).cpu().numpy()).astype(np.uint8)

        PIL.Image.fromarray(image_color).save(save_dir / f'{args.cfg}_{args.input_id}_{seed}_color.png')
        PIL.Image.fromarray(image_seg).save(save_dir / f'{args.cfg}_{args.input_id}_{seed}_label.png')



if __name__ == '__main__':
    main()