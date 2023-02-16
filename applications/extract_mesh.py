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

import trimesh
import pyrender
import mcubes

os.environ["PYOPENGL_PLATFORM"] = "egl"

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

def get_sigma_field_np(nerf, styles, resolution=512, block_resolution=64):
    # return numpy array of forwarded sigma value
    # bound = (nerf.rendering_kwargs['ray_end'] - nerf.rendering_kwargs['ray_start']) * 0.5
    bound = nerf.rendering_kwargs['box_warp'] * 0.5
    X = torch.linspace(-bound, bound, resolution).split(block_resolution)

    sigma_np = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(X):
            for zi, zs in enumerate(X):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).to(styles.device)  # B, H, H, H, C
                block_shape = [1, len(xs), len(ys), len(zs)]
                out = nerf.sample_mixed(pts.reshape(1,-1,3), None, ws=styles, noise_mode='const')
                feat_out, sigma_out = out['rgb'], out['sigma']
                sigma_np[xi * block_resolution: xi * block_resolution + len(xs), \
                yi * block_resolution: yi * block_resolution + len(ys), \
                zi * block_resolution: zi * block_resolution + len(zs)] = sigma_out.reshape(block_shape[1:]).detach().cpu().numpy()
                # print(feat_out.shape)

    return sigma_np, bound


def extract_geometry(nerf, styles, resolution, threshold):

    # print('threshold: {}'.format(threshold))
    u, bound = get_sigma_field_np(nerf, styles, resolution)
    vertices, faces = mcubes.marching_cubes(u, threshold)
    # vertices, faces, normals, values = skimage.measure.marching_cubes(
    #     u, level=10
    # )
    b_min_np = np.array([-bound, -bound, -bound])
    b_max_np = np.array([ bound,  bound,  bound])

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices.astype('float32'), faces



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate samples from a trained model')
    parser.add_argument('--network', help='Path to the network pickle file', required=True)
    parser.add_argument('--outdir', help='Directory to save the output', required=True)

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

         # Save the visualized input label map
        PIL.Image.fromarray(color_mask(input_label[0,0].cpu().numpy()).astype(np.uint8)).save(save_dir / f'{args.cfg}_input.png')        
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
    z = torch.from_numpy(np.random.RandomState(int(0)).randn(1, G.z_dim).astype('float32')).to(device)

    with torch.no_grad():
        ws = G.mapping(z, input_pose, {'mask': input_label, 'pose': input_pose})
    
    mesh_trimesh = trimesh.Trimesh(*extract_geometry(G, ws, resolution=512, threshold=50.))

    verts_np = np.array(mesh_trimesh.vertices)
    colors = torch.zeros((verts_np.shape[0], 3), device=device)
    semantic_colors = torch.zeros((verts_np.shape[0], 6), device=device)
    samples_color = torch.tensor(verts_np, device=device).unsqueeze(0).float()

    head = 0
    max_batch = 10000000
    with tqdm(total = verts_np.shape[0]) as pbar:
        with torch.no_grad():
            while head < verts_np.shape[0]:
                torch.manual_seed(0)
                out = G.sample_mixed(samples_color[:, head:head+max_batch], None, ws, truncation_psi=1, noise_mode='const')
                # sigma = out['sigma']
                colors[head:head+max_batch, :] = out['rgb'][0,:,:3]
                seg = out['rgb'][0, :, 32:32+6]
                semantic_colors[head:head+max_batch, :] = seg
                # semantics[:, head:head+max_batch] = out['semantic']
                head += max_batch
                pbar.update(max_batch)

    semantic_colors = torch.tensor(color_list)[torch.argmax(semantic_colors, dim=-1)]

    mesh_trimesh.visual.vertex_colors = semantic_colors.cpu().numpy().astype(np.uint8)

    # Save mesh.
    mesh_trimesh.export(os.path.join(save_dir, f'semantic_mesh.ply'))

    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    r = pyrender.OffscreenRenderer(512, 512)
    camera = pyrender.OrthographicCamera(xmag=0.3, ymag=0.3)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                    innerConeAngle=np.pi/4)

    frames_mesh = []
    num_frames = 120
    pitch_range = 0.25
    yaw_range = 0.35

    for frame_idx in tqdm(range(num_frames)):
        scene = pyrender.Scene()
        scene.add(mesh)

        camera_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames),
                                            torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device), radius=1, device=device)
        camera_pose = camera_pose.reshape(4, 4).cpu().numpy().copy()
        camera_pose[:, 1] = -camera_pose[:, 1]
        camera_pose[:, 2] = -camera_pose[:, 2]

        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, depth = r.render(scene)
        frames_mesh.append(color)

    imageio.mimsave(os.path.join(save_dir, f'rendered_mesh_colored.gif'), frames_mesh, fps=60)
    r.delete()



if __name__ == '__main__':
    main()