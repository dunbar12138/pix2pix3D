import sys
sys.path.append('./')

import os
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from ui_qt.ui_clean import Ui_Form_Seg2cat as Ui_Form
from ui_qt.mouse_event import GraphicsScene

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import dnnlib
import legacy
from torch_utils import misc

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane_cond import TriPlaneGenerator
from training.training_loop import setup_snapshot_image_grid, get_image_grid, save_image_grid
from train import init_conditional_dataset_kwargs

from matplotlib import pyplot as plt

from pathlib import Path

from rich.progress import track
import json

import imageio

from torch import nn
import torch.nn.functional as F

import argparse

from scipy.spatial.transform import Rotation as R

from training.utils import color_mask as color_mask_np

color_list = [QColor(255, 255, 255), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(255, 204, 204), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

def color_mask(m):
    my_color_list = [[255, 255, 255], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    if len(m.shape) == 2:
        im_base = np.zeros((m.shape[0], m.shape[1], 3))
        for idx, color in enumerate(my_color_list):
            im_base[m == idx] = color
        return im_base
    elif len(m.shape) == 3:
        im_base = np.zeros((m.shape[0], m.shape[1], m.shape[2], 3))
        for idx, color in enumerate(my_color_list):
            im_base[m == idx] = color
        return im_base

def get_camera_traj(model, pitch, yaw, fov=12, batch_size=1, device='cuda'):
    gen = model.synthesis
    range_u, range_v = gen.C.range_u, gen.C.range_v
    u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
    v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
    cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device, fov=fov)
    return cam

def get_yaw_pitch(cam2world):
    forward = cam2world[0:3, 2]
    yaw = np.arctan2(forward[0], forward[2]) - np.pi / 2
    phi = np.arccos(forward[1])
    v = (1 - np.cos(phi)) / 2
    pitch = (1+forward[1]) / 2 * np.pi
    return yaw, pitch

def create_cam2world_fromeuler(euler, radius):
    r = R.from_euler('zyx', euler, degrees=False)
    cam2world = r.as_matrix()
    cam2world = np.concatenate([cam2world, np.array([[0, 0, 0]])], axis=0)
    cam2world = np.concatenate([cam2world, np.array([0, 0, 0, 1])[...,None]], axis=1)
    cam2world[:3, 3] = -cam2world[:3, 2] * radius
    return cam2world

class Ex(QWidget, Ui_Form):
    def __init__(self):
        super(Ex, self).__init__()
        self.size = 35
        self.yaw = 100
        self.pitch = 50
        self.roll = 0
        self.truncation = 0.75
        
        self.setupUi(self)
        self.show()
        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.mask = None
        self.mask_m = np.ones((512, 512, 1), dtype=np.uint8) * 255
        self.img = None

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None
        self.device = torch.device('cuda')

        # Parse arguments
        parser = argparse.ArgumentParser(description='Real-time 3D editing demo')
        parser.add_argument('--network', help='Path to the network pickle file', required=True)
        parser.add_argument('--data_dir', default='data/', help='Directory to the data', required=False)
        args = parser.parse_args()

        network_pkl = args.network
        self.G = self.get_model(network_pkl)
        
        # Initialize dataset.
        data_path = Path(args.data_dir) / 'afhq_v2_train_cat_512.zip'
        mask_data = Path(args.data_dir) / 'afhqcat_seg_6c.zip'
        data_type= 'seg'
        dataset_kwargs, dataset_name = init_conditional_dataset_kwargs(data_path, mask_data, data_type)
        self.training_data = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        
        self.input_batch = None
        # self.ws = None
        self.ws_texture = None

        self.buffer_mask = None

        focal_length = 4.2647 # shapenet has higher FOV
        self.intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)

        os.makedirs('examples/ui', exist_ok=True)


    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:    
            mat_img = cv2.imread(fileName)
            self.mask = mat_img.copy()
            self.mask_m = mat_img     
            mat_img = mat_img.copy()
            image = QImage(mat_img, 512, 512, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(512):
                for j in range(512):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)


    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18

    def increase(self):
        if self.scene.size < 50:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 

    def changeBrushSize(self, s):
        self.scene.size = s

    def changeYaw(self, s):
        self.yaw = s
        if self.ws is not None:
            self.generate()

    def changePitch(self, s):
        self.pitch = s
        # print('changing pitch', self.pitch)
        if self.ws is not None:
            self.generate()

    def changeRoll(self, s):
        self.roll = s
        if self.ws is not None:
            self.generate()

    def changeTruncation(self, s):
        self.truncation = s / 100
        self.reconstruct()
        self.generate()

    def inputID(self):
        input_id = int(self.text_inputID.toPlainText())
        self.input_batch = self.training_data[input_id]

        self.mask = self.input_batch['mask'].transpose(1,2,0).astype(np.uint8)
        # self.mask_m = self.input_batch['mask'].transpose(1,2,0).astype(np.uint8)
        mat_img = cv2.resize(self.mask[:,:,0], (512, 512), interpolation=cv2.INTER_NEAREST)
        self.mask = mat_img[:,:,np.newaxis]
        self.mask_m = self.mask.copy()
        image = QImage(mat_img, 512, 512, QImage.Format_RGB888)


        for i in range(512):
            for j in range(512):
                # r, g, b, a = image.pixelColor(i, j).getRgb()
                r = mat_img[j, i]
                # print(r)
                image.setPixel(i, j, color_list[r].rgb()) 
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image)  
        self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        self.scene.reset()
        if len(self.scene.items())>0:
            self.scene.reset_items() 
        self.scene.addPixmap(self.image)

        self.ws = None

        roll, yaw, pitch = R.from_matrix(self.input_batch['pose'][:16].reshape(4,4)[:3,:3]).as_euler('zyx', degrees=False)
        # print(yaw, pitch)
        # print(self.input_batch['pose'][:16].reshape(4, 4))
        pitch_range = np.pi
        yaw_range = np.pi / 2
        roll_range = np.pi / 4

        pitch = pitch - np.pi
        pitch = pitch + 2 * np.pi if pitch < -np.pi else pitch

        self.yaw = ((yaw) / yaw_range * 100)
        self.pitch = ((pitch) / pitch_range * 100)
        self.roll = ((roll) / (roll_range) * 100)
        print(self.roll, self.yaw, self.pitch)

        self.intrinsics = torch.tensor(self.input_batch['pose'][16:].reshape(3,3)).float().to(self.device)


        self.slider_yawselect.setValue(self.yaw)
        self.slider_pitchselect.setValue(self.pitch)
        self.slider_rollselect.setValue(self.roll)


    def get_mask(self): # get from output
        mat_img = self.buffer_mask[0].astype(np.uint8)

        # print(self.mask.shape)
        # mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.mask = mat_img[:,:,np.newaxis]
        self.mask_m = self.mask.copy()
        # print(mat_img.shape)
        image = QImage(mat_img, 512, 512, QImage.Format_RGB888)


        for i in range(512):
            for j in range(512):
                # r, g, b, a = image.pixelColor(i, j).getRgb()
                r = mat_img[j, i]
                # print(r)
                image.setPixel(i, j, color_list[r].rgb()) 
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image)  
        self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        self.scene.reset()
        if len(self.scene.items())>0:
            self.scene.reset_items() 
        self.scene.addPixmap(self.image)



    def generate(self):
        ws = self.ws

        pitch_range = np.pi
        yaw_range = np.pi / 2
        roll_range = np.pi / 4
        pitch = self.pitch / 100 * pitch_range
        yaw = self.yaw / 100 * yaw_range
        roll = self.roll / 100 * roll_range
        
        cam2world_pose = torch.tensor(create_cam2world_fromeuler([roll, yaw, pitch+np.pi], radius=2.7)).float().to(self.device)

        # print(cam2world_pose)
        pose = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)

        out = self.G.synthesis(ws, pose.to(self.device), noise_mode='const', neural_rendering_resolution=128)
        img = ((out['image'].permute(0,2,3,1).squeeze(0).cpu().numpy().clip(-1, 1) * 0.5 + 0.5) * 255).astype(np.uint8).copy()
        if out['image'].shape[-1] != 512:
            # print(f"Resizing {out['image'].shape[-1]} to {512}")
            img = cv2.resize(img, (512, 512))

        qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        if len(self.result_scene.items())>0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))

        self.buffer_mask = torch.argmax(out['semantic'].detach(), dim=1).cpu().numpy() # 1 x 512 x 512
        self.output_img = img.copy()
        self.get_mask()


    def reconstruct(self):
        if self.input_batch is None:
            return
        ws = self.ws

        out = self.G.synthesis(ws, torch.tensor(self.input_batch['pose']).unsqueeze(0).to(self.device))
        if out['img'].shape[-1] != 512:
            print(f"Resizing {out['img'].shape[-1]} to {512}")
            img = resize_image(out['img'].detach(), 512)
        else:
            img = out['img'].detach()
        img = ((img.permute(0,2,3,1).squeeze(0).cpu().numpy().clip(-1, 1) * 0.5 + 0.5) * 255).astype(np.uint8).copy()
        qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        if len(self.ref_scene.items())>0:
            self.ref_scene.removeItem(self.ref_scene.items()[-1])
        self.ref_scene.addPixmap(QPixmap.fromImage(qim))

        if out['semantic'].shape[-1] != 512:
            seg = resize_image(out['semantic'].detach(), 512)
        else:
            seg = out['semantic'].detach()
        seg = color_mask(torch.argmax(seg, dim=1).cpu())[0].astype(np.uint8).copy()
        qim = QImage(seg.data, seg.shape[1], seg.shape[0], seg.strides[0], QImage.Format_RGB888)
        if len(self.ref_seg_scene.items())>0:
            self.ref_seg_scene.removeItem(self.ref_seg_scene.items()[-1])
        self.ref_seg_scene.addPixmap(QPixmap.fromImage(qim))

    def get_ws(self):
        z = torch.from_numpy(np.random.RandomState(int(self.text_seed.toPlainText())).randn(1, self.G.z_dim).astype('float32')).to(self.device)
    
        for i in range(6):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)
        
        cv2.imwrite('examples/ui/mask_input.png', self.mask_m)

        forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor(self.G.rendering_kwargs['avg_camera_pivot'], device=self.device), 
                                                radius=self.G.rendering_kwargs['avg_camera_radius'], device=self.device)
        focal_length = 4.2647 # shapenet has higher FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)
        forward_pose = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        self.ws = self.G.mapping(z, forward_pose.to(self.device), 
                        {'mask': torch.tensor(self.mask_m[None,...,0]).unsqueeze(0).to(self.device), 'pose': torch.tensor(self.input_batch['pose']).unsqueeze(0).to(self.device)})

        if self.ws_texture is None:
            self.ws_texture = self.ws[:,8:,:]
        else:
            self.ws[:,8:,:] = self.ws_texture
        # print(self.ws[:,8:,:])

    
    def generateAndReconstruct(self):
        self.get_ws()
        self.generate()
        # self.reconstruct()


    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        for i in range(6):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)
        mask_np = np.array(self.mask_m)[..., 0]
        print(mask_np.shape)
        cv2.imwrite('examples/ui/mask.png', mask_np)
        cv2.imwrite('examples/ui/mask_color.png', color_mask_np(mask_np).astype(np.uint8)[...,::-1])
        cv2.imwrite('examples/ui/output.png',self.output_img[...,::-1])


    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

        if len(self.result_scene.items())>0:
            self.result_scene.removeItem(self.result_scene.items()[-1])

        self.ws = None

    def get_model(self, network_pkl):
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

        return G

    def clear_ws(self):
        self.ws = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ex()
    sys.exit(app.exec_())
