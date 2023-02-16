import numpy as np

color_list = [[255, 255, 255], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

def color_mask(m):
    if len(m.shape) == 2:
        im_base = np.zeros((m.shape[0], m.shape[1], 3))
        for idx, color in enumerate(color_list):
            im_base[m == idx] = color
        return im_base
    elif len(m.shape) == 3:
        im_base = np.zeros((m.shape[0], m.shape[1], m.shape[2], 3))
        for idx, color in enumerate(color_list):
            im_base[m == idx] = color
        return im_base

# sns_colors = sns.color_palette("hls", 30)

# def color_kp(kp_map):
#     if len(kp_map.shape) == 3:
#         k, h, w = kp_map.shape
#         img = np.zeros((h,w,3), dtype=np.float32)
#         for i in range(k):
#             img += kp_map[i][...,None] * sns_colors[i] * 255
#         return img.astype(np.uint8)
#     elif len(kp_map.shape) == 4:
#         n, k, h, w = kp_map.shape
#         img = np.zeros((n,h,w,3), dtype=np.float32)
#         for i in range(k):
#             img += kp_map[:,i][...,None] * np.array(sns_colors[i])[None,...] * 255
#         return img.astype(np.uint8)