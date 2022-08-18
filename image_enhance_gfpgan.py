import sys
sys.path.insert(0, "/python/GFPGAN/")

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.utils import GFPGANer

arch = 'clean'
channel_multiplier = 2

realesrganer_model_path = '/python/GFPGAN/experiments/pretrained_models/RealESRGAN_x2plus.pth'
gfpganer_model_path ='/python/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth'


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path=realesrganer_model_path,
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True)  # need to set False in CPU mode

restorer = GFPGANer(
    model_path=gfpganer_model_path,
    upscale=2,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)


def restore_image(img_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')

    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=False, paste_back=True)

    if restored_img is None:
        print("enhanced failed!")
    else:
        print("enhanced success")

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # save cropped face
        save_crop_path = os.path.join(output_path, 'cropped_faces', f'{basename}_{idx:02d}.png')
        imwrite(cropped_face, save_crop_path)

        save_face_name = f'{basename}_{idx:02d}.png'
        save_restore_path = os.path.join(output_path, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)

    # save restored img
    if restored_img is not None:
        extension = ext[1:]
        save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}.{extension}')

        imwrite(restored_img, save_restore_path)

    return save_restore_path

