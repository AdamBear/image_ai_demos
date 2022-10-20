import pdb
import cv2
import os
from collections import OrderedDict

import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from PIL import Image
import base64
import io
import random


import models
import torch

from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--port', type=int, default=8897)
        parser.add_argument('--load_baseg', action="store_true")
        parser.add_argument('--dataset_mode', type=str, default='testimage')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')

        parser.set_defaults(name="objrmv")
        parser.set_defaults(model="inpaint")

        parser.set_defaults(netG="baseconv")
        parser.set_defaults(image_dir="./datasets/places2sample1k_val/places2samples1k_crop256")
        parser.set_defaults(mask_dir="./datasets/places2sample1k_val/places2samples1k_256_mask_square128")
        parser.set_defaults(output_dir="./results")

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=1024, load_size=1024,
                            display_winsize=1024)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser


max_size = 1024
max_num_examples = 200
UPLOAD_FOLDER = 'static/images'


def get_model():
    import sys
    sys.argv.append("--port")
    sys.argv.append("8897")
    sys.argv.append("--image_dir")
    sys.argv.append("./datasets/places2sample1k_val/places2samples1k_crop256")
    sys.argv.append("--mask_dir")
    sys.argv.append("./datasets/places2sample1k_val/places2samples1k_256_mask_square128")
    sys.argv.append("--output_dir")
    sys.argv.append("./results")

    opt = TestOptions().parse()

    model = models.create_model(opt)
    model.eval()
    return model


def process_image_cv(model, img, mask):
    pass


def process_image(model, img, mask):
    img = img.convert("RGB")

    w_raw, h_raw = img.size

    h_t, w_t = h_raw // 8 * 8, w_raw // 8 * 8
    img = img.resize((w_t, h_t))
    img_raw = np.array(img)

    img = np.array(img).transpose((2, 0, 1))

    mask = mask.resize((w_t, h_t))
    mask_raw = np.array(mask)[..., None] > 0

    mask = np.array(mask)
    mask = (torch.Tensor(mask) > 0).float()
    img = (torch.Tensor(img)).float()
    img = (img / 255 - 0.5) / 0.5
    img = img[None]
    mask = mask[None, None]

    with torch.no_grad():
        generated, _ = model(
            {'image': img, 'mask': mask},
            mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated + 1) / 2 * 255
    generated = generated.cpu().numpy().astype(np.uint8)
    generated = generated[0].transpose((1, 2, 0))
    result = generated * mask_raw + img_raw * (1 - mask_raw)

    result = result.astype(np.uint8)
    result = Image.fromarray(result).resize((w_raw, h_raw))

    return result