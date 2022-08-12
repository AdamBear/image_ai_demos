import sys
sys.path.insert(0, "d:\\apps\\nlp\\prompt\\modnet_docker")

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
import cv2
#from tqdm import tqdm

import streamlit as st
import time






ckpt_path = "d:\\apps\\nlp\\prompt\\modnet_docker\\pretrained\\modnet_webcam_portrait_matting.ckpt"

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

GPU = True if torch.cuda.device_count() > 0 else False

def remove_background(image, matte):
    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

    return Image.fromarray(np.uint8(foreground))


def get_model():
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if GPU:
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def process_video_mate(modnet, video_path, result, fps=30, alpha_matte = False, tqdm=None, st_progress_bar=None):
    vc = cv2.VideoCapture(video_path)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    if not rval:
        print('Failed to read the video: {0}'.format(video_path))
        exit()

    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    h, w = frame.shape[:2]
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32

    # video writer
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')

    video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))

    print('Start matting...')
    i = 0
    with tqdm(range(int(num_frame)), st_progress_bar=st_progress_bar) as t:
        for c in t:
            i += 1
            if i > 150:
                break

            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            if alpha_matte:
                view_np = matte_np * np.full(frame_np.shape, 255.0)
            else:
                view_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            view_np = cv2.cvtColor(view_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            view_np = cv2.resize(view_np, (w, h))
            video_writer.write(view_np)

            rval, frame = vc.read()
            c += 1

    video_writer.release()
    print('Save the result video to {0}'.format(result))

    st_progress_bar.empty()


