from __future__ import print_function

import sys
sys.path.insert(0, "d:\\apps\\nlp\\prompt\\Deep-Exemplar-based-Video-Colorization")

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.cuda.set_device(0)


def colorize_video(opt, input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    # processing folders
    mkdir_if_not(output_path)
    files = glob.glob(output_path + "*")
    print("processing the folder:", input_path)
    path, dirs, filenames = os.walk(input_path).__next__()
    file_count = len(filenames)
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

    # NOTE: resize frames to 216*384
    transform = transforms.Compose(
        [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    )

    # if frame propagation: use the first frame as reference
    # otherwise, use the specified reference image
    ref_name = input_path + filenames[0] if opt.frame_propagate else reference_file
    print("reference name:", ref_name)
    frame_ref = Image.open(ref_name)

    IB_lab, I_last_lab_predict, I_reference_lab, features_B = get_ref(frame_ref, transform, vggnet)

    for index, frame_name in enumerate(tqdm(filenames)):
        frame1 = Image.open(os.path.join(input_path, frame_name))

        IA_predict_rgb = image_colorize(IB_lab=IB_lab, I_last_lab_predict=I_last_lab_predict, I_reference_lab=I_reference_lab, colornet=colornet, features_B=features_B, frame1=frame1,
                                        lambda_value=lambda_value, nonlocal_net=nonlocal_net, opt=opt, sigma_color=sigma_color, transform=transform, vggnet=vggnet, wls_filter_on=wls_filter_on)

        # save the frames
        save_frames(IA_predict_rgb, output_path, index)

    # output video
    video_name = "video.avi"
    folder2vid(image_folder=output_path, output_dir=output_path, filename=video_name)
    print()


def get_ref(frame_ref, transform, vggnet):
    I_last_lab_predict = None
    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
    return IB_lab, I_last_lab_predict, I_reference_lab, features_B


def image_colorize(IB_lab, I_last_lab_predict, I_reference_lab, colornet, features_B, frame1, lambda_value,
                   nonlocal_net, opt, sigma_color, transform, vggnet, wls_filter_on):

    #IA_lab_large = transform(frame1).unsqueeze(0).cuda()
    IA_lab_large = transform(frame1).unsqueeze(0)
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
    IA_l = IA_lab[:, 0:1, :, :]
    IA_ab = IA_lab[:, 1:3, :, :]
    if I_last_lab_predict is None:
        if opt.frame_propagate:
            I_last_lab_predict = IB_lab
        else:
            #I_last_lab_predict = torch.zeros_like(IA_lab).cuda()
            I_last_lab_predict = torch.zeros_like(IA_lab)
    # start the frame colorization
    with torch.no_grad():
        I_current_lab = IA_lab
        I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
            I_current_lab,
            I_reference_lab,
            I_last_lab_predict,
            features_B,
            vggnet,
            nonlocal_net,
            colornet,
            feature_noise=0,
            temperature=1e-10,
        )
        I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)
    # upsampling
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
    )
    # filtering
    if wls_filter_on:
        guide_image = uncenter_l(curr_bs_l) * 255 / 100
        wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
        )
        curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
        curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
        curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
        curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
        curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
        IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
    else:
        IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])
    return IA_predict_rgb


def get_model():
    import sys
    sys.path.insert(0, "d:\\apps\\nlp\\prompt\\Deep-Exemplar-based-Video-Colorization")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
    )
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
    #parser.add_argument("--clip_path", type=str, default="./sample_videos/clips/v32", help="path of input clips")
    #parser.add_argument("--ref_path", type=str, default="./sample_videos/ref/v04", help="path of refernce images")
    #parser.add_argument("--output_path", type=str, default="./sample_videos/output", help="path of output clips")
    opt, _ = parser.parse_known_args()

    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    # clip_name = opt.clip_path.split("/")[-1]
    # refs = os.listdir(opt.ref_path)
    # refs.sort()

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("d:\\apps\\nlp\\prompt\\Deep-Exemplar-based-Video-Colorization\\data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("d:\\apps\\nlp\\prompt\\Deep-Exemplar-based-Video-Colorization\\checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("d:\\apps\\nlp\\prompt\\Deep-Exemplar-based-Video-Colorization\\checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    #nonlocal_net.cuda()
    #colornet.cuda()
    #vggnet.cuda()

    return (opt, nonlocal_net, colornet, vggnet)


def process_image_colorization(models, frame1, frame_ref):
    wls_filter_on = False
    lambda_value = 500
    sigma_color = 4

    opt, nonlocal_net, colornet, vggnet = models
    transform_t = transforms.Compose(
        [CenterPad(frame1.size), transform_lib.CenterCrop(frame1.size), RGB2Lab(), ToTensor(), Normalize()]
    )

    IB_lab, I_last_lab_predict, I_reference_lab, features_B = get_ref(frame_ref, transform_t, vggnet)
    IA_predict_rgb = image_colorize(IB_lab=IB_lab, I_last_lab_predict=I_last_lab_predict, I_reference_lab=I_reference_lab, colornet=colornet, features_B=features_B, frame1=frame1,
                                        lambda_value=lambda_value, nonlocal_net=nonlocal_net, opt=opt, sigma_color=sigma_color, transform=transform_t, vggnet=vggnet, wls_filter_on=wls_filter_on)
    return Image.fromarray(IA_predict_rgb.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
    )
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpu_ids", type=str, default="1", help="separate by comma")
    parser.add_argument("--clip_path", type=str, default="./sample_videos/clips/v32", help="path of input clips")
    parser.add_argument("--ref_path", type=str, default="./sample_videos/ref/v32", help="path of refernce images")
    parser.add_argument("--output_path", type=str, default="./sample_videos/output", help="path of output clips")
    opt = parser.parse_args()
    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    clip_name = opt.clip_path.split("/")[-1]
    refs = os.listdir(opt.ref_path)
    refs.sort()

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()

    for ref_name in refs:
        try:
            colorize_video(
                opt,
                opt.clip_path,
                os.path.join(opt.ref_path, ref_name),
                os.path.join(opt.output_path, clip_name + "_" + ref_name.split(".")[0]),
                nonlocal_net,
                colornet,
                vggnet,
            )
        except Exception as error:
            print("error when colorizing the video " + ref_name)
            print(error)

    # video_name = "video.avi"
    # clip_output_path = os.path.join(opt.output_path, clip_name)
    # mkdir_if_not(clip_output_path)
    # folder2vid(image_folder=opt.clip_path, output_dir=clip_output_path, filename=video_name)
