import os.path
import torch
import utils_image as util
import numpy as np
import cv2
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model2(model_name="BSRGANx2"):
    from network_rrdbnet import RRDBNet as net
    model_folder = "d:\\apps\\nlp\\prompt\\BSRGAN\\model_zoo\\"

    sf = 4
    if model_name in ['BSRGANx2']:
        sf = 2

    model_path = os.path.join(model_folder, model_name + '.pth')  # set model path

    # torch.cuda.set_device(0)      # set GPU ID
    torch.cuda.empty_cache()

    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    from network_swinir import SwinIR as net
    model_folder = "d:\\apps\\nlp\\prompt\\SwinIR\\model_zoo\\"
    large_model = False
    if not large_model:
        # use 'nearest+conv' to avoid block artifacts
        model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    else:
        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=248,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')

    param_key_g = 'params_ema'
    model_path = model_folder + "/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    model.eval()
    model = model.to(device)

    return model


def pil_to_cv2(im):
    # PIL RGB 'im' to CV2 BGR 'imcv'
    # imcv = np.asarray(im)[:, :, ::-1].copy()

    # # To gray image
    # imcv = np.asarray(im.convert('L'))
    # # Or
    # imcv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(img.astype(np.uint8))


def process_by_tile(model, img_lq, scale, tile, tile_overlap):
    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)

    sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
            W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
    output = E.div_(W)
    return output


def process_image_enhance(model, img, tile=720, tile_overlap=32):
    # test the image tile by tile

    # basewidth = 256
    # wpercent = (basewidth/float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    # img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    torch.cuda.empty_cache()

    window_size = 8
    scale = 4

    img_lq = pil_to_cv2(img)
    img_lq = img_lq.astype(np.float32) / 255.

    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()

        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = process_by_tile(model, img_lq, scale, tile, tile_overlap)
        output = output[..., :h_old * scale, :w_old * scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    return cv2_to_pil(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def process_image_enhance2(model, img):
    torch.cuda.empty_cache()

    #img_L = util.imread_uint(img, n_channels=3)
    img_L = pil_to_cv2(img)
    img_L = util.uint2tensor4(img_L)
    img_L = img_L.to(device)

    # --------------------------------
    # (2) inference
    # --------------------------------
    img_E = model(img_L)

    # --------------------------------
    # (3) img_E
    # --------------------------------
    img_E = util.tensor2uint(img_E)
    color_coverted = cv2.cvtColor(img_E, cv2.COLOR_BGR2RGB)
    img_E = Image.fromarray(color_coverted)
    return img_E

# def process_image_enhance2(model, img):
#     return process_image_enhance(model, img)