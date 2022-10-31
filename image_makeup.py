import sys
import os
spmt_path = "d:\\apps\\nlp\\prompt\\SpMt\\"
sys.path.insert(0,  spmt_path)
model_path = os.path.join(spmt_path, "image_models/networks/face_parsing/79999_iter.pth")

import os
import time
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from options.demo_options import DemoOptions
from image_models.pix2pix_model import Pix2PixModel
from image_models.networks.sync_batchnorm import DataParallelWithCallback
from image_models.networks.face_parsing.parsing_model import BiSeNet

opt = DemoOptions().parse()
opt.checkpoints_dir = os.path.join(spmt_path, "checkpoints")

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def get_mk_model():
    model = Pix2PixModel(opt)

    if len(opt.gpu_ids) > 0:
        model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
    model.eval()

    opt.beyond_mt = True

    n_classes = 19
    parsing_net = BiSeNet(n_classes=n_classes)
    parsing_net.load_state_dict(torch.load(model_path))
    parsing_net.eval()
    for param in parsing_net.parameters():
        param.requires_grad = False
    return model, parsing_net


def tensor_2_im(data):
    grid = make_grid(data, nrow=1)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def do_makeup(c, s, degree, model, parsing_net):
    height, width = c.size[0], c.size[1]
    c_m = c.resize((512, 512))
    s_m = s.resize((512, 512))
    c = c.resize((256, 256))
    s = s.resize((256, 256))
    print(c.size)
    c_tensor = trans(c).unsqueeze(0)
    s_tensor = trans(s).unsqueeze(0)
    c_m_tensor = trans(c_m).unsqueeze(0)
    s_m_tensor = trans(s_m).unsqueeze(0)

    x_label = parsing_net(c_m_tensor)[0]
    y_label = parsing_net(s_m_tensor)[0]
    x_label = F.interpolate(x_label, (256, 256), mode='bilinear', align_corners=True)
    y_label = F.interpolate(y_label, (256, 256), mode='bilinear', align_corners=True)
    x_label = torch.softmax(x_label, 1)
    y_label = torch.softmax(y_label, 1)

    nonmakeup_unchanged = (
                x_label[0, 0, :, :] + x_label[0, 4, :, :] + x_label[0, 5, :, :] + x_label[0, 11, :, :] + x_label[0, 16,
                                                                                                         :,
                                                                                                         :] + x_label[0,
                                                                                                              17, :,
                                                                                                              :]).unsqueeze(
        0).unsqueeze(0)
    makeup_unchanged = (
                y_label[0, 0, :, :] + y_label[0, 4, :, :] + y_label[0, 5, :, :] + y_label[0, 11, :, :] + y_label[0, 16,
                                                                                                         :,
                                                                                                         :] + y_label[0,
                                                                                                              17, :,
                                                                                                              :]).unsqueeze(
        0).unsqueeze(0)
    print(c_tensor.shape, s_tensor.shape)
    print(x_label.shape, y_label.shape)
    input_dict = {'nonmakeup': c_tensor,
                  'makeup': s_tensor,
                  'label_A': x_label,
                  'label_B': y_label,
                  'makeup_unchanged': makeup_unchanged,
                  'nonmakeup_unchanged': nonmakeup_unchanged
                  }

    time_start = time.time()

    synthetic_image = model([input_dict], mode='inference', alpha=degree)

    time_end = time.time()
    print(time_end - time_start)

    out = denorm(synthetic_image[0])
    out = F.interpolate(out, (361, 361 * height // width), mode='bilinear', align_corners=False)

    return tensor_2_im(out)
