from dotenv import load_dotenv
import os
import sys
import sys, os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from lab.oss_storage import http_get_file, obs_storage, get_hash


load_dotenv()
not_local_test = os.getenv('LOCAL_TEST') == "false"

if not_local_test:
    sys.path.insert(0, "/mnt/RobustVideoMatting/")
    save_path = "/data/video/"
    PREDICTOR_PATH = '/mnt/FaceSwap/models/shape_predictor_68_face_landmarks.dat'
    xiuke_path = "/data/video/xiuke/"
    syn_path = "/data/video/xk_syn/"
else:
    sys.path.insert(0, "d:\\apps\\nlp\\prompt\\RobustVideoMatting\\")
    save_path = "e:/video/"
    PREDICTOR_PATH = 'd:\\apps\\nlp\\prompt\\FaceSwap\\models\\shape_predictor_68_face_landmarks.dat'
    xiuke_path = "d:/"
    syn_path = "d:/"


syn_path = "/data/video/xk_syn/"
obs_syn_path = "xiuke/syn/"
xiuke_path = "/data/video/xiuke"

WAV2LIP_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/wav2lip_hq.pdparams'
mel_step_size = 16



import scipy.spatial as spatial
import logging
import dlib

from os import listdir, path, makedirs
import platform
import numpy as np
import cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from PIL import Image
from pyvad import split


import os
from pydub import AudioSegment

import paddle
from paddle.utils.download import get_weights_path_from_url
from ppgan.faceutils import face_detection
from ppgan.utils import audio
from ppgan.models.generators.wav2lip import Wav2Lip

initialized, wav2lip_predictor, two_sec_segment, rvm_model, predictor = False, None, None, None, None


def init():
    global initialized
    if initialized:
        return

    global wav2lip_predictor, rvm_model, predictor, two_sec_segment

    two_sec_segment = AudioSegment.silent(duration=1000)
    wav2lip_predictor = Wav2LipPredictor(static = False,
                                        face_enhancement = True, resize_factor=1)

    predictor = dlib.shape_predictor(PREDICTOR_PATH)


    import torch
    rvm_model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    rvm_model = rvm_model.eval().cuda()
    initialized = True

## Face detection
def face_detection2(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces


## Face and points detection
def face_points_detection(img, bbox:dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords


def select_face(im, r=10, choose=False):
    faces = face_detection2(im)

    if len(faces) == 0:
        return None, None, None

    if len(faces) == 1 or not choose:
        idx = np.argmax([(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces])
        bbox = faces[idx]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        #cv2.imshow('Click the Face:', im_copy)
        #cv2.setMouseCallback('Click the Face:', click_on_face)
        #while len(bbox) == 0:
        #    cv2.waitKey(1)
        #cv2.destroyAllWindows()
        #bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]


def select_all_faces(im, r=10):
    faces = face_detection2(im)

    if len(faces) == 0:
        return None

    faceBoxes = {k : {"points" : None,
                      "shape" : None,
                      "face" : None} for k in range(len(faces))}
    for i, bbox in enumerate(faces):
        points = np.asarray(face_points_detection(im, bbox))

        im_w, im_h = im.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)

        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y
        faceBoxes[i]["points"] = points - np.asarray([[x, y]])
        faceBoxes[i]["shape"] = (x, y, w, h)
        faceBoxes[i]["face"] = im[y:y + h, x:x + w]

    return faceBoxes


## 3D Transform
def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img


## 2D Transform
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack([s2 / s1 * R,
                                (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                      np.array([[0., 0., 1.]])])


def warp_image_2d(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im


## Generate Mask
def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


## Color Correction
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


## Alpha blending
def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

    return result_img


def check_points(img,points):
    # Todo: I just consider one situation.
    if points[8,1]>img.shape[0]:
        logging.error("Jaw part out of image")
    else:
        return True
    return False


def face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, correct_color=True, warp_2d=True, end=48):
    h, w = dst_face.shape[:2]

    ## 3d warp
    warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h, w))
    ## Mask for blending
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)
    ## Correct color
    if correct_color:
        warped_src_face = apply_mask(warped_src_face, mask)
        dst_face_masked = apply_mask(dst_face, mask)
        warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
    ## 2d warp
    if warp_2d:
        unwarped_src_face = warp_image_3d(warped_src_face, dst_points[:end], src_points[:end], src_face.shape[:2])
        warped_src_face = warp_image_2d(unwarped_src_face, transformation_from_points(dst_points, src_points),
                                        (h, w, 3))

        mask = mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)

    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output

    return dst_img_cp

def get_cv2_img(data):
    return Image.fromarray(cv2.cvtColor(data,cv2.COLOR_BGR2RGB))


def process_segs(segs, fps):
    new_segs = []
    for i in range(len(segs)):
        seg = segs[i].copy()
        # always prcess the end of wav
        if i == len(segs) - 1:
            seg[0] = seg[0] / 16000 * fps
            seg[1] = seg[1] / 16000 * fps
        else:
            seg[1] = seg[1] / 16000 * fps
            # ignore too short interval
            if segs[i + 1][0] / 16000 * fps - seg[1] < 15:
                seg[1] = segs[i + 1][0] / 16000 * fps

            seg[0] = seg[0] / 16000 * fps

        new_segs.append(seg)
    return new_segs


class BasePredictor(object):
    def __init__(self):
        pass

    def build_inference_model(self):
        if paddle.in_dynamic_mode():
            # todo self.model = build_model(self.cfg)
            pass
        else:
            place = paddle.get_device()
            self.exe = paddle.static.Executor(place)
            file_names = os.listdir(self.weight_path)
            for file_name in file_names:
                if file_name.find('model') > -1:
                    model_file = file_name
                elif file_name.find('param') > -1:
                    param_file = file_name

            self.program, self.feed_names, self.fetch_targets = paddle.static.load_inference_model(
                self.weight_path,
                executor=self.exe,
                model_filename=model_file,
                params_filename=param_file)

    def base_forward(self, inputs):
        if paddle.in_dynamic_mode():
            out = self.model(inputs)
        else:
            feed_dict = {}
            if isinstance(inputs, dict):
                feed_dict = inputs
            elif isinstance(inputs, (list, tuple)):
                for i, feed_name in enumerate(self.feed_names):
                    feed_dict[feed_name] = inputs[i]
            else:
                feed_dict[self.feed_names[0]] = inputs

            out = self.exe.run(self.program,
                               fetch_list=self.fetch_targets,
                               feed=feed_dict)

        return out

    def is_image(self, input):
        try:
            if isinstance(input, (np.ndarray, Image.Image)):
                return True
            elif isinstance(input, str):
                if not os.path.isfile(input):
                    raise ValueError('input must be a file')
                img = Image.open(input)
                _ = img.size
                return True
            else:
                return False
        except:
            return False

    def run(self):
        raise NotImplementedError


class Wav2LipPredictor(BasePredictor):
    def __init__(self, checkpoint_path=None,
                 static=False,
                 fps=25,
                 pads=[0, 10, 0, 0],
                 face_det_batch_size=16,
                 wav2lip_batch_size=128,
                 resize_factor=1,
                 crop=[0, -1, 0, -1],
                 box=[-1, -1, -1, -1],
                 rotate=False,
                 nosmooth=False,
                 face_detector='sfd',
                 face_enhancement=False
                 ):
        self.img_size = 96
        self.checkpoint_path = checkpoint_path
        self.static = static
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        self.face_detector = face_detector
        self.face_enhancement = face_enhancement
        if face_enhancement:
            from ppgan.faceutils.face_enhancement import FaceEnhancement
            self.faceenhancer = FaceEnhancement()
        makedirs('./temp', exist_ok=True)

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i:i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector)

        batch_size = self.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(
                        detector.get_detections_for_batch(
                            np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument'
                    )
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(
                    batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite(
                    'temp/faulty_frame.jpg',
                    image)  # check this frame where the face was not detected.
                raise ValueError(
                    'Face not detected! Ensure the video contains a face in all the frames.'
                )

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                   for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(
                    frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print(
                'Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                                for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)

            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(
                    mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate(
                    (img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch,
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def run(self, face, audio_seq, outfile, use_close_mouth_face=False, close_mouth_face_file=""):
        global first_frame
        if os.path.isfile(face) and path.basename(
                face).split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.static = True

        if not os.path.isfile(face):
            raise ValueError(
                '--face argument must be a valid path to video/image file')

        elif path.basename(
                face).split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face)]
            fps = self.fps

        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            first_frame = None
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(
                        frame, (frame.shape[1] // self.resize_factor,
                                frame.shape[0] // self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                if len(full_frames) == 0:
                    first_frame = frame.copy()

                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        print("Number of frames available for inference: " +
              str(len(full_frames)))

        if not audio_seq.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(
                audio_seq, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            audio_seq = 'temp/temp.wav'

        close_mouth_face_point = None
        close_mouth_face = None

        if close_mouth_face_file:
            use_close_mouth_face = True
            src_img = cv2.imread(close_mouth_face_file)
        else:
            if use_close_mouth_face:
                src_img = first_frame

        # Select src face
        if use_close_mouth_face:
            close_mouth_face_point, src_shape, close_mouth_face = select_face(src_img)
            if not src_shape:
                return {"success": False, "msg": "can not detect face!"}

        wav = audio.load_wav(audio_seq, 16000)

        self.segs = process_segs(split(wav, 16000, 16000, hop_length=10, vad_mode=2), fps)

        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again'
            )

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            # for slow the mel movment
            if i > 0 and i % 2 == 0:
                start_idx = start_idx - 1
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        model = Wav2Lip()
        if self.checkpoint_path is None:
            model_weights_path = get_weights_path_from_url(WAV2LIP_WEIGHT_URL)
            weights = paddle.load(model_weights_path)
        else:
            weights = paddle.load(self.checkpoint_path)
        model.load_dict(weights)
        model.eval()
        print("Model loaded")

        cur_frame = 0
        for i, (img_batch, mel_batch, frames, coords) in enumerate(
                tqdm(gen,
                     total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi',
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                      (frame_w, frame_h))

            img_batch = paddle.to_tensor(np.transpose(
                img_batch, (0, 3, 1, 2))).astype('float32')
            mel_batch = paddle.to_tensor(np.transpose(
                mel_batch, (0, 3, 1, 2))).astype('float32')

            with paddle.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.numpy().transpose(0, 2, 3, 1) * 255.

            last_f = []

            for p, f, c in zip(pred, frames, coords):
                cur_frame += 1
                y1, y2, x1, x2 = c

                need_process = False
                if use_close_mouth_face:
                    for seg in self.segs:
                        if cur_frame >= seg[0] and cur_frame <= seg[1]:
                            need_process = True
                            break
                else:
                    need_process = True

                replace_last = False
                if not need_process:
                    if replace_last:
                        if len(last_f) > 0:
                            f = last_f
                    else:
                        target_face = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = target_face

                        dst_faceBoxes = select_all_faces(f)
                        for k, dst_face in dst_faceBoxes.items():
                            f = face_swap(close_mouth_face, dst_face["face"], close_mouth_face_point,
                                          dst_face["points"], dst_face["shape"],
                                          f)
                            break

                        p = f[y1:y2, x1:x2]
                        p = cv2.resize(p.astype(np.uint8), (512, 512))
                        p = paddle.to_tensor(np.transpose(p, (2, 0, 1))).astype('float32')
                        if self.face_enhancement:
                            p = self.faceenhancer.enhance_from_image(p)
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                        f[y1:y2, x1:x2] = p

                    out.write(f)
                    continue

                if self.face_enhancement:
                    p = self.faceenhancer.enhance_from_image(p)

                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                # f_org = f.copy()

                f[y1:y2, x1:x2] = p
                last_f = f.copy()

                out.write(f)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2  -r 25 -q:v 1 {}'.format(
            audio_seq, 'temp/result.avi', outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')
        return {"success": True, "result": outfile}



def wav2lip(input_video,input_audio,output, use_close_mouth_face = False, close_mouth_face_file = ""):
    ret = wav2lip_predictor.run(input_video, input_audio, output, use_close_mouth_face, close_mouth_face_file)
    return ret


def get_mask_mp4(input_mp4, foreground=False):
    from inference import convert_video
    foreground_mp4 = None

    path, file = os.path.split(input_mp4)
    mask_mp4 = os.path.join(path, file+".alpha.mp4")
    if foreground:
        foreground_mp4 = os.path.join(path, file+".cut_out.mp4")

    convert_video(
        rvm_model,                     # The model, can be on any device (cpu or cuda).
        input_source=input_mp4,        # A video file or an image sequence directory.
        output_type='video',             # Choose "video" or "png_sequence"
        output_composition=foreground_mp4,    # File path if video; directory path if png sequence.
        output_alpha=mask_mp4,          # [Optional] Output the raw alpha prediction.
        output_foreground=None,     # [Optional] Output the raw foreground prediction.
        output_video_mbps=0.5,             # Output video mbps. Not needed for png sequence.
        downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
        seq_chunk=12,                    # Process n frames at once for better parallelism.
    )
    return mask_mp4, foreground_mp4


def merge_wavs(wavs, output_wav, debug=False):
    temp_path = "/tmp_output/wavs"
    is_begin = True
    final_wav = None
    total = 0
    for wav_url in wavs:
        wav_file = wav_url.split("/")[-1]
        wav_file_path = os.path.join(temp_path, wav_file)
        http_get_file(wav_url, wav_file_path)
        wav = AudioSegment.from_wav(wav_file_path)
        if debug:
            print(wav_file, len(wav), total)
        os.remove(wav_file_path)
        if is_begin:
            total = len(wav)
            final_wav = wav
            is_begin = False
        else:
            total += len(wav) + 1000
            final_wav = final_wav + two_sec_segment + wav

        # add tail silence
        #final_wav = final_wav
        final_wav.export(output_wav, format="wav")

    return output_wav


def append_tail_silence(wav_file_path):
    wav = AudioSegment.from_wav(wav_file_path)
    os.remove(wav_file_path)
    final_wav = wav + two_sec_segment
    final_wav.export(wav_file_path, format="wav")


def get_task_wav(task_url, wav_file):
    task_name = get_hash(task_url)
    json_file = task_name + ".json"

    json_file_path = os.path.join(os.path.split(wav_file)[0], json_file)
    http_get_file(task_url, json_file_path)
    with open(json_file_path, "r") as f:
        json_data = json.loads(f.read())
        f.close()
    contents = json_data["frames"]["content"]
    wavs = [i["introduction"]["audio_url"] for i in contents]
    merge_wavs(wavs, wav_file)
    return wav_file

import time


def get_day_path():
    localtime = time.strftime('%Y_%m_%d', time.localtime(time.time()))
    cur_save_path = os.path.join(syn_path, localtime + "/")
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
    return cur_save_path, localtime


def get_xiuke_video_file(xiuke_url):
    xiuke_file_name = get_hash(xiuke_url)
    xiuke_file_name = os.path.join(xiuke_path, xiuke_file_name + ".mp4")
    http_get_file(xiuke_url, xiuke_file_name)
    return xiuke_file_name


def cut_out(video_url, only_mask=True):
    init()
    day_path, localtime = get_day_path()

    input_mp4 = get_xiuke_video_file(video_url)

    mask_file, output_mp4 = get_mask_mp4(input_mp4, not only_mask)

    file_name = os.path.split(input_mp4)[1]

    if output_mp4:
        ret = obs_storage(obs_syn_path + localtime + "/" + file_name, output_mp4)
        if not ret["success"]:
            return ret
        output_mp4_url = ret["url"]

    ret = obs_storage(obs_syn_path + localtime + "/" + file_name + ".alpha.mp4", mask_file)
    if not ret["success"]:
        return ret
    mask_file_url = ret["url"]

    return {"success": True, "result": {"cut_out": output_mp4_url, "mask": mask_file_url}}


def make_xiuke(task_or_wav_url, xiuke_idx, xiuke_url=None):
    init()
    day_path, localtime = get_day_path()
    wav_file_name = get_hash(task_or_wav_url)
    wav_file_path = os.path.join(day_path, wav_file_name)

    if not os.path.isfile(wav_file_path):
        if task_or_wav_url[-4:] == ".wav":
            http_get_file(task_or_wav_url, wav_file_path)
        else:
            get_task_wav(task_or_wav_url, wav_file_path)
        if not os.path.isfile(wav_file_path):
            return {"success": False, "msg": "download task or wav file failed!"}

    append_tail_silence(wav_file_path)

    syn_name = str(xiuke_idx) + "_" + wav_file_name + ".mp4"

    if xiuke_url:
        input_mp4 = get_xiuke_video_file(xiuke_url)
    else:
        input_mp4 = os.path.join(xiuke_path, str(xiuke_idx) + ".mp4")
    output_mp4 = os.path.join(day_path, syn_name)
    ret = wav2lip(input_mp4, wav_file_path, output_mp4, True)
    if not ret["success"]:
        return ret
    output_mp4 = ret["result"]

    mask_file, _ = get_mask_mp4(output_mp4)

    ret = obs_storage(obs_syn_path + localtime + "/" + syn_name, output_mp4)
    if not ret["success"]:
        return ret
    output_mp4_url = ret["url"]

    ret = obs_storage(obs_syn_path + localtime + "/" + syn_name + ".alpha.mp4", mask_file)
    if not ret["success"]:
        return ret
    mask_file_url = ret["url"]

    return {"success": True, "result": {"xiuke": output_mp4_url, "mask": mask_file_url}}



if __name__ == '__main__':
    make_xiuke("https://huidouxiao.qyt.com/capi/video/params?id=62306e010390117af5ce4416", 3347)
