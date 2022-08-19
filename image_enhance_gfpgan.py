import sys
import os

gfpgan_path = "/python/GFPGAN/"
sys.path.insert(0, gfpgan_path)

realesrganer_model_path = os.path.join(gfpgan_path, 'experiments/pretrained_models/RealESRGAN_x2plus.pth')
gfpganer_model_path = os.path.join(gfpgan_path, 'experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
frames_path = "./upload"

from basicsr.utils import imwrite

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import glob

from gfpgan.utils import GFPGANer

arch = 'clean'
channel_multiplier = 2

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


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    size2 = (0, 0)

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        size2 = size
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size2)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def convert_video_to_frames(f, frames_path):
    # video to frames
    cam = cv2.VideoCapture(str(f))

    try:
        # PATH TO STORE VIDEO FRAMES
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (True):
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = os.path.join(frames_path, 'frame' + str(currentframe).zfill(8) + '.jpg')

            # writing the extracted images
            cv2.imwrite(name, frame)
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            # print(currentframe)
        else:
            break

    # Release all space and windows once done
    cam.release()


def merge_audio(video_source_file, audio_source_file, target_file):
    cmd = f"ffmpeg -i {video_source_file} -i {audio_source_file}  -c copy -map 0 -map 1:1 -y -shortest {target_file}"
    ret = os.system(cmd)
    if ret == 0:
        return target_file
    else:
        return ""


def restore_video(video_file, result_folder, fps=25):
    convert_video_to_frames(video_file, result_folder)
    img_list = sorted(glob.glob(os.path.join(result_folder, '*')))
    for img_path in img_list:
        restore_image(img_path, result_folder)

    target_video_file = video_file + ".sr_.mp4"
    target_mp4_file = video_file + ".sr.mp4"
    convert_frames_to_video(os.path.join(result_folder, "restored_imgs"), target_video_file, fps)
    merge_audio(target_video_file, video_file, target_mp4_file)
    return target_mp4_file