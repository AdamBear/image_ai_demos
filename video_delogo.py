import hashlib
import json
import os
import shutil
import time
from collections import Counter
import string
import config
import cv2
import requests
import unicodedata
from Levenshtein import ratio
from scenedetect.detectors import ContentDetector
from scenedetect import VideoManager
from scenedetect import SceneManager
from PIL import Image
import numpy as np
from image_fill import process_image


def post_to_recognize(image_file_list):
    url = "http://127.0.0.1:8868/predict/ocr_system"

    headers = {"Content-type": "application/json"}
    total_time = 0
    starttime = time.time()
    data = {'images': [], 'paths': image_file_list}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    elapse = time.time() - starttime
    total_time += elapse
    return r.json()


def get_hash(url):
    # encoding the string using encode()
    en = url.encode()
    # passing the encoded string to MD5
    hex_result = hashlib.md5(en)
    # printing the equivalent hexadecimal value
    return str(hex_result.hexdigest())


def is_chinese_char(uchar):
    if (uchar >= u'\u4e00' and uchar <= u'\u9fff'):
        return True
    else:
        return False


def is_all_chinese_char(sentence):
    for s in sentence:
        if not is_chinese_char(s):
            return False
    return True


def is_all_english_char(sentence):
    for s in sentence:
        if is_chinese_char(s):
            return False
    return True


def get_rec_area(rec):
    ymin = min(rec[2], rec[3])
    ymax = max(rec[3], rec[2])
    xmin = min(rec[0], rec[1])
    xmax = max(rec[1], rec[0])
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def add_mask(maskimg, area):
    for i in range(area[3]):
        for j in range(area[2]):
            maskimg[area[1] + i][area[0] + j] = 255
    return maskimg


def get_douyin_rec(rec, w, h):
    return (max(rec[0] - 70, 0), min(rec[1] + 60, w), max(rec[2] - 18, 0), min(rec[3] + 18, h))


def get_douyin_hao_rec(rec, w, h):
    return (max(rec[0] - 18, 0), min(rec[1] + 30, w), max(rec[2] - 18, 0), min(rec[3] + 18, h))


def grow_rec(rec, w, h, pad=10):
    return (max(rec[0] - pad, 0), min(rec[1] + pad, w), max(rec[2] - pad, 0), min(rec[3] + pad, h))


def save_mask(output_file, masking):
    cv2.imwrite(output_file, masking)


class AutoSubtitleExtractor():
    """
    视频字幕提取类
    """

    def __init__(self, vd_path, export_key_frames=False, detect_scene=True, start_ms=-1, end_ms=-1, generate=True,
                 model=None):
        self.sub_area = None
        self.export_key_frames = export_key_frames
        self.detect_scene = detect_scene

        self.debug = False
        self.remove_too_common = True
        self.detect_subtitle = True
        self.generate = generate
        if not generate:
            self.detect_scene = False
            self.export_key_frames = False

        self.start_ms = start_ms
        self.end_ms = end_ms
        self.model = model
        self.num_frame = 0

        # 字幕区域位置
        self.subtitle_area = config.SubtitleArea.LOWER_PART

        # 临时存储文件夹
        # self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output')
        self.temp_output_dir = os.path.join(config.TEMP_OUTPUT_DIR, get_hash(vd_path))

        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频秒数
        self.video_length = float(self.frame_count / self.fps)
        # 视频尺寸
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 字幕出现区域
        self.subtitle_area = config.SUBTITLE_AREA
        # 提取的视频帧储存目录
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # 提取的字幕文件存储目录
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # 定义vsf的字幕输出路径
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # 不存在则创建文件夹
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # 提取的原始字幕文本存储路径
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')

        # 处理进度
        self.progress = 0

        self.h = -1
        self.w = -1
        self.start_frame = -1
        self.end_frame = -1

        # 每帧毫秒数先计算出来
        self.ms_per_frame = 1000 / self.fps
        self.mask_cache = {}

    def run(self):
        """
        运行整个提取视频的步骤
        """
        self.extract_frame_by_fps()
        print("frame extracted!")

        self.progress = 0
        self.extract_subtitles()
        print("subtitle ocr complete!")

        if self.detect_scene:
            scenes_codes = self.find_scenes()
            self.scenes = [[] for i in range(len(scenes_codes))]
            print("scenes detecte complete!")

        # 将坐标区域内的文本和坐标进行记录统计
        self.coordinates_list, self.frame_contents, self.cord_frame_list = self._get_content_list()

        # 先检测并删除可能的台标或者背景，然后才能，所以需要考虑
        if self.remove_too_common:
            self._detect_too_common()

        # 检查水印区域，并处理区域合并
        self._detect_watermark_area()
        print("watermark detect complete!")

        if self.detect_subtitle:
            self.filter_scene_text()
            print("subtitle detect complete!")

        if self.generate:
            result = self.generate_subtitle_file()
            # 清理临时文件
            # self._delete_frame_cache()
            return result
        else:
            return True

    # done: 继续优化，为了兼容去台标处理，可以考虑不切分，而使用全OCR识别，字幕区域可以使用算法去除
    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快
        """
        # 删除缓存
        self.delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

        self.frame_key_frame = {}
        self.next_key_frame = {}
        last_key_frame = -1
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break

            frame_no += 1

            # 处理指定的提取范围
            if self.start_ms > 0:
                if self.ms_per_frame * frame_no < self.start_ms:
                    continue

            if self.end_ms > 0:
                if self.ms_per_frame * frame_no > self.end_ms:
                    break

            # 读取视频帧成功
            if self.h < 0:
                self.h, self.w = frame.shape[0], frame.shape[1]

            if self.start_frame < 0:
                self.start_frame = frame_no

            # 记录当前帧所对应的处理关键帧
            if last_key_frame > 0:
                self.next_key_frame[last_key_frame] = frame_no

            last_key_frame = frame_no
            self.frame_key_frame[frame_no] = last_key_frame
            self.num_frame += 1

            filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.jpg')

            # 保存截取的原视频帧
            cv2.imwrite(filename, frame)

            # 跳过剩下的帧
            for i in range(int(self.fps // config.EXTRACT_FREQUENCY) - 1):
                ret, _ = self.video_cap.read()
                if ret:
                    frame_no += 1
                    self.frame_key_frame[frame_no] = last_key_frame
                    self.num_frame += 1
                    # 更新进度条
                    self.progress = (frame_no / self.frame_count) * 100

        self.end_frame = frame_no
        self.video_cap.release()

    def extract_subtitles(self):
        """
        提取视频帧中的字幕信息，生成一个txt文件
        """
        global image_file_list

        # 删除缓存
        if os.path.exists(self.raw_subtitle_path):
            os.remove(self.raw_subtitle_path)
        # 新建文件
        f = open(self.raw_subtitle_path, mode='w+', encoding='utf-8')

        # 视频帧列表
        frame_list = [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]
        image_file_list = [os.path.join(self.frame_output_dir, i).replace("\\", "/") for i in frame_list]

        rec_result = post_to_recognize(image_file_list)

        for i, (frame, rec_ret) in enumerate(zip(frame_list, rec_result["results"])):
            dt_box = [r["text_region"] for r in rec_ret]
            rec_res = [(r["text"], r["confidence"]) for r in rec_ret]
            coordinates = self._get_coordinates(dt_box)

            # 写入返回结果
            for content, coordinate in zip(rec_res, coordinates):
                if content[1] > config.DROP_SCORE:
                    f.write(f'{os.path.splitext(frame)[0]}\t'
                            f'{coordinate}\t'
                            f'{content[0]}\n')
                    # 关闭文件
        f.close()

        if self.debug:
            shutil.copyfile(self.raw_subtitle_path, self.raw_subtitle_path + ".raw.txt")

    def find_scenes(self, threshold=30.0):
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold))

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Each returned scene is a tuple of the (start, end) timecode.
        return scene_manager.get_scene_list()

        # mask处理

    def _make_content_mask(self, maskimg, content, w, h, remove_text=True, remove_watermark=True, more_grow=False):
        pad = 50 if more_grow else 10
        # print(pad)
        for item in content:
            if remove_watermark:
                if "抖音\n" == item[1]:
                    maskimg = add_mask(maskimg, get_rec_area(get_douyin_rec(item[0], w, h)))
                if "抖音号：" == item[1][:4]:
                    maskimg = add_mask(maskimg, get_rec_area(get_douyin_hao_rec(item[0], w, h)))
                if item[3] in self.watermark_areas:
                    if item[0][2] <= self.sub_ymin and item[0][3] >= self.sub_ymax:
                        maskimg = add_mask(maskimg, get_rec_area(grow_rec(item[0], w, h, pad)))

            if remove_text:
                if item[0][2] > self.sub_ymin and item[0][3] < self.sub_ymax:
                    maskimg = add_mask(maskimg, get_rec_area(grow_rec(item[0], w, h, pad)))
        return maskimg

    # 获取可能水印台标遮罩
    def _get_frame_mask(self, frame_no, remove_text=True, remove_watermark=True):
        frame_contents = self.frame_contents
        w = self.w
        h = self.h

        pre_k = self.frame_key_frame[frame_no]
        if pre_k in self.next_key_frame:
            found_k = self.next_key_frame[pre_k]
        else:
            found_k = pre_k

        if pre_k in self.mask_cache:
            return self.mask_cache[pre_k]

        a_maskimg = np.zeros((h, w), dtype=np.uint8)

        # 两张全部加入到mask中
        if pre_k in frame_contents:
            pre_content = frame_contents[pre_k]
            a_maskimg = self._make_content_mask(a_maskimg, pre_content, w, h, remove_text, remove_watermark,
                                                found_k == pre_k)

            if found_k in frame_contents:
                next_content = frame_contents[found_k]
                a_maskimg = self._make_content_mask(a_maskimg, next_content, w, h, remove_text, remove_watermark,
                                                found_k == pre_k)

            self.mask_cache[pre_k] = a_maskimg

        return a_maskimg

    def remove_text_watermark(self, output_file=None, remove_text=True, remove_watermark=True, tqdm=None,
                              st_progress_bar=None):
        if not self.model:
            print("model not initialized!")
            return ""

        if not output_file:
            name = "fixed"
            if remove_text:
                name += "_detexted"
            if remove_watermark:
                name += "_dewatermark"
            name += ".mp4"
            output_file = os.path.join(self.temp_output_dir, name)

        output_file = ".".join(output_file.split(".")[:-1]) + "_wa.mp4"

        writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"x264"), self.fps, (self.w, self.h))

        video_cap = cv2.VideoCapture(self.video_path)
        frame_no = 0

        t = None
        if tqdm:
            if st_progress_bar:
                t = tqdm(range(int(self.num_frame)), st_progress_bar=st_progress_bar)
            else:
                t = tqdm(range(int(self.num_frame)))

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            else:
                frame_no += 1

                # 处理指定的提取范围
                if self.start_ms > 0:
                    if self.ms_per_frame * frame_no < self.start_ms:
                        continue

                if self.end_ms > 0:
                    if self.ms_per_frame * frame_no > self.end_ms:
                        break

                masking = self._get_frame_mask(frame_no, remove_text, remove_watermark)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                a_mask = Image.fromarray(np.array(masking))

                # todo: 此处不要再转两次，直接提供cv版的fill接口
                img = process_image(self.model, img, a_mask)

                #                 filename = os.path.join("e:/Temp_Output/test_filled", str(frame_no).zfill(8) + '.png')
                #                 img.save(filename)

                writer.write(cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB))
                if t:
                    t.update()

        writer.release()

        start_time = self._frame_to_timecode(self.start_frame - 1, smpte_token=".")
        end_time = self._frame_to_timecode(self.end_frame - 1, smpte_token=".")
        sourceVideo = self.video_path
        tempAudioFileName = os.path.join(self.temp_output_dir, "temp.aac")
        mp4_file = output_file
        output_file = mp4_file[:-len("_wa.mp4")] + ".mp4"

        os.system(f'ffmpeg -y -ss {start_time} -to {end_time} -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')
        os.system(f'ffmpeg -y -i "{mp4_file}" -i {tempAudioFileName} -c copy "{output_file}"')

        return output_file

    def generate_subtitle_file(self, srt_filename=None):
        """
        生成srt格式的字幕文件
        """
        subtitle_content = self._remove_duplicate_subtitle()

        if self.detect_scene:
            scenes_codes = self.find_scenes()
            self.scenes = [[] for i in range(len(scenes_codes))]

        if not srt_filename:
            srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        processed_subtitle = []

        with open(srt_filename, mode='w', encoding='utf-8') as f:
            for index, content in enumerate(subtitle_content):
                line_code = index + 1
                frame_no_start = int(content[0])
                frame_start = self._frame_to_timecode(frame_no_start)
                # 比较起始帧号与结束帧号， 如果字幕持续时间不足1秒，则将显示时间设为1s
                if abs(int(content[1]) - int(content[0])) < self.fps:
                    frame_no_end = int(int(content[0]) + self.fps)
                    frame_end = self._frame_to_timecode(frame_no_end)
                else:
                    frame_no_end = int(content[1])
                    frame_end = self._frame_to_timecode(frame_no_end)
                frame_content = content[2]
                processed_subtitle.append([frame_start, frame_end, frame_no_start, frame_no_end, frame_content])
                subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                f.write(subtitle_line)

                # 简单算法，按场景将字幕时间轴分组出来
                if self.detect_scene:
                    last_scene_no = 0
                    for i in range(last_scene_no, len(scenes_codes)):
                        s = scenes_codes[i]
                        if frame_no_start >= int(s[0]) and frame_no_end <= int(s[1]):
                            last_scene_no = i
                            if len(self.scenes[i]) > 0:
                                self.scenes[i][1] = frame_end
                                self.scenes[i][2].append(index)
                            else:
                                self.scenes[i] = [frame_start, frame_end, [index]]

        # 保存原始关键帧
        # 取得保存路径，拼到kfs目录下
        if self.export_key_frames:
            for f in processed_subtitle:
                org_filename = str(f[2]).zfill(8) + ".jpg"
                f.append(os.path.join(self.frame_output_dir, org_filename))
                f.append(org_filename)

        return processed_subtitle

    def _frame_preprocess(self, frame):
        """
        将视频帧进行裁剪
        """
        # 对于分辨率大于1920*1080的视频，将其视频帧进行等比缩放至1280*720进行识别
        # paddlepaddle会将图像压缩为640*640
        # if self.frame_width > 1280:
        #     scale_rate = round(float(1280 / self.frame_width), 2)
        #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
        cropped = int(frame.shape[0] // 2) - config.SUBTITLE_AREA_DEVIATION_PIXEL

        # 如果字幕出现的区域在下部分
        if self.subtitle_area == config.SubtitleArea.LOWER_PART:
            # 将视频帧切割为下半部分
            frame = frame[cropped:]
        # 如果字幕出现的区域在上半部分
        elif self.subtitle_area == config.SubtitleArea.UPPER_PART:
            # 将视频帧切割为下半部分
            frame = frame[:cropped]
        return frame

    def _frame_to_timecode(self, frame_no, smpte_token=','):
        """
        将视频帧转换成时间
        :param frame_no: 视频的帧号，i.e. 第几帧视频帧
        :returns: SMPTE格式时间戳 as string, 如'01:02:12:32' 或者 '01:02:12;32'
        """
        # 设置当前帧号
        # 直接通过fps计算出当前帧的毫秒数

        # cap = cv2.VideoCapture(self.video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        # cap.read()

        # 获取当前帧号对应的时间戳
        # milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        milliseconds = self.ms_per_frame * frame_no

        seconds = milliseconds // 1000
        milliseconds = int(milliseconds % 1000)
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
        if minutes >= 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)

        # cap.release()
        return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, milliseconds)

    def _remove_duplicate_subtitle(self):
        """
        读取原始的raw txt，去除重复行，返回去除了重复后的字幕列表
        """
        self._concat_content_with_same_frameno()

        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            content = line.split('\t')[2]
            # 只有一个字的一般是误识别，可以忽略
            if len(content) < 3:
                continue

            content_list.append((frame_no, content))

        # 循环遍历每行字幕，记录开始时间与结束时间
        index = 0
        # 去重后的字幕列表
        unique_subtitle_list = []
        for i in content_list:
            # TODO: 时间复杂度非常高，有待优化
            # 定义字幕开始帧帧号
            start_frame = i[0]
            for j in content_list[index:]:
                # 计算当前行与下一行的Levenshtein距离
                distance = ratio(i[1], j[1])
                if distance < config.THRESHOLD_TEXT_SIMILARITY or j == content_list[-1]:
                    # 定义字幕结束帧帧号
                    end_frame = content_list[content_list.index(j) - 1][0]
                    if end_frame == start_frame:
                        end_frame = j[0]
                    # 如果是第一行字幕，直接添加进列表
                    if len(unique_subtitle_list) < 1:
                        unique_subtitle_list.append((start_frame, end_frame, i[1]))
                    else:
                        string_a = unique_subtitle_list[-1][2].replace(' ', '').translate(
                            str.maketrans('', '', string.punctuation))
                        string_b = i[1].replace(' ', '').translate(str.maketrans('', '', string.punctuation))
                        similarity_ratio = ratio(string_a, string_b)
                        # 打印相似度
                        # print(f'{similarity_ratio}: {unique_subtitle_list[-1][2]} vs {i[1]}')
                        # 如果相似度小于阈值，说明该两行字幕不一样
                        if similarity_ratio < config.THRESHOLD_TEXT_SIMILARITY:
                            unique_subtitle_list.append((start_frame, end_frame, i[1]))
                        else:
                            # todo，相似的取出现次数更多的来保留!!!!!，现在的算法是有问题的。取长的是为了防止飞字，但是第一个容易误识别
                            # 如果大于阈值，但又不完全相同，说明两行字幕相似
                            # 可能出现以下情况: "但如何进人并接管上海" vs "但如何进入并接管上海"
                            # OCR识别出现了错误识别
                            if similarity_ratio < 1:
                                # TODO:
                                # 1) 取出两行字幕的并集
                                # 2) 纠错
                                # print(f'{round(similarity_ratio, 2)}, 需要手动纠错:\n {string_a} vs\n {string_b}')
                                # 保存较长的
                                if len(string_a) < len(string_b):
                                    unique_subtitle_list[-1] = (start_frame, end_frame, i[1])
                    index += 1
                    break
                else:
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()

        content_list = []
        frame_no_list = []
        contents = []
        for line in lines:
            frame_no = line.split('\t')[0]
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            contents.append(content)
            content_list.append([frame_no, coordinate, content])

            # 找出那些不止一行的帧号
        frame_no_list = [i[0] for i in Counter(frame_no_list).most_common() if i[1] > 1]

        # 找出这些帧号出现的位置
        concatenation_list = []
        for frame_no in frame_no_list:
            position = [i for i, x in enumerate(content_list) if x[0] == frame_no]
            concatenation_list.append((frame_no, position))

        for i in concatenation_list:
            content = []
            for j in i[1]:
                txt = content_list[j][2]
                txt = txt.replace(" ", "").replace('\n', '')
                # 全部是英文的不要处理，过短的不要处理，一般都是识别错误
                if not is_all_english_char(txt) and len(txt) > 2:
                    content.append(txt)

            content = ' '.join(content) + '\n'
            for k in i[1]:
                content_list[k][2] = content

        # 将多余的字幕行删除
        to_delete = []
        for i in concatenation_list:
            for j in i[1][1:]:
                to_delete.append(content_list[j])

        # 移出空白行
        for i in content_list:
            if len(i[2].replace("\n", "").strip()) == 0:
                to_delete.append(i)

        for i in to_delete:
            if i in content_list:
                content_list.remove(i)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                content = unicodedata.normalize('NFKC', content)
                f.write(f'{frame_no}\t{coordinate}\t{content}')

    # 需要返回特别重复的台标或水印位置，并保留后续处理
    # 结合前面的处理逻辑将，此函数需要进一步加工将可识别台标与水印标识出来
    def _detect_too_common(self):
        """
        将raw txt文本中具有相同帧号的字幕行合并
        # 通过重复计数，移除超过次数的文本，可能是台标
        """
        contents = []
        for frame_no in self.frame_contents:
            frame_content = self.frame_contents[frame_no]
            for item in frame_content:
                contents.append(item[1])

                # 总共出现5秒以上可以认为是台标或水印
        max_common_count = 5 * config.EXTRACT_FREQUENCY

        self.too_commons = set()
        self.counter = Counter(contents).most_common()
        for c in self.counter:
            if c[1] > max_common_count:
                self.too_commons.add(c[0])
            else:
                break

    def _get_content_list(self):
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        line = f.readline()  # 以行的形式进行读取文件
        # 坐标点列表
        coordinates_list = []
        frame_contents = {}
        cord_frame_list = []

        last_frame_no = -1
        while line:
            frame_no = int(line.split('\t')[0])
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]

            cord = (int(text_position[0]),
                    int(text_position[1]),
                    int(text_position[2]),
                    int(text_position[3]))

            cord_index = len(coordinates_list)
            coordinates_list.append(cord)

            if frame_no != last_frame_no:
                frame_contents[frame_no] = []
                last_frame_no = frame_no

            frame_contents[frame_no].append([cord, content, cord_index, None])
            cord_frame_list.append(frame_no)

            line = f.readline()
        f.close()
        return coordinates_list, frame_contents, cord_frame_list

    def _detect_watermark_area(self):
        """
        根据识别出来的raw txt文件中的坐标点信息，查找水印区域
        假定：水印区域（台标）的坐标在水平和垂直方向都是固定的，也就是具有(xmin, xmax, ymin, ymax)相对固定
        根据坐标点信息，进行统计，将一直具有固定坐标的文本区域选出
        :return 返回最有可能的水印区域
        """
        # 将坐标列表的相似值统一
        self._unite_coordinates()
        counter = Counter(self.coordinates_list).most_common()

        # 总共出现5秒以上可以认为是台标或水印
        max_common_count = 3 * config.EXTRACT_FREQUENCY

        # todo: 统计之后再标识出可能是字幕、水印或台标的坐标区域
        self.most_areas = counter
        self.watermark_areas = set()
        for c in counter:
            if c[1] >= max_common_count:
                self.watermark_areas.add(c[0])
            else:
                break

    # 重要的函数，统一后还需要在frame_contents里记录处理后的相似坐标位置
    def _unite_coordinates(self):
        """
        给定一个坐标列表，将这个列表中相似的坐标统一为一个值
        e.g. 由于检测框检测的结果不是一致的，相同位置文字的坐标可能一次检测为(255,123,456,789)，另一次检测为(253,122,456,799)
        因此要对相似的坐标进行值的统一
        :param coordinates_list 包含坐标点的列表
        :return: 返回一个统一值后的坐标列表
        """
        # 将相似的坐标统一为一个
        index = 0
        for coordinate in self.coordinates_list:  # TODO：时间复杂度n^2，待优化
            for i in self.coordinates_list:
                if self._is_coordinate_similar(coordinate, i):
                    self.coordinates_list[index] = i
                    frame_c = self.frame_contents[self.cord_frame_list[index]]
                    for item in frame_c:
                        if item[2] == index:
                            item[3] = i
            index += 1

    def _detect_subtitle_area(self):
        """
        读取过滤水印区域后的raw txt文件，根据坐标信息，查找字幕区域
        假定：字幕区域在y轴上有一个相对固定的坐标范围，相对于场景文本，这个范围出现频率更高
        :return 返回字幕的区域位置
        """
        # 打开去水印区域处理过的raw txt
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        line = f.readline()  # 以行的形式进行读取文件
        # y坐标点列表
        y_coordinates_list = []
        while line:
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]
            if content not in self.too_commons:
                y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
            line = f.readline()
        f.close()
        return Counter(y_coordinates_list).most_common(1)

    def filter_scene_text(self):
        # 加上字幕区域判断，可选的。
        subtitle_area = self._detect_subtitle_area()[0][0]

        # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
        ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
        ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL

        self.sub_ymin = ymin
        self.sub_ymax = ymax

        with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
            content = f.readlines()
            f.seek(0)
            for i in content:
                c = i.split('\t')[2]
                if c in self.too_commons:
                    continue
                i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                if ymin <= i_ymin and i_ymax <= ymax:
                    f.write(i)
            f.truncate()

    def _get_coordinates(self, dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def _is_coordinate_similar(self, coordinate1, coordinate2):
        """
        计算两个坐标是否相似，如果两个坐标点的xmin,xmax,ymin,ymax的差值都在像素点容忍度内
        则认为这两个坐标点相似
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[1] - coordinate2[1]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[2] - coordinate2[2]) < config.PIXEL_TOLERANCE_Y and \
               abs(coordinate1[3] - coordinate2[3]) < config.PIXEL_TOLERANCE_Y

    def delete_frame_cache(self):
        if len(os.listdir(self.frame_output_dir)) > 0:
            for i in os.listdir(self.frame_output_dir):
                os.remove(os.path.join(self.frame_output_dir, i))


# 封装给streamlit调用的函数入口
def process_delogo(model, video_path, remove_text=True, remove_watermark=True, tqdm=None, st_progress_bar=None):
    se = AutoSubtitleExtractor(video_path, True, start_ms=0, end_ms=10000, model=model, generate=False)
    se.detect_scene = False
    se.run()
    return se.remove_text_watermark(tqdm=tqdm, st_progress_bar=st_progress_bar, remove_text=remove_text,
                                    remove_watermark=remove_watermark)
