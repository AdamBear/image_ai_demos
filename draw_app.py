import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import jsons

from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
import numpy as np
import tempfile
import streamlit_parameters
import logging

parameters = streamlit_parameters.parameters.Parameters()
parameters.register_string_list_parameter(key="action", default_value=["makeup", "beauty"])

parameters.set_url_fields()
action = parameters.action.value[0]
logging.info("action:" + action)

class tqdm:
    def __init__(self, iterable, st_progress_bar):
        self.prog_bar = st_progress_bar.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        return False

    def update(self):
        self.i += 1
        current_prog = self.i / self.length
        self.prog_bar.progress(current_prog)


# 所有的demo都需要转换成接口，其中的resize方法原来改进，最终需要出原图比例
st.set_page_config(
     page_title="慧抖销工具服务演示",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "#慧抖销工具服务，厦门书生企友通科技"
     }
 )

st.header("慧抖销AI处理功能演示")

base_demo, person_demo = False, False

beauty_index = 0
if action == "beauty":
    beauty_index = 0
elif action == "makeup":
    beauty_index = 1

# # 1. as sidebar menu
with st.sidebar:
    # demo_s = st.radio("请选择演示功能分类", ("基础AI", "音频语音", "虚拟主播"))
    demo_s = st.radio("请选择演示功能分类", ("虚拟主播", "基础AI"))

    # with cols[1]:
    #     if st.button("秀客AI"):
    #         person_demo = True

    if not demo_s or demo_s == "基础AI":
        selected = option_menu("基础AI", ["图片修复", '智能抠图', '图片超分',
                                        '视频调色', '---',
                                        '两图转视频',
                                        '视频补帧',
                                        '视频抠图',
                                        # '视频超分',
                                        '---', '字幕水印去除'],
            icons=['bandaid', 'heart', 'house',
                   'palette', '',
                   'images',
                   'collection-play-fill',
                   'person-video2',
                   # 'file-earmark-richtext',
                   '', 'eyedropper'],
                  menu_icon="cast", default_index=0)

    elif demo_s == "音频语音":
        selected = option_menu("音频语音", ['英文一句话语音克隆', '中文语音克隆', '---', '视频自动配乐', '人声音乐分离'],
                               icons=['soundwave', 'soundwave', '', 'music-note-beamed', 'play-btn-fill'],
                               menu_icon="cast", default_index=0)

    elif demo_s == "虚拟主播":
        selected = option_menu("虚拟主播", ['人脸美化', '装容美化', '动作识别', '视频换脸'],
                               icons=['emoji-laughing-fill', 'person-square', 'person-lines-fill', 'person-x-fill'],
                               menu_icon="cast", default_index=beauty_index)
    else:
        selected = ""
        st.text("正在赶工中...")

st.subheader(selected)

# selected2 = option_menu(None, ["主页", "图片上传", "任务", '设置'],
#     icons=['house', 'cloud-upload', "list-task", 'gear'],
#     menu_icon="cast", default_index=0, orientation="horizontal")

# Specify canvas parameters in application
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_fill_model():
    from image_fill import get_model
    return get_model()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_hub():
    from image_beauty import get_hub_module
    return get_hub_module()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_makeup_model():
    from image_makeup import get_mk_model
    return get_mk_model()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_mate_model():
    from image_mate import get_model
    return get_model()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_video_mate_model():
    from video_mate import get_model
    return get_model()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_enhance_model():
    from image_enhance import get_model, get_model2
    return get_model()


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_colorization_model():
    from image_colorization import get_model
    return get_model()


#########################
@st.cache(show_spinner=False, allow_output_mutation=True)
def get_fill_demos():
    with open("demo_mask.json", "r") as f:
        return jsons.loads(f.read())


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_mate_demos():
    with open("demo_mate.json", "r") as f:
        return jsons.loads(f.read())


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_enhance_demos():
    with open("demo_enhance.json", "r") as f:
        return jsons.loads(f.read())

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_colorize_demos():
    with open("demo_colorization.json", "r") as f:
        return jsons.loads(f.read())

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_motion_demos():
    with open("demo_motion.json", "r") as f:
        return jsons.loads(f.read())


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_movie_IF_demos():
    with open("demo_if.json", "r") as f:
        return jsons.loads(f.read())


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_movie_mate_demos():
    with open("demo_video_mate.json", "r") as f:
        return jsons.loads(f.read())


@st.cache(show_spinner=False, allow_output_mutation=True)
def get_video_delogo_demos():
    with open("demo_video_delogo.json", "r") as f:
        return jsons.loads(f.read())


max_size = 720


# 将图片绽放到固定长宽比
def resize_image(image, max_size=720):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = (max_size, max_size)  # 目标图像的尺寸

    print("original size: ", (iw, ih))
    print("new size: ", (w, h))

    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    print("now nums are: ", (nw, nh))

    image = image.resize((nw, nh), Image.BICUBIC)  # 更改图像尺寸，双立法插值效果很好
    # image.show()
    new_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))  # 生成黑色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式
    # new_image.show()

    return new_image


def restore_image(image, org_img, max_size=720):
    iw, ih = org_img.size  # 原始图像的尺寸
    w, h = (max_size, max_size)  # 目标图像的尺寸

    print("original size: ", (iw, ih))
    print("new size: ", (w, h))

    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    area = ((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh)
    print("crop area", area)
    cropped_img = image.crop(area)
    return cropped_img.resize((iw, ih), Image.BICUBIC)


bg_image = None
image_bg = None

fill_demos = get_fill_demos()
mate_demos = get_mate_demos()
enhance_demos = get_enhance_demos()
colorize_demos = get_colorize_demos()
motion_demos = get_motion_demos()
if_demos = get_movie_IF_demos()
video_mate_demos = get_movie_mate_demos()
video_delogo_demos = get_video_delogo_demos()


def clear_fill_session():
    global bg_image, image_bg
    bg_image = None
    image_bg = None
    if "mask_image" in st.session_state:
        del st.session_state["mask_image"]


def test_mate():
    from image_mate import process_image_mate
    global bg_image, image_bg

    demos_images = {
        "人物1": 0,
        "人物2": 1,
        "多人": 2,
        "有遮挡人物": 3,
        "商品": 4
    }

    cols = st.columns(2)
    demo_image = cols[0].selectbox(
        "示例图片:", demos_images.keys(), on_change=clear_fill_session
    )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传图片"):
            bg_image = st.file_uploader("上传原始图片", help="请上传待处理的原始图片", type=["png", "jpg"], on_change=clear_fill_session)
            if bg_image:
                image_bg = resize_image(Image.open(bg_image), 704)

    if not image_bg:
        bg_image = mate_demos[demos_images[demo_image]]["filename"]
        image_bg = resize_image(Image.open(bg_image), 704)

    st.text("点击'AI抠图'会尝试智能抠出图片的主体人物。")
    if st.button("AI抠图"):
        with st.spinner("AI正在处理中..."):
            mated_image = process_image_mate(get_mate_model(), image_bg)
        st.balloons()
        image_comparison(
            img1=image_bg,
            img2=mated_image,
            label1="原图",
            label2="智能抠图",
            width=max_size,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )
    else:
        st.image(image_bg)


def test_fill():
    from image_fill import process_image

    global bg_image, image_bg
    demos_images = {
        "城市背景": 0,
        "人物1": 1,
        "人物2": 2,
        "人物3": 3,
        "山水风景": 4
    }

    cols = st.columns(2)
    demo_image = cols[0].selectbox(
        "示例图片:", demos_images.keys(), on_change=clear_fill_session
    )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传图片"):
            bg_image = st.file_uploader("上传原始图片", help="请上传待处理的原始图片", type=["png", "jpg"], on_change=clear_fill_session)
            if bg_image:
                image_bg = resize_image(Image.open(bg_image))

    if bg_image:
        init_json_data = None
    else:
        bg_image = fill_demos[demos_images[demo_image]]["filename"]
        init_json_data = fill_demos[demos_images[demo_image]]["json_data"]
        image_bg = resize_image(Image.open(bg_image))


    stroke_width = 3
    stroke_color = "#000"
    bg_color = "#eee"

    st.text("请选择原图使用多边形画笔工具对原图进行遮罩抹除破坏，再点击'修复'按钮尝试AI修复。")

    image_mask = None

    if st.button("AI修复"):
        if "mask_image" not in st.session_state:
            bg_image = None
            st.error("请先选择原图，并对原图进行抹除破坏！")
        else:
            with st.spinner("AI正在处理中..."):
                image_mask = st.session_state["mask_image"]
                mask = Image.fromarray(image_mask.astype(np.uint8)).convert("L")
                #st.image(image_mask)
                result = process_image(get_fill_model(), image_bg, mask)
                st.balloons()
            image_comparison(
                img1=image_bg,
                img2=result,
                label1="原图",
                label2="AI修复图",
                width=max_size,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )
    else:
        st.text("在图片上点击鼠标左键新增多边形顶点，右键完成图边形。")
        drawing_mode = "polygon"
        realtime_update = False

        placeholder = st.empty()
        with placeholder.container():
            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=image_bg if bg_image else None,
                update_streamlit=realtime_update,
                width= max_size,
                height= max_size,
                drawing_mode=drawing_mode,
                display_toolbar=True,
                initial_drawing=init_json_data,
                key="full_app",
            )

            # Do something interesting with the image data and paths
            if canvas_result.json_data is not None:
                #canvas_result.json_data
                if len(canvas_result.json_data["objects"]) > 0:
                    st.session_state["mask_image"] = canvas_result.image_data


def test_enhance(max_size=256):
    from image_enhance import process_image_enhance2, process_image_enhance

    global bg_image, image_bg

    demos_images = {
        "老照片1": 0,
        "老照片2": 1,
        "糊照片1": 2,
        "糊照片2": 3,
    }

    cols = st.columns(2)
    demo_image = cols[0].selectbox(
        "示例图片:", demos_images.keys(), on_change=clear_fill_session
    )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传图片"):
            bg_image = st.file_uploader("上传原始图片", help="请上传待处理的原始图片", type=["png", "jpg"],
                                        on_change=clear_fill_session)
            if bg_image:
                nm = Image.open(bg_image)
                max_size = max(nm.size[0], nm.size[1])
                # if max_size > 1500:
                #     max_size = int(max_size * (1500 / max_size))
                # if max_size > 720:
                #    max_size = int(max_size / 2)
                image_bg = resize_image(nm, max_size)

    if not image_bg:
        bg_image = enhance_demos[demos_images[demo_image]]["filename"]
        nm = Image.open(bg_image)
        image_bg = resize_image(nm, max_size)

    st.text("点击'AI超清'会尝试智能补充图片细节并提升图片分辨率。")
    if st.button("AI超清"):
        with st.spinner("AI正在处理中..."):
            new_image = resize_image(image_bg, max_size)

            result = process_image_enhance(get_enhance_model(), new_image)

            if not result:
                return {"success": False, "msg": "server is too busy!"}

            result.save("d:/big_result.jpg")

            result = resize_image(result, max_size)

            enhanced_image = restore_image(result, image_bg, max_size)
            enhanced_image_restored = restore_image(result, nm, max_size)
            enhanced_image_restored.save("d:/result.jpg")

        st.balloons()
        image_comparison(
            img1=image_bg,
            img2=enhanced_image,
            label1="原图",
            label2="超清图",
            width=max_size,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )
    else:
        st.image(image_bg)


def test_image_motion():
    from image_colorization import process_image_colorization

    # 选择原图， 选择颜色参考图，点击智能调色
    images_pairs = {
        "人物动作照片组1": 2,
        "人物动作照片组2": 0,
        "人物动作照片组3": 1,
    }

    need_process = False
    st.text("使用两张有小幅度动作差异的照片，使用frame interpolation智能补全中间的动作并生成动作视频，是视频自动插帧的技术基础。")

    demo_image = st.selectbox(
        "示例动作照片组", images_pairs.keys(), on_change=None
    )

    if st.button("AI生成动作视频"):
        need_process = True
        # st.text("正在赶工中...")

    if not need_process:
        cols = st.columns(2)

        start_image_path, ref_image_path = None, None
        with cols[0]:

            # with st.expander("上传图片"):
            #     start_image_path = st.file_uploader("上传动作开始图片", help="请上传动作开始图片", type=["png", "jpg"],
            #                                               on_change=None)
            #     if start_image_path:
            #         start_image = resize_image(Image.open(start_image_path))

            if not start_image_path:
                start_image_path = motion_demos[images_pairs[demo_image]]["start"]
                start_image = resize_image(Image.open(start_image_path))

            if not need_process:
                st.text("动作开始图片")
                st.image(start_image)

        with cols[1]:
            # with st.expander("上传图片"):
            #     ref_image_path = st.file_uploader("上传动作结束照片", help="请上传动作结束照片", type=["png", "jpg"],
            #                                       on_change=None)
            #     if ref_image_path:
            #         ref_image = resize_image(Image.open(ref_image_path), max_size)

            if not ref_image_path:
                ref_image_path = motion_demos[images_pairs[demo_image]]["end"]
                ref_image = resize_image(Image.open(ref_image_path), max_size)

            if not need_process:
                st.text("动作结束图片")
                st.image(ref_image)

    if need_process:
        cols = st.columns(4)
        with cols[0]:
            st.balloons()
            st.text("由于计算资源限制，以下示例视频并非实时生成。")
            st.video(motion_demos[images_pairs[demo_image]]["out"])





def test_video_if():
    # 选择原图， 选择颜色参考图，点击智能调色
    images_pairs = {
        "人物动作视频": 0,
        "主持人演讲视频": 1,
        "镜头移动视频": 2,
    }

    need_process = False
    st.text("视频补帧是视频加工的核心技术之一，是提高视频压缩比率、降低视频网络传输时延、可以提高视频帧特效加工速度和性能。")
    st.text("以下演示先将原视频通过抽帧压缩为每秒5帧的跳帧视频，然后再通过AI补帧技术进行8倍插帧来还原视频帧率。")

    demo_if = st.selectbox(
        "选择演示视频组", images_pairs.keys(), on_change=None
    )

    if st.button("AI视频补帧"):
        need_process = True
        # st.text("正在赶工中...")

    cols = st.columns(4)

    start_image_path, ref_image_path = None, None
    if not ref_image_path:
        ref_image_path = if_demos[images_pairs[demo_if]]["d5"]

    if not need_process:
        with cols[0]:
            if not start_image_path:
                start_image_path = if_demos[images_pairs[demo_if]]["org"]
                #start_image = resize_image(Image.open(start_image_path))

            if not need_process:
                st.text("原始视频")
                st.video(start_image_path)

        with cols[1]:
            if not ref_image_path:
                ref_image_path = if_demos[images_pairs[demo_if]]["d5"]
                #ref_image = resize_image(Image.open(ref_image_path), max_size)

            if not need_process:
                st.text("抽帧为5帧/秒的掉帧视频")
                st.video(ref_image_path)

    if need_process:
        st.text("由于计算资源限制，以下示例视频并非实时生成。")

        cols = st.columns(4)

        with cols[0]:
            st.text("抽帧为5帧/秒的掉帧视频")
            st.video(ref_image_path)

        with cols[1]:
            st.balloons()
            st.text("AI补帧后视频")
            st.video(if_demos[images_pairs[demo_if]]["f40"])



def test_video_mate():
    from video_mate import process_video_mate

    global bg_image, image_bg

    demos_images = {
        "主持人短视频": 0,
        "室外固定背景人物": 1,
        "室内固定背景人物": 2,
        "多人复杂背景带镜头移动": 3,
    }

    need_process = False
    st.text("视频人物抠图是慧抖销的秀客视频主持人的技术基础之一，可以将原视频内的人物通过AI算法自动抠取出来。")
    st.text("不再需要绿幕拍摄，即可将前景人物影像从原始视频中自动抠出，然后即可以作为秀客素材叠加到不同背景而生成新的视频。")
    st.text("注意，为保证AI能够自动分析出前景和背景，原始视频的背景变化不能太大；为抠出清晰边缘，人物动作不要与背景事物有接触。")

    org_video_path = None

    cols = st.columns(2)
    with cols[0]:
        demo_if = st.selectbox(
            "选择演示视频", demos_images.keys(), on_change=None
        )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传视频"):
            org_video_file = st.file_uploader("上传原始视频", help="请上传待处理的原始视频", type=["mp4"],
                                        on_change=None)
            if org_video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(org_video_file.read())
                org_video_path = tfile.name

    if not org_video_path:
        org_video_path = video_mate_demos[demos_images[demo_if]]["filename"]

    if st.button("AI抠视频"):
        need_process = True
        # st.text("正在赶工中...")

    cols = st.columns(4)

    if not need_process:
        with cols[0]:
            st.text("原始视频")
            st.video(org_video_path)
    else:
        with cols[0]:
            output_path = ".".join(org_video_path.split(".")[:-1]) + "_output.mp4"
            with st.spinner("视频加工中，为绿色环保演示仅处理视频前5秒"):
                st_progress_bar = st.empty()
                process_video_mate(get_video_mate_model(), org_video_path, output_path, tqdm=tqdm, st_progress_bar=st_progress_bar)
                st_progress_bar.empty()
                st.balloons()

        with cols[0]:
            st.text("原始视频")
            st.video(org_video_path)

        with cols[1]:
            st.text("抠图后视频")
            st.video(output_path)


def test_delogo():
    from video_delogo import process_delogo

    demos_images = {
        "测试字幕视频": 0,
        #"带水印字幕视频": 1
    }

    need_process = False
    st.text("史上最强的水印、台标、字幕智能去除服务，针对硬嵌入到视频中的台标、水印和字幕去除，并尽可能地不影响原视频质量。")
    st.text("综合应用了图片文字识别、图片修复、视频光流分析、视频抽帧插帧等多种技术。")

    org_video_path = None

    cols = st.columns(2)
    with cols[0]:
        demo_if = st.selectbox(
            "选择演示视频", demos_images.keys(), on_change=None
        )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传视频"):
            org_video_file = st.file_uploader("上传原始视频", help="请上传待处理的原始视频", type=["mp4"],
                                        on_change=None)
            if org_video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(org_video_file.read())
                org_video_path = tfile.name

    if not org_video_path:
        org_video_path = video_delogo_demos[demos_images[demo_if]]["filename"]

    cols = st.columns(5)
    with cols[0]:
        remove_text = st.checkbox("去字幕", value=True)
    with cols[1]:
        remove_watermark = st.checkbox("去水印和台标", value=True)

    if st.button("AI去水印字幕"):
        need_process = True
        # st.text("正在赶工中...")

    cols = st.columns(4)

    if not need_process:
        with cols[0]:
            st.text("原始视频")
            st.video(org_video_path)
    else:
        with cols[0]:
            output_path = ".".join(org_video_path.split(".")[:-1]) + "_output.mp4"
            if not remove_watermark and not remove_text:
                st.error("请至少选择一个去除内容")
                st.stop()

            #output_path
            with st.spinner("视频加工中，为绿色环保演示仅处理视频前10秒，请确保视频前10秒有字幕或台标"):
                st_progress_bar = st.empty()
                output_path = process_delogo(get_fill_model(), org_video_path, output_path, tqdm=tqdm, st_progress_bar=st_progress_bar)
                st_progress_bar.empty()
                st.balloons()

        with cols[0]:
            st.text("原始视频")
            st.video(org_video_path)

        with cols[1]:
            st.text("去除水印字幕视频")
            st.video(output_path)


def get_gray(rgb):
    width, height = rgb.size
    gray = Image.new('L', (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b = rgb.getpixel((x, y))
            value = r * 299.0 / 1000 + g * 587.0 / 1000 + b * 114.0 / 1000
            value = int(value)
            gray.putpixel((x, y), value)

    return gray


def test_colorize(max_size=384):
    from image_colorization import process_image_colorization
    #global need_process
    #demo_source_image_path = None

    # 选择原图， 选择颜色参考图，点击智能调色
    demos_source_images = {
        "黑白照片": 6,
        "白衣人": 0,
        "灰衣人": 1,
        "黑衣人": 2,
        "红衣人": 3,
        "绿衣人": 4,
        "城市夜景": 5
    }

    ref_images = {
        "白衣人": 0,
        "红衣人": 3,
        "灰衣人": 1,
        "黑衣人": 2,
        "绿衣人": 4,
     }


    need_process = False
    st.text("AI智能图片调色将使用参考图片的颜色对原图片重新上色，使其颜色与参考图片颜色协调，可以用于给黑白灰度照片上色，也是将不同照片融入背景视频的技术基础。")
    if st.button("AI调色"):
        need_process = True
        #st.text("正在赶工中...")


    cols = st.columns(2)

    with cols[0]:
        demo_source_image = cols[0].selectbox(
            "待调色图片:", demos_source_images.keys(), on_change=None
        )
        with st.expander("上传图片"):
            demo_source_image_path = st.file_uploader("上传待调色图片", help="请上传待处理的原始图片", type=["png", "jpg"], on_change=None)
            if demo_source_image_path:
                demo_source_image = resize_image(Image.open(demo_source_image_path))

        if not demo_source_image_path:
            demo_source_image_path = colorize_demos[demos_source_images[demo_source_image]]["filename"]
            demo_source_image = resize_image(Image.open(demo_source_image_path))

        if not need_process:
            st.image(demo_source_image)

    with cols[1]:
        ref_image = cols[1].selectbox(
            "颜色参考图片:", ref_images.keys(), on_change=None
        )
        with st.expander("上传图片"):
            ref_image_path = st.file_uploader("上传颜色参考图片", help="请上传颜色参考图片", type=["png", "jpg"],
                                        on_change=None)
            if ref_image_path:
                ref_image = resize_image(Image.open(ref_image_path), max_size)

        if not ref_image_path:
            ref_image_path = colorize_demos[ref_images[ref_image]]["filename"]
            ref_image = resize_image(Image.open(ref_image_path), max_size)

        st.image(ref_image)

    if need_process:
        models = get_colorization_model()

        with st.spinner("AI正在处理中..."):
            frame1 = resize_image(demo_source_image, max_size * 2)
            result = process_image_colorization(models, frame1, ref_image)

        st.balloons()

        with cols[0]:
            image_comparison(
                img1=frame1,
                img2=result,
                label1="原图",
                label2="AI调色后",
                width=max_size * 2,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )


# 人脸美化
def test_beauty():
    from image_beauty import image_beautify
    st.text("基于图形学而非AI的人脸美化算法，仅支持正脸")

    global bg_image, image_bg

    demos_images = {
        "正大脸": 7,
        "美女1": 6,
        "美女2": 1,
        "多人": 5,
    }

    cols = st.columns(2)
    demo_image = cols[0].selectbox(
        "示例图片:", demos_images.keys(), on_change=clear_fill_session
    )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传图片"):
            bg_image = st.file_uploader("上传原始图片", help="请上传待处理的原始图片", type=["png", "jpg"], on_change=clear_fill_session)
            if bg_image:
                image_bg = resize_image(Image.open(bg_image), 704)

    if not image_bg:
        bg_image = mate_demos[demos_images[demo_image]]["filename"]
        image_bg = resize_image(Image.open(bg_image), 704)

    image_bg_cv2 = cv2.cvtColor(np.asarray(image_bg), cv2.COLOR_RGB2BGR)

    cols = st.columns(4)
    thin = cols[0].slider("瘦脸程度", 0, 100, 50, 1)
    enlarge = cols[1].slider("大眼程度", 0, 100, 50, 1)
    whiten = cols[2].slider("美白程度", 0, 100, 50, 1)
    details = cols[3].slider("磨皮程度", 0, 100, 50, 1)

    #if st.button("美化"):
    if True:
        with st.spinner("正在处理中..."):
            mated_image = image_beautify([image_bg_cv2], thin, enlarge, whiten, details, get_hub())
            mated_image[0] = Image.fromarray(cv2.cvtColor(mated_image[0], cv2.COLOR_BGR2RGB).astype(np.uint8))

            st.balloons()
            image_comparison(
                img1=image_bg,
                img2=mated_image[0],
                label1="原图",
                label2="美化图",
                width=max_size,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )
    else:
        st.image(image_bg)


def test_makeup():
    from image_makeup import do_makeup
    st.text("选择目标人物，即可将其脸部化妆迁移到源人物的脸上")

    global bg_image, image_bg

    demos_images = {
        "女性4": 8,
        "女性1": 7,
        "女性2": 6,
        "女性3": 1
    }

    cols = st.columns(2)
    demo_image = cols[0].selectbox(
        "示例源人物:", demos_images.keys(), on_change=clear_fill_session
    )

    cols[1].text("或者")
    with cols[1]:
        with st.expander("上传图片"):
            bg_image = st.file_uploader("上传源人物图片", help="请上传待处理的源人物图片", type=["png", "jpg"], on_change=clear_fill_session)
            if bg_image:
                image_bg = resize_image(Image.open(bg_image), 361).convert('RGB')

    if not image_bg:
        bg_image = mate_demos[demos_images[demo_image]]["filename"]
        image_bg = resize_image(Image.open(bg_image), 361).convert('RGB')

    st.image("examples/makeup/makeups.png")
    cols = st.columns(2)
    with cols[0]:
        selected_makeup = st.selectbox("目标妆容", [str(i+1) + ".png" for i in range(10)], 5)
    with cols[1]:
        degree = st.slider("妆容迁移程度", 0, 100, 99, 1)

    # 结果显示，五列，三图片
    cols_def = (5, 1, 5, 1, 5, 10)
    col1, col2, col3, col4, col5, col6 = st.columns(cols_def)
    with col1:
        st.text("源人物")
        st.image(image_bg)
    with col2:
        st.header("+")
    with col3:
        st.text("目标妆容")
        image_mr = Image.open("examples/makeup/" + selected_makeup).convert('RGB')
        st.image(image_mr)
    with col4:
        st.header("=")

    if True:
        with st.spinner("正在处理中..."):
            model, parsing_net = get_makeup_model()
            output = do_makeup(image_bg, image_mr, degree / 100, model, parsing_net )
            with col5:
                st.text("妆容迁移结果")
                st.image(output)

# if  action == "beauty":
#     test_beauty()
# elif action == "makeup":
#     test_makeup()
# else:
if selected == "图片修复":
    test_fill()
elif selected == "智能抠图":
    test_mate()
elif selected == "图片超分":
    test_enhance(384)
elif selected == "图片调色":
    test_colorize(384)
elif selected == "两图转视频":
    test_image_motion()
elif selected == "视频补帧":
    test_video_if()
elif selected == "视频抠图":
    test_video_mate()
elif selected == "字幕水印去除":
    test_delogo()
elif selected == "人脸美化" or action == "beauty":
    test_beauty()
elif selected == "装容美化" or action == "makeup":
    test_makeup()
else:
    st.text("正在赶工中...")

# elif selected == "智能抠图":
#     st.text("正在制作")
