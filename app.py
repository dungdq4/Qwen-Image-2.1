import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces
import gradio as gr
import numpy as np
import random
import io
import time
import uuid
from datetime import datetime

from PIL import Image

import base64
import json
import re

import torch
from diffusers import QwenImage21Pipeline

import ncii_guard

# ============== Tham số cấu hình ==============
# Thư mục log, có thể tùy chỉnh qua biến môi trường LOG_DIR
LOG_DIR = os.environ.get("LOG_DIR", "./generation_logs_paper_case")

# ============== Cấu hình mô hình ==============
MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image-2.1")
# Prompt rewriting (Qwen-Image-2.1-PE-T2I / PE-I2I) runs in a companion Space so this
# Space only holds the diffusion pipeline and fits a `large` ZeroGPU slice.
PE_SPACE_ID = os.environ.get("PE_SPACE_ID", "hugging-apps/qwen-image-2-1-prompt-enhancer")
PE_MAX_NEW_TOKENS = {"t2i": 1536, "i2i": 2048}
GUARD_THRESHOLD = 0.5
NUM_INFERENCE_STEPS = 40
QUALITY_RESOLUTIONS = {"speed": 1024, "quality": 2048}
TRUE_CFG_SCALE = 1.0
# The prefix KV cache costs ~2 GB per 1K condition image; above this budget it is
# switched off so many-image edits still fit next to the weights.
KV_CACHE_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
KV_CACHE_BUDGET_GB = 10.0

MAX_INPUT_IMAGES = 10
DEFAULT_LANGUAGE = "vi"

pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
pipe.to("cuda")
# Decode large outputs in tiles so a 2K VAE decode fits next to the weights.
pipe.vae.enable_tiling(
    tile_sample_min_height=1536,
    tile_sample_min_width=1536,
    tile_sample_stride_height=1152,
    tile_sample_stride_width=1152,
)

# The NCII classifier runs in a CPU subprocess: a transformers forward in the main
# process breaks the ZeroGPU worker fork.
ncii_guard.start()

# ============== Độ phân giải đặt sẵn ==============
SIZE_PRESETS_2K = {
    "2688x1536 (16:9)": (2688, 1536),
    "1536x2688 (9:16)": (1536, 2688),
    "2048x2048 (1:1)": (2048, 2048),
    "2368x1728 (4:3)": (2368, 1728),
    "1728x2368 (3:4)": (1728, 2368),
}
SIZE_PRESETS_1K = {
    "1344x768 (16:9)": (1344, 768),
    "768x1344 (9:16)": (768, 1344),
    "1184x864 (4:3)": (1184, 864),
    "864x1184 (3:4)": (864, 1184),
    "1024x1024 (1:1)": (1024, 1024),
}
# Hợp nhất tất cả cấu hình sẵn, thêm tùy chọn tùy chỉnh
ALL_SIZE_PRESETS = {"Tùy chỉnh (Custom)": None}
ALL_SIZE_PRESETS.update({f"2K - {k}": v for k, v in SIZE_PRESETS_2K.items()})
ALL_SIZE_PRESETS.update({f"1K - {k}": v for k, v in SIZE_PRESETS_1K.items()})


# ============== Lớp ghi log ==============
class GenerationLogger:
    """Ghi log quá trình tạo: JSON (thông tin prompt) + PNG (hình ảnh)"""

    def __init__(self, log_dir="./logs"):
        """
        Khởi tạo bộ ghi log

        Args:
            log_dir: Thư mục lưu log
        """
        self.log_dir = log_dir
        self._ensure_dir()


    def _ensure_dir(self):
        """Đảm bảo thư mục log tồn tại"""
        os.makedirs(self.log_dir, exist_ok=True)

    def set_log_dir(self, log_dir):
        """
        Thiết lập thư mục log động

        Args:
            log_dir: Đường dẫn thư mục log mới
        """
        self.log_dir = log_dir
        self._ensure_dir()

    def log_generation(self, original_prompt, enhanced_prompt, image, seed, params, gpu_id=None, input_images=None, username=None, api_response=None):
        """
        Lưu toàn bộ bản ghi của một lần tạo

        Args:
            original_prompt: Prompt gốc do người dùng nhập
            enhanced_prompt: Prompt sau khi được LLM viết lại
            image: Đối tượng ảnh PIL được tạo
            seed: Seed ngẫu nhiên sử dụng
            params: Dictionary tham số tạo ảnh
            gpu_id: GPU ID được dùng
            input_images: Danh sách ảnh đầu vào (đối tượng PIL Image)
            username: Tên người dùng thực hiện tạo ảnh

        Returns:
            str: Tên cơ sở của tệp log (không kèm phần mở rộng)
        """
        self._ensure_dir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"{timestamp}_{unique_id}"

        # Lưu ảnh đầu vào
        input_image_paths = []
        if input_images and len(input_images) > 0:
            for i, img in enumerate(input_images):
                input_path = os.path.join(self.log_dir, f"{base_name}.{i}.png")
                img.save(input_path)
                input_image_paths.append(os.path.abspath(input_path))
            print(f"[Logger] Đã lưu {len(input_images)} ảnh đầu vào")

        # Lưu ảnh đầu ra
        image_path = os.path.join(self.log_dir, f"{base_name}.png")
        image.save(image_path)

        # Lưu log JSON
        json_path = os.path.join(self.log_dir, f"{base_name}.json")
        log_data = {
            "timestamp": timestamp,
            "username": username,
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "seed": seed,
            "gpu_id": gpu_id,
            "parameters": params,
            "input_images": input_image_paths,
            "image_path": os.path.abspath(image_path),
            "api_response": api_response
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"[Logger] Đã lưu log: {base_name}.json / {base_name}.png")
        return base_name



def validate_image_count(images):
    if images is not None and len(images) > MAX_INPUT_IMAGES:
        raise gr.Error("Hỗ trợ tối đa 10 ảnh đầu vào. / Up to 10 input images are supported.")


def check_prompt_guard(prompt):
    """Reject image-editing prompts the NCII classifier flags. The error is
    deliberately generic and does not say which classifier fired."""
    if ncii_guard.score(prompt or "") >= GUARD_THRESHOLD:
        raise gr.Error("prompt invalid based on our classifiers, try again")


def ratio_to_size(ratio, area=1024 * 1024, multiple=32):
    """Turn a rewriter aspect ratio like "16:9" into a height/width pair of about `area` pixels."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", str(ratio or ""))
    if not match or float(match.group(2)) == 0:
        return None, None
    aspect = float(match.group(1)) / float(match.group(2))
    if not 1 / 4 <= aspect <= 4:
        return None, None
    return aspect_to_size(aspect, area, multiple)


def aspect_to_size(aspect, area, multiple=32):
    width = round((area * aspect) ** 0.5 / multiple) * multiple
    height = round((area / aspect) ** 0.5 / multiple) * multiple
    return height, width


_pe_client = None


def enhance_prompt(prompt, image_paths):
    """Rewrite the prompt with the official PE models in the companion Space.
    Returns (rewritten_prompt, wh_ratio)."""
    global _pe_client
    from gradio_client import Client, handle_file

    if _pe_client is None:
        _pe_client = Client(PE_SPACE_ID, token=os.environ.get("HF_TOKEN"), httpx_kwargs={"timeout": 900}, verbose=False)
    rewritten, wh_ratio, *_ = _pe_client.predict(
        prompt=prompt,
        image_paths=[handle_file(p) for p in image_paths],
        max_new_tokens=PE_MAX_NEW_TOKENS["i2i" if image_paths else "t2i"],
        enable_thinking=False,
        seed=0,
        randomize_seed=True,
        api_name="/enhance",
    )
    return str(rewritten or "").strip(), str(wh_ratio or "").strip()


def kv_cache_fits(n_images):
    return n_images * (1024 // 16) ** 2 * KV_CACHE_BYTES_PER_TOKEN <= KV_CACHE_BUDGET_GB * 1e9


def generation_duration(prompt, image_paths, height, width, negative_prompt, seed):
    """GPU seconds, fitted on this pipeline (eager, large ZeroGPU): step cost scales as
    latent_tokens ** 1.363 and each cached condition image adds a fifth of its tokens."""
    n_images = len(image_paths or [])
    pixels = (int(width) * int(height)) if (width and height) else 1024 * 1024
    prefix = n_images * (1024 // 16) ** 2
    cached = kv_cache_fits(n_images)
    tokens = pixels / 256 + (0.2 * prefix if cached else prefix)
    per_step = 3.4e-6 * tokens ** 1.363 * 1.3
    if negative_prompt:
        per_step *= 2
    fixed = 4 + 2 * n_images + 1.5e-6 * pixels
    return int(min(300, (fixed + NUM_INFERENCE_STEPS * per_step) * 1.25))


@spaces.GPU(duration=generation_duration)
def run_pipeline(prompt, image_paths, height, width, negative_prompt, seed):
    images = [Image.open(p) for p in image_paths] or None
    kwargs = {"use_kv_cache": kv_cache_fits(len(image_paths))}
    tile, stride = (1024, 768) if max(height or 0, width or 0) > 1536 else (1536, 1152)
    pipe.vae.enable_tiling(
        tile_sample_min_height=tile,
        tile_sample_min_width=tile,
        tile_sample_stride_height=stride,
        tile_sample_stride_width=stride,
    )
    if negative_prompt:
        kwargs.update(negative_prompt=negative_prompt, true_cfg_scale=TRUE_CFG_SCALE)
    return pipe(
        prompt,
        image=images,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator("cuda").manual_seed(int(seed)),
        **kwargs,
    ).images[0]


def gallery_paths(input_images):
    """Save gallery images to PNG files so they can be sent to the rewriter Space and
    reopened in the GPU worker without loss (RGBA inputs keep their alpha)."""
    import tempfile

    paths = []
    for item in input_images or []:
        try:
            if isinstance(item, (tuple, list)) and item:
                item = item[0]
            if isinstance(item, str):
                item = Image.open(item)
            elif hasattr(item, "name") and not isinstance(item, Image.Image):
                item = Image.open(item.name)
            if not isinstance(item, Image.Image):
                continue
            if item.mode not in ("RGB", "RGBA"):
                item = item.convert("RGBA")
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            item.save(path)
            paths.append(path)
        except Exception as e:
            print(f"[Warning] Failed to load input image: {e}")
    return paths


# Khởi tạo bộ ghi log
print(f"Initializing logger with directory: {LOG_DIR}")
logger = GenerationLogger(log_dir=LOG_DIR)


# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Stage 1: screen and rewrite the prompt (CPU, no GPU quota) ---
def prepare_stage(input_images, original_prompt, enable_extend, custom_size, quality, seed, randomize_seed):
    """
    Kiểm tra tính hợp lệ của đầu vào, kiểm tra NCII (chỉ khi có ảnh đầu vào), tối ưu hóa prompt (tùy chọn), toàn bộ thực hiện ngoài GPU.
    Returns (image_paths, final_prompt, rewritten_prompt, seed, auto_height, auto_width).
    """
    validate_image_count(input_images)
    if not original_prompt or not original_prompt.strip():
        raise gr.Error("Vui lòng nhập prompt. / Please enter a prompt.")
    image_paths = gallery_paths(input_images)

    if image_paths:
        check_prompt_guard(original_prompt)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    rewritten_prompt, wh_ratio = "", ""
    if enable_extend:
        try:
            rewritten_prompt, wh_ratio = enhance_prompt(original_prompt, image_paths)
        except Exception as e:
            print(f"[Warning] Prompt enhancement failed: {e!r}")
            gr.Warning("Tính năng tối ưu prompt tạm thời không khả dụng, đang dùng prompt gốc. / Prompt enhancement is unavailable; using the original prompt.")

    auto_height, auto_width = (None, None)
    if not custom_size:
        area = QUALITY_RESOLUTIONS.get(quality, 1024) ** 2
        if image_paths:
            last_width, last_height = Image.open(image_paths[-1]).size
            auto_height, auto_width = aspect_to_size(last_width / last_height, area)
        else:
            auto_height, auto_width = ratio_to_size(wh_ratio, area)
            if auto_height is None:
                auto_height, auto_width = aspect_to_size(1.0, area)
    return image_paths, rewritten_prompt or original_prompt, rewritten_prompt, seed, auto_height, auto_width


# --- Stage 2: Image Generation Function ---
def generate_image_stage(
    image_paths,
    original_prompt,
    final_prompt,
    custom_size,
    log_dir,
    seed,
    height,
    width,
    auto_height,
    auto_width,
    negative_prompt=" ",
    prompt_extend=True,
    username=None,
):
    """
    Tự động xác định chế độ dựa trên việc có ảnh đầu vào hay không: Có ảnh -> Chỉnh sửa (Edit), Không có ảnh -> Văn bản thành ảnh (T2I).
    """
    # Cập nhật thư mục log (nếu người dùng thay đổi)
    if log_dir and log_dir.strip():
        logger.set_log_dir(log_dir.strip())

    negative_prompt = (negative_prompt or "").strip()
    is_edit_mode = len(image_paths) > 0
    mode = "edit" if is_edit_mode else "t2i"

    # Cấu hình size: khi bật kích thước tùy chỉnh thì dùng giá trị thiết lập, nếu không thì dùng tỉ lệ mô hình gợi ý hoặc tự động quyết định
    if custom_size:
        out_height, out_width = int(height), int(width)
    else:
        out_height, out_width = auto_height, auto_width

    print(f"Mode: {mode}, prompt extend: {prompt_extend}, input images: {len(image_paths)}, "
          f"seed: {seed}, size: {f'{out_width}x{out_height}' if out_width else 'auto'}")

    image = run_pipeline(final_prompt, image_paths, out_height, out_width, negative_prompt, seed)

    # Ghi log quá trình tạo
    params = {
        "mode": mode,
        "height": height,
        "width": width,
        "negative_prompt": negative_prompt,
        "prompt_extend": prompt_extend,
        "input_images_count": len(image_paths),
        "model": MODEL_ID,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    }

    log_name = logger.log_generation(
        original_prompt=original_prompt,
        enhanced_prompt=final_prompt,
        image=image,
        seed=seed,
        params=params,
        gpu_id=None,
        input_images=[Image.open(p) for p in image_paths] or None,
        username=username,
    )

    print(f"Generation complete, logged as: {log_name}")

    return image


def make_placeholder_image(text, width=512, height=320, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
    """Tạo một ảnh giữ chỗ với văn bản căn giữa"""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGBA", (width, height), (*bg_color, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    return img


# --- Two-step flow: prepare (CPU) then generate (GPU) ---
def prepare_request(input_images, original_prompt, enable_extend, custom_size, quality, seed, randomize_seed):
    image_paths, final_prompt, rewritten_prompt, seed, auto_height, auto_width = prepare_stage(
        input_images, original_prompt, enable_extend, custom_size, quality, seed, randomize_seed,
    )
    request_state = {
        "image_paths": image_paths,
        "final_prompt": final_prompt,
        "auto_height": auto_height,
        "auto_width": auto_width,
    }
    return make_placeholder_image("Generating image..."), seed, rewritten_prompt, request_state


def generate_request(
    request_state,
    original_prompt,
    enable_extend,
    custom_size,
    log_dir,
    seed,
    height,
    width,
    negative_prompt,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Tạo ảnh. Bước tối ưu hóa prompt đã hoàn thành ở bước trước (nếu bật).
    """
    if not request_state:
        raise gr.Error("Vui lòng nhấn nút tạo ảnh lại. / Please click generate again.")
    username = request.username if request else None
    print(f"Request from user: {username}")
    return generate_image_stage(
        request_state["image_paths"], original_prompt, request_state["final_prompt"],
        custom_size, log_dir, seed, height, width,
        request_state["auto_height"], request_state["auto_width"],
        negative_prompt=negative_prompt,
        prompt_extend=enable_extend,
        username=username,
    )


def make_example_loader(images, text, extend):
    """Bind each example to a zero-argument callback without late binding."""
    def load_example():
        return list(images), text, extend
    return load_example


# UI translations are registered once; callbacks update only the current session.
localized_components = []
localized_properties = []


def localize(component, **properties):
    localized_components.append(component)
    localized_properties.append(properties)
    for name, values in properties.items():
        setattr(component, name, values[1 if DEFAULT_LANGUAGE == "en" else 0])
    return component


def switch_language(language):
    index = 1 if language == "en" else 0
    return [gr.update(**{name: values[index] for name, values in props.items()})
            for props in localized_properties]


EXAMPLE_TITLES_EN = {
    "电影角色三视图": "Film character turnaround",
    "小羊肖恩生日故事": "Shaun the Sheep birthday story",
    "民国漫剧分镜": "Period drama storyboard",
    "祈年殿拆解科普卡": "Temple architecture infographic",
    "AI 进化时间轴": "AI evolution timeline",
    "唇膏广告分镜": "Lipstick commercial storyboard",
    "巨兽对决故事板": "Giant creature battle storyboard",
}

EXAMPLE_TITLES_VI = {
    "电影角色三视图": "Thiết kế nhân vật 3 góc nhìn",
    "小羊肖恩生日故事": "Cừu Shaun - Câu chuyện sinh nhật",
    "民国漫剧分镜": "Phân cảnh truyện tranh thời Dân Quốc",
    "祈年殿拆解科普卡": "Infographic kiến trúc điện Kỳ Niên",
    "AI 进化时间轴": "Dòng thời gian tiến hóa AI",
    "唇膏广告分镜": "Phân cảnh quảng cáo son môi",
    "巨兽对决故事板": "Storyboard trận chiến quái thú",
}

CASE_TITLES_VI = {
    # Chinese title fallbacks
    "透明婚纱角色立绘": "Nhân vật anime váy cưới trong suốt",
    "信息图": "Infographic phong cách nhân vật",
    "分镜图": "Phân cảnh truyện tranh 6 ô",
    "全景图": "Ảnh toàn cảnh 360 độ",
    "粉底液商品场景": "Bối cảnh sản phẩm kem nền",
    "手链佩戴": "Phối phụ kiện vòng tay",
    "抱枕节日布置": "Trang trí gối tựa ngày lễ",
    "项链与手链展示": "Trưng bày dây chuyền và vòng tay",
    "传统服饰人像": "Chân dung trang phục truyền thống",
    "跑鞋广告": "Chiến dịch quảng cáo giày chạy bộ",
    "红树林人像": "Chân dung rừng ngập mặn",
    "室内抓拍人像": "Chân dung chụp tự nhiên trong nhà",
    "发型编辑": "Chỉnh sửa kiểu tóc",
    "表情编辑": "Chỉnh sửa biểu cảm",
    "老照片修复": "Phục chế ảnh cũ",
    "风格化": "Phong cách hóa hình ảnh",
    "旅行规划应用界面": "Giao diện app lịch trình du lịch",
    "数学试卷": "Đề thi toán học",
    "学术信息图": "Infographic học thuật",
    "建筑讲解图板": "Bảng thuyết minh kiến trúc",
    "多人合照（6 图）": "Ghép ảnh chân dung nhóm (6 ảnh)",
    "多品穿戴（5 图）": "Phối nhiều trang phục (5 ảnh)",
    "多物布置（10 图）": "Bày trí đồ nội thất (10 ảnh)",
    "局部编辑：圈选": "Chỉnh sửa cục bộ: Khoanh vùng",
    "局部编辑：掩码": "Chỉnh sửa cục bộ: Mặt nạ (Mask)",
    "局部编辑：涂抹": "Chỉnh sửa cục bộ: Tô vẽ vùng chọn",
    # English title mappings
    "Transparent anime bride": "Nhân vật anime váy cưới trong suốt",
    "Character style infographic": "Infographic phong cách nhân vật",
    "Six-panel storyboard": "Phân cảnh truyện tranh 6 ô",
    "360-degree panorama": "Ảnh toàn cảnh 360 độ",
    "Foundation product scene": "Bối cảnh sản phẩm kem nền",
    "Bracelet styling": "Phối phụ kiện vòng tay",
    "Holiday cushion scene": "Trang trí gối tựa ngày lễ",
    "Necklace and bracelet styling": "Trưng bày dây chuyền và vòng tay",
    "Traditional clothing portrait": "Chân dung trang phục truyền thống",
    "Running shoe campaign": "Chiến dịch quảng cáo giày chạy bộ",
    "Mangrove portrait": "Chân dung rừng ngập mặn",
    "Indoor candid portrait": "Chân dung chụp tự nhiên trong nhà",
    "Hairstyle editing": "Chỉnh sửa kiểu tóc",
    "Expression editing": "Chỉnh sửa biểu cảm",
    "Vintage photo restoration": "Phục chế ảnh cũ",
    "Photo stylization": "Phong cách hóa hình ảnh",
    "Travel planner UI": "Giao diện app lịch trình du lịch",
    "Mathematics exam paper": "Đề thi toán học",
    "Academic infographic": "Infographic học thuật",
    "Architecture presentation board": "Bảng thuyết minh kiến trúc",
    "Group portrait (6 images)": "Ghép ảnh chân dung nhóm (6 ảnh)",
    "Outfit styling (5 images)": "Phối nhiều trang phục (5 ảnh)",
    "Room furnishing (10 images)": "Bày trí đồ nội thất (10 ảnh)",
    "Local editing: annotations": "Chỉnh sửa cục bộ: Khoanh vùng",
    "Local editing: mask": "Chỉnh sửa cục bộ: Mặt nạ (Mask)",
    "Local editing: painted region": "Chỉnh sửa cục bộ: Tô vẽ vùng chọn",
}

ENGLISH_EXAMPLES = [
    ("Editorial portrait", "Create an editorial portrait of a botanist in a sunlit greenhouse, surrounded by ferns and delicate orchids. Natural skin texture, linen clothing, soft morning backlight, subtle film grain, medium-format photography, calm expression, no text or watermark."),
    ("Typography poster", 'Design a refined travel poster for a fictional night train. Render the headline exactly as "THE MIDNIGHT EXPRESS" and the subtitle "A journey under the stars". A silver train curves through dark blue mountains beneath a crescent moon. Art Deco geometry, ivory and gold lettering, clear typographic hierarchy, generous margins, print-ready composition.'),
    ("Six-panel storyboard", "Create a six-panel cinematic storyboard about a small robot restoring an abandoned rooftop garden. Show: arrival at dawn, discovery of a dried seedling, repairing an irrigation pipe, planting new seeds, the first rain, and a lush garden at sunset. Keep the robot's round yellow body and blue eyes consistent in every panel. Clear panel borders, expressive visual storytelling, detailed environments, no captions."),
    ("Product photography", "Photograph a translucent emerald perfume bottle on pale limestone beside a shallow pool. Rippling sunlight reflects through the glass onto the stone. A single olive branch frames the upper left corner. Luxury product photography, realistic refraction, crisp bottle edges, soft shadows, uncluttered composition, no logo or text."),
]


# --- Examples and UI Layout ---
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
with open(os.path.join(EXAMPLE_DIR, "cases.json"), encoding="utf-8") as case_file:
    DEMO_CASES = json.load(case_file)
with open(os.path.join(EXAMPLE_DIR, "generated_references.json"), encoding="utf-8") as reference_file:
    GENERATED_REFERENCES = {case["prompt"]: case for case in json.load(reference_file)}


def add_generated_reference(prompt):
    reference = GENERATED_REFERENCES[prompt]
    localize(gr.Gallery(
        value=[os.path.join(EXAMPLE_DIR, name) for name in reference["outputs"]],
        columns=1, height=280, interactive=False,
    ), label=("Kết quả tham khảo", "Reference output"))



def add_case_examples(cases, input_images, prompt, enable_extend):
    """Hiển thị các ảnh tham khảo đi kèm và tải prompt đầy đủ theo thứ tự ban đầu."""
    for case in cases:
        paths = [os.path.join(EXAMPLE_DIR, name) for name in case["inputs"]]
        references = [os.path.join(EXAMPLE_DIR, name) for name in case["outputs"]]
        title_vi = case.get("title_vi") or CASE_TITLES_VI.get(case["title"], case.get("title_en", case["title"]))
        with gr.Accordion(title_vi, open=False) as panel:
            localize(panel, label=(title_vi, case["title_en"]))
            with gr.Row():
                if paths:
                    localize(gr.Gallery(
                        value=[(path, str(index)) for index, path in enumerate(paths, 1)],
                        columns=min(len(paths), 5), height=280, interactive=False,
                    ), label=("Ảnh tham khảo đầu vào (theo thứ tự đánh số)", "Input references (in numbered order)"))
                if references:
                    localize(gr.Gallery(value=references, columns=1, height=280, interactive=False),
                             label=("Kết quả tham khảo", "Reference output"))
            localize(gr.Textbox(value=case["prompt"], lines=4, max_lines=20, interactive=False),
                     label=("Prompt đầy đủ", "Full prompt"))
            button = localize(gr.Button(), value=("Dùng ví dụ này", "Use this example"))
            button.click(
                fn=make_example_loader(paths, case["prompt"], case["prompt_extend"]),
                inputs=[], outputs=[input_images, prompt, enable_extend], queue=False,
            )


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(title="Qwen Image 2.1 Demo") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<a href="https://huggingface.co/Qwen/Qwen-Image-2.1" target="_blank"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;"></a>')
        language = gr.Radio(choices=[("Tiếng Việt", "vi"), ("English", "en")], value=DEFAULT_LANGUAGE, label="Ngôn ngữ / Language")
        instructions = gr.Markdown("""
## Hướng dẫn sử dụng Qwen Image 2.1 Demo
1. Nếu không tải ảnh lên, mặc định vào chế độ tạo ảnh từ văn bản (Text-to-Image); nếu tải ảnh lên, app sẽ vào chế độ chỉnh sửa ảnh (hỗ trợ 1–10 ảnh). Hãy tham chiếu các ảnh theo thứ tự tải lên trong prompt.
2. Mặc định "Tối ưu hóa prompt" (Enable Prompt Extend) được bật, hệ thống sẽ tự động viết lại prompt thông minh. Nếu muốn dùng trực tiếp prompt gốc, bạn có thể tắt tùy chọn này.
3. Bên dưới cung cấp các mẫu thử nghiệm tạo ảnh từ văn bản và chỉnh sửa ảnh để tham khảo năng lực của mô hình.
4. Khi tạo ảnh nền trong suốt (RGBA), bạn nên viết prompt theo định dạng sau (thay thế phần mô tả bằng nội dung mong muốn):

   `Đây là ảnh RGBA có độ trong suốt. .... Ảnh có kênh alpha và nền trong suốt.`
   (Hoặc dùng tiếng Anh: `This is an RGBA image with transparency. .... The image has alpha channel and the background is transparent.`)
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(
                    label="Input Images (for editing)",
                    show_label=True,
                    type="pil",
                    interactive=True,
                    columns=5,
                    height="auto"
                )
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, type="pil", image_mode="RGBA", format='png')

        # Prompt Input
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    lines=6,
                    max_lines=24,
                    show_label=True,
                    placeholder="Describe what you want to generate or edit...",
                    container=True,
            )
        with gr.Row():
            enable_extend = gr.Checkbox(
                label="Tối ưu hóa prompt (Enable Prompt Extend)",
                value=True,
            )
            quality = gr.Radio(
                choices=[("Speed (1024px)", "speed"), ("Quality (2048px)", "quality")],
                value="speed",
                show_label=False,
            )
            generate_button = gr.Button("Generate Image", variant="primary")

        rewritten_prompt_output = localize(
            gr.Textbox(value="", lines=4, max_lines=20, interactive=False),
            label=("Kết quả tối ưu hóa prompt", "Rewritten prompt"),
            placeholder=("Sau khi bật tối ưu hóa prompt và tạo ảnh, prompt được viết lại sẽ hiển thị ở đây.",
                         "After generation with prompt enhancement enabled, the rewritten prompt appears here."),
        )
        enable_extend.change(fn=lambda: "", inputs=[], outputs=[rewritten_prompt_output], queue=False)



        with gr.Accordion("Advanced Settings", open=False) as advanced:
            # Cấu hình thư mục lưu log
            log_dir_input = gr.Textbox(
                label="Thư mục lưu log (Log Directory)",
                show_label=True,
                value=LOG_DIR,
                placeholder="Nhập đường dẫn thư mục lưu log, ví dụ: ./generation_logs",
                interactive=True,
                visible=False,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Prompt loại trừ)",
                value=" ",
                placeholder="Nhập nội dung không mong muốn xuất hiện",
                lines=2,
                interactive=True,
            )

            # Tùy chỉnh kích thước đầu ra
            custom_size = gr.Checkbox(
                label="Tùy chỉnh kích thước đầu ra (Customize output size)",
                value=False,
                info="Khi tắt, mô hình tự động quyết định kích thước; khi bật, sử dụng cài đặt độ phân giải bên dưới",
            )

            # Cài đặt độ phân giải đặt sẵn
            resolution_help = gr.Markdown("**Cài đặt độ phân giải (Resolution Settings) — cần bật tùy chọn kích thước tùy chỉnh ở trên**")
            with gr.Row():
                resolution_preset = gr.Dropdown(
                    choices=list(ALL_SIZE_PRESETS.keys()),
                    value="2K - 2688x1536 (16:9)",
                    label="Độ phân giải đặt sẵn (Resolution Preset)",
                    info="Chọn độ phân giải đặt sẵn hoặc tùy chỉnh",
                )

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2688,
                    step=8,
                    value=2688,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2688,
                    step=8,
                    value=1536,
                )

            # Callback cập nhật độ phân giải đặt sẵn
            def update_resolution(preset_name):
                if preset_name in ("Tùy chỉnh (Custom)", "自定义 (Custom)") or ALL_SIZE_PRESETS.get(preset_name) is None:
                    # Chế độ tùy chỉnh, không thay đổi giá trị hiện tại
                    return gr.update(), gr.update()
                w, h = ALL_SIZE_PRESETS[preset_name]
                return w, h

            resolution_preset.change(
                fn=update_resolution,
                inputs=[resolution_preset],
                outputs=[width, height],
            )

        # --- Examples ---
        examples_heading = gr.Markdown("### Ví dụ mẫu")

        localize(gr.Markdown(), value=("**Ví dụ tạo ảnh từ văn bản nổi bật**", "**Featured text-to-image examples**"))
        add_case_examples([case for case in DEMO_CASES if not case["inputs"]],
                          input_images, prompt, enable_extend)
        edit_heading = gr.Markdown("**Ví dụ chỉnh sửa ảnh**")
        add_case_examples([case for case in DEMO_CASES if case["inputs"]],
                          input_images, prompt, enable_extend)

        # T2I examples (no input images)
        t2i_heading = gr.Markdown("**Ví dụ tạo ảnh từ văn bản**")
        t2i_examples = [
                ["""生成一张真人电影质感的单角色三视图设定参考图。干净灰色背景，左侧为一张较大的完整头肩半身细节图，必须是同一角色从头部到肩部/胸口的连续完整特写，清楚展示面部、发型、上身服装、配饰和材质细节；左侧不要拆成多个局部小图、不要拼贴多个细节框、不要只给眼睛/衣料/配饰等碎片特写。右侧展示同一角色正面、侧面、背面全身三视图，全身可见。角色设定如下：丹尼尔·斯通，纯白背景，无其他人物和场景；丹尼尔·斯通的单人全身立绘，正对镜头站立，表情自然；30岁青年男性，2020年代现代美国，欧洲裔白人外貌，浅肤色，身高约185cm，9头身，魁梧健壮体型，肌肉发达，2020年代棕色短寸头，发质粗硬，眼窝深陷，瞳孔呈灰蓝色，鼻梁高挺且鼻头宽大，唇形厚实且唇色苍白，方下颌轮廓分明，面部皮肤粗糙并带有污垢痕迹，穿着2020年代脏污灰色连帽卫衣配同色系工装裤，脚穿2020年代黑色防滑工装靴，双手佩戴破旧皮革手套，双手自然下垂；光照均匀，高画质，手部完美，无文字水印。 三视图中面部特征、身形比例、发型、服装、配饰、鞋履、姿态和关键视觉元素保持一致；表情自然中性，站姿稳定，比例统一，皮肤和布料金属等材质细节清楚，电影级写实光照，高画质，无文字、无logo、无水印。""", True],
                ["""3D粘土漫画，生成小羊肖恩\n镜头一：清晨的青苔底农场暖意融融，小羊肖恩发现农夫正在精心装饰巧克力生日蛋糕。镜头二：随即心生主意，独自带着小羊提米靠近厨房。镜头三：它悄悄避开熟睡的牧羊犬，带着提米溜进屋内。镜头四：不料提米意外打翻面粉、蹭损蛋糕，场面十分狼狈。镜头五：关键时刻肖恩灵机一动，伸出手指，在厚厚的白色粉末上流畅地画了起来。\n首先是一个巨大的、歪歪扭拙却充满爱意的爱心；接着是"HAPPY BIRTHDAY"的字样。随后，他指挥提米将散落的鲜红草莓摆放在爱心周围，又将从窗外采摘的野花（雏菊、矢车菊）插在蛋糕受损的边缘，巧妙地遮盖了瑕疵。原本狼藉的桌面，瞬间变成了一幅质朴而温馨的田园画作。肖恩退后一步，满意地拍了拍手上的面粉，脸上露出自豪的微笑。镜头六：农夫归来后，看见这份质朴又温馨的布置开怀大笑，这场小小的意外，最终变成了农场治愈又暖心的生日惊喜。""", True],
                ["""一组连贯短视频故事板分镜图，竖版漫剧短剧叙事节奏，2.5D 拟真人细腻质感，民国新旧交替时代背景，主角是清冷民国女学生，配儒雅留洋青年配角，镜头遵循全景 - 中景 - 近景 - 特写顺序；镜头内容： 老街全景雨雾、 二人撑伞对视中景、女生垂眸特写、青年递信手部特写、 巷口转身远景、灯下拆信近景、含泪侧脸特写、留白空镜；每格底部标注简短剧情字幕，画面低饱和复古色调，柔和分层柔光，人物五官写实精致，布料纹理清晰，无杂乱水印，分镜边框规整，镜头衔接流畅，叙事紧凑适配短视频，整体光影层次丰富，全程统一 2.5D 拟真人画风，人物形象全程不崩，场景道具贴合民国时代设定""", True],
                ["""竖版祈年殿拆解科普卡，纯白色宣纸底色，左竖排书法"祈年殿"、小字天坛与红印章，附线稿和简介；主体从上到下分层拆解全建筑构件，右侧虚线序号标注名称，中式手绘，配色棕、深蓝、朱红、米白。""", True],
                ["""生成一张计算机/技术主题的"AI进化历程"时间演化卡，采用线性顺序的时间轴/编年结构，风格为手绘/涂鸦。背景用大地棕褐色系单色渐变，搭配手绘涂鸦元素与统一科普插图。版式采用卡片网格、上图下文，图文留白清晰。标题用粗黑标题体，正文用中文手写体，重点文字采用渐变色填充字与双色调文字，文字信息整体规整易读，完整呈现AI发展节点。""", True],
                ["""生成一张原创高端美妆广告分镜展示板，主题为"LUNE 丝绒唇膏 15 秒广告脚本"。整体像真实品牌团队制作的商业广告脚本视觉板，包含 5 个连续分镜画面 + 时间轴 + 字幕 / 台词 + 镜头说明 + 转场提示。排版清晰、统一、精致、高级，具有女性向、轻奢、电影感、美妆广告质感。主色调为 奶油白、黑色、酒红、玫瑰豆沙、暖金色，灯光柔和，肤质细腻，材质高级。""", True],
                ["""核心大纲 1. 场景：3DCG质感，废弃都市遗迹（黄昏），尘土漫天，绯红色夕阳，地面有巨大脚印。 2. 人物：炎狱巨蜥（熔岩鳞片、喷烈焰）、霜牙巨象（雪白、喷寒气）、幸存人类（远景蜷缩）。 3. 剧情脉络： （1）对峙（0-20秒）：两只巨兽在废墟对峙，人类蜷缩避险，氛围压迫。 （2）开战（21-50秒）：炎狱巨蜥喷烈焰，霜牙巨象喷寒气对抗，引发冲击，高楼坍塌。 （3）激战（51-90秒）：双方互相攻击，均受重伤，嘶吼震彻废墟。 （4）落幕（91-110秒）：两只巨兽重伤对峙，镜头拉远，定格巨兽与废墟，留下悬念""", True],
            ]
        for example_title, example in zip(['电影角色三视图', '小羊肖恩生日故事', '民国漫剧分镜', '祈年殿拆解科普卡', 'AI 进化时间轴', '唇膏广告分镜', '巨兽对决故事板'], t2i_examples):
            title_vi = EXAMPLE_TITLES_VI.get(example_title, example_title)
            with gr.Accordion(title_vi, open=False) as example_panel:
                localize(example_panel, label=(title_vi, EXAMPLE_TITLES_EN[example_title]))
                add_generated_reference(example[0])
                gr.Markdown(example[0])
                example_button = localize(gr.Button(), value=("Dùng prompt này", "Use this prompt"))
                example_button.click(
                    fn=make_example_loader([], example[0], example[1]),
                    inputs=[],
                    outputs=[input_images, prompt, enable_extend],
                    queue=False,
                )

        english_heading = localize(gr.Markdown(), value=("**Ví dụ tạo ảnh từ văn bản (Tiếng Anh)**", "**English text-to-image examples**"))
        for title, text in ENGLISH_EXAMPLES:
            with gr.Accordion(title, open=False):
                add_generated_reference(text)
                gr.Markdown(text)
                button = localize(gr.Button(), value=("Dùng prompt này", "Use this prompt"))
                button.click(fn=make_example_loader([], text, True), inputs=[],
                             outputs=[input_images, prompt, enable_extend], queue=False)

    localize(instructions, value=(instructions.value, """## Qwen Image 2.1 Demo
1. Generate from text without an input image, or upload 1–10 images to edit or combine them. Refer to images by their upload order in your prompt.
2. Prompt enhancement is enabled by default. Turn it off to use your original prompt directly.
3. Expand an example to read its full prompt, then click its button to fill the input. Chinese and English prompts are both supported.
4. For transparent image generation, use the following prompt format and replace `xxxxx` with your image description:

   `This is an RGBA image with transparency. xxxxx The image has alpha channel and the background is transparent.`
"""))
    localize(input_images, label=("Ảnh đầu vào (để chỉnh sửa, tối đa 10 ảnh)", "Input images (editing, up to 10)"))
    localize(result, label=("Kết quả tạo ảnh", "Result"))
    localize(prompt, label=("Prompt (Lời nhắc)", "Prompt"), placeholder=("Mô tả nội dung muốn tạo hoặc chỉnh sửa, có thể tham chiếu ảnh 1–10 theo thứ tự tải lên…", "Describe what to generate or edit; refer to images 1–10 in upload order…"))
    localize(enable_extend, label=("Tối ưu hóa prompt thông minh", "Enhance prompt"))
    localize(quality, choices=(
        [("Tốc độ (1024px)", "speed"), ("Chất lượng (2048px)", "quality")],
        [("Speed (1024px)", "speed"), ("Quality (2048px)", "quality")]))
    localize(generate_button, value=("Tạo ảnh", "Generate image"))
    localize(advanced, label=("Cài đặt nâng cao", "Advanced settings"))
    localize(log_dir_input, label=("Thư mục lưu log", "Log directory"), placeholder=("Nhập đường dẫn thư mục lưu log", "Enter a log directory"))
    localize(seed, label=("Seed ngẫu nhiên", "Seed"))
    localize(randomize_seed, label=("Tự động đổi seed ngẫu nhiên", "Randomize seed"))
    localize(negative_prompt_input, label=("Prompt loại trừ (Negative prompt)", "Negative prompt"), placeholder=("Nhập chi tiết không mong muốn xuất hiện", "Describe what to exclude"))
    localize(custom_size, label=("Tùy chỉnh kích thước đầu ra", "Customize output size"), info=("Khi tắt sẽ tự động chọn kích thước; bật lên để dùng cài đặt bên dưới", "When off, size is chosen automatically. Enable to use the settings below."))
    localize(resolution_help, value=("**Cài đặt độ phân giải — cần bật tùy chọn kích thước tùy chỉnh ở trên**", "**Resolution settings — enable custom output size first**"))
    preset_values = list(ALL_SIZE_PRESETS)
    localize(resolution_preset, label=("Độ phân giải đặt sẵn", "Resolution preset"), info=("Chọn độ phân giải đặt sẵn hoặc tùy chỉnh", "Choose a preset or custom dimensions"), choices=(
        [("Tùy chỉnh" if ALL_SIZE_PRESETS[v] is None else v, v) for v in preset_values],
        [("Custom" if ALL_SIZE_PRESETS[v] is None else v, v) for v in preset_values]))
    localize(width, label=("Chiều rộng", "Width"))
    localize(height, label=("Chiều cao", "Height"))
    localize(examples_heading, value=("### Ví dụ mẫu", "### Examples"))
    localize(t2i_heading, value=("**Ví dụ tạo ảnh từ văn bản**", "**Text-to-image examples**"))
    localize(edit_heading, value=("**Ví dụ chỉnh sửa ảnh**", "**Image-editing examples**"))
    language.change(fn=switch_language, inputs=[language], outputs=localized_components, queue=False)
    input_images.upload(fn=validate_image_count, inputs=[input_images], outputs=[], queue=False)

    # Generate Image button event: screen/rewrite the prompt off-GPU, then generate.
    # Two events so the GPU step is scheduled with a fresh ZeroGPU token after a long rewrite.
    request_state = gr.State(None)
    generate_button.click(
        fn=prepare_request,
        inputs=[
            input_images,     # input_images (Có ảnh -> Edit mode, Không ảnh -> T2I mode)
            prompt,           # original_prompt
            enable_extend,    # enable_extend (Có bật tối ưu hóa prompt hay không)
            custom_size,      # custom_size (Có tùy chỉnh kích thước đầu ra hay không)
            quality,
            seed,
            randomize_seed,
        ],
        outputs=[result, seed, rewritten_prompt_output, request_state]
    ).success(
        fn=generate_request,
        inputs=[
            request_state,
            prompt,
            enable_extend,
            custom_size,
            log_dir_input,    # log_dir
            seed,
            height,
            width,
            negative_prompt_input,  # negative_prompt (Prompt loại trừ)
        ],
        outputs=[result]
    )

if __name__ == "__main__":
    # Sử dụng os.path.realpath để phân giải đường dẫn thực tế, tránh symlink NAS khiến kiểm tra quyền tệp của Gradio bị lỗi (404)
    _script_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    _allowed = [
        os.path.join(_script_dir, "examples"),
        os.path.realpath(EXAMPLE_DIR),
    ]
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=_allowed,
        css=css,
    )
