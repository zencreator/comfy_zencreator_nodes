import requests
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import time
import urllib3
import json
import copy
from datetime import datetime
urllib3.disable_warnings()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import boto3
from botocore.client import Config

# ---------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST", "GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retry))
session.mount("http://", HTTPAdapter(max_retries=retry))

# ---------------------------------------------------
# resolutions
# ---------------------------------------------------
RESOLUTIONS = [
    "1:1 — 1024×1024",
    "1:1 — 1536×1536",
    "2:3 — 768×1152",
    "2:3 — 1024×1536",
    "3:2 — 1152×768",
    "3:2 — 1536×1024",
    "3:4 — 960×1280",
    "3:4 — 1080×1440",
    "4:3 — 1280×960",
    "4:3 — 1440×1080",
    "9:16 — 720×1280",
    "9:16 — 1080×1920",
    "16:9 — 1280×720",
    "16:9 — 1920×1080",
    "21:9 — 1344×576",
    "21:9 — 2048×872",
    "Custom — width*height",
]

# ---------------------------------------------------
# Models
# ---------------------------------------------------
QWEN_IMAGE_MODELS = [
    "qwen-image-2.0-pro",
    "qwen-image-2.0-pro-2026-03-03",
    "qwen-image-2.0",
    "qwen-image-2.0-2026-03-03",
]

def parse_resolution(res, width=None, height=None):
    if res == "Custom — width*height":
        if width is None or height is None:
            raise ValueError("Для Custom укажите width и height")
        if not (512 <= width <= 4096 and 512 <= height <= 4096):
            raise ValueError("Ширина и высота: 512–4096 px")
        total = width * height
        if total > 4096 * 4096:
            raise ValueError("Макс. 4096×4096 пикселей")
        width = round(width / 16) * 16
        height = round(height / 16) * 16
        return f"{width}*{height}"
    else:
        return res.split("—")[1].strip().replace("×", "*")

# ---------------------------------------------------
# tensor helpers
# ---------------------------------------------------
def tensor_to_base64(tensor):
    img = tensor.cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(img)

    max_side = 1024
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def url_to_tensor(url):
    print(f"[Qwen] Скачивание: {url}")
    dl_start = time.time()
    r = session.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    print(f"[Qwen] Скачано за {time.time() - dl_start:.2f} сек")
    return torch.from_numpy(arr)[None,]

def make_loggable_payload(payload):
    """
    Копия payload для логов:
    - base64-изображения заменяются короткой заглушкой
    - URL изображений оставляются как есть
    """
    p = copy.deepcopy(payload)

    try:
        content = p["input"]["messages"][0]["content"]
        for item in content:
            if "image" in item and isinstance(item["image"], str):
                if item["image"].startswith("data:image"):
                    item["image"] = f"<base64 image omitted, length={len(item['image'])}>"
    except Exception:
        pass

    return p

# ---------------------------------------------------
# task polling
# ---------------------------------------------------
def poll_task(task_id, headers):
    task_url = f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"
    poll_start = time.time()
    print(f"[Qwen] Polling задачи {task_id}")

    for attempt in range(1, 181):
        time.sleep(2)
        elapsed = time.time() - poll_start
        try:
            r = session.get(task_url, headers=headers, timeout=30)
            data = r.json()
            status = data.get("output", {}).get("task_status", "UNKNOWN")
            print(
                f"[Qwen Poll] {datetime.now().strftime('%H:%M:%S')} | "
                f"попытка {attempt} | статус: {status} | прошло {elapsed:.0f} сек"
            )

            if status == "SUCCEEDED":
                print(f"[Qwen Poll] Успех за {elapsed:.1f} сек")
                return data

            if status == "FAILED":
                msg = data.get("output", {}).get("task_message", "Нет деталей")
                raise Exception(f"Задача провалилась: {msg}")

        except Exception as e:
            print(f"[Qwen Poll] Ошибка опроса: {str(e)}")

    raise Exception(f"Polling таймаут после {time.time() - poll_start:.0f} сек")

# ---------------------------------------------------
# Node
# ---------------------------------------------------
class QwenImage2Pro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "model": (QWEN_IMAGE_MODELS,),
                "resolution": (RESOLUTIONS,),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": ""}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "prompt_extend": ("BOOLEAN", {"default": True, "label_on": "Вкл (расширение промпта)", "label_off": "Выкл"}),
                "use_r2_url": ("BOOLEAN", {"default": False, "label_on": "R2 URL", "label_off": "Base64"}),
                "r2_access_key_id": ("STRING", {"default": ""}),
                "r2_secret_access_key": ("STRING", {"default": ""}),
                "r2_bucket_name": ("STRING", {"default": "cpu-comfyui-assets"}),
                "r2_endpoint": ("STRING", {"default": "https://9039edcc862f2346df7bd4673dce1982.r2.cloudflarestorage.com"}),
                "r2_public_domain": ("STRING", {"default": "https://cpu.storage.zncr.pro"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ZenCreator/Qwen"

    def run(
        self,
        prompt,
        api_key,
        model,
        resolution,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        negative_prompt="",
        num_images=1,
        width=None,
        height=None,
        prompt_extend=True,
        use_r2_url=False,
        r2_access_key_id="",
        r2_secret_access_key="",
        r2_bucket_name="cpu-comfyui-assets",
        r2_endpoint="https://9039edcc862f2346df7bd4673dce1982.r2.cloudflarestorage.com",
        r2_public_domain="https://cpu.storage.zncr.pro",
    ):
        if not api_key.strip():
            raise ValueError("API ключ обязателен")

        overall_start = time.time()
        print(
            f"\n[Qwen] Запуск | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Модель: {model} | Prompt extend: {prompt_extend}"
        )

        size = parse_resolution(resolution, width, height)
        print(f"[Qwen] Разрешение: {size}")

        images = [image1, image2, image3, image4, image5, image6]
        valid_images = [img for img in images if img is not None]

        if not valid_images:
            raise ValueError("Требуется хотя бы одно изображение (image1)")

        print(f"[Qwen] Входных изображений: {len(valid_images)}")

        content = []
        uploaded_objects = []
        s3 = None

        if use_r2_url and r2_access_key_id.strip() and r2_secret_access_key.strip():
            print("[Qwen] Режим R2 URL")
            try:
                s3 = boto3.client(
                    "s3",
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_access_key_id.strip(),
                    aws_secret_access_key=r2_secret_access_key.strip(),
                    config=Config(signature_version="s3v4"),
                    region_name="auto",
                )

                for idx, img_tensor in enumerate(valid_images, 1):
                    arr = np.clip(img_tensor[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG", optimize=True)
                    buf.seek(0)

                    object_name = f"qwen_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"

                    print(f"[Qwen R2] Upload {idx}/{len(valid_images)} → {object_name}")
                    upload_start = time.time()

                    s3.put_object(
                        Bucket=r2_bucket_name,
                        Key=object_name,
                        Body=buf.getvalue(),
                        ContentType="image/png",
                    )

                    img_url = f"{r2_public_domain.rstrip('/')}/{object_name}"
                    print(f"[Qwen R2] URL: {img_url} ({time.time() - upload_start:.2f} сек)")

                    content.append({"image": img_url})
                    uploaded_objects.append(object_name)

            except Exception as e:
                print(f"[Qwen R2] Ошибка R2: {str(e)} → fallback на base64")

        if len(content) < len(valid_images):
            print("[Qwen] Fallback: base64")
            content = []
            for idx, img in enumerate(valid_images, 1):
                b64_start = time.time()
                b64 = tensor_to_base64(img[0])
                print(f"[Qwen] Base64 {idx} за {time.time() - b64_start:.2f} сек")
                content.append({"image": f"data:image/png;base64,{b64}"})

        content.append({"text": prompt})

        payload = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "size": size,
                "n": num_images,
                "prompt_extend": prompt_extend,
            },
        }

        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-DataInspection": '{"input":"disable", "output":"disable"}',
        }

        url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        loggable_payload = make_loggable_payload(payload)
        payload_json = json.dumps(loggable_payload, ensure_ascii=False, indent=2)

        safe_headers = dict(headers)
        auth_value = safe_headers.get("Authorization", "")
        if auth_value.startswith("Bearer ") and len(auth_value) > 16:
            token = auth_value.replace("Bearer ", "")
            safe_headers["Authorization"] = f"Bearer {token[:6]}...{token[-4:]}"
        else:
            safe_headers["Authorization"] = "Bearer ***"

        print(f"[Qwen] Payload size: {len(payload_json.encode('utf-8')):,} байт")
        print("[Qwen] Отправка запроса:")
        print(f"[Qwen] URL: {url}")
        print(f"[Qwen] Headers:\n{json.dumps(safe_headers, ensure_ascii=False, indent=2)}")
        print(f"[Qwen] Body:\n{payload_json}")

        request_start = time.time()

        try:
            r = session.post(url, headers=headers, json=payload, timeout=(30, 900))
            elapsed = time.time() - request_start
            print(f"[Qwen] Ответ | статус {r.status_code} | время {elapsed:.1f} сек")
            r.raise_for_status()
            data = r.json()
        except requests.Timeout:
            print("[Qwen] TIMEOUT > 900 сек")
            raise
        except Exception as e:
            print(f"[Qwen] Ошибка запроса: {type(e).__name__} — {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    print(f"[Qwen] Ответ сервера:\n{e.response.text[:800]}...")
                except Exception:
                    pass
            raise

        if "task_id" in data.get("output", {}):
            task_id = data["output"]["task_id"]
            print(f"[Qwen] Async task_id: {task_id}")
            data = poll_task(task_id, headers)

        images_out = []
        try:
            choices = data.get("output", {}).get("choices", [])
            print(f"[Qwen] Вариантов получено: {len(choices)}")

            if not choices:
                print("[Qwen] WARNING: choices пустой!")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                raise ValueError("Нет choices в ответе")

            for idx, choice in enumerate(choices, 1):
                items = choice.get("message", {}).get("content", [])
                print(f"[Qwen] Choice {idx}: {len(items)} элементов")
                for item in items:
                    if isinstance(item, dict) and "image" in item:
                        img_url = item["image"]
                        tensor = url_to_tensor(img_url)
                        images_out.append(tensor)
                        print(f"[Qwen] Добавлено изображение {idx} из {img_url}")

        except Exception as parse_err:
            print(f"[Qwen] Ошибка парсинга: {str(parse_err)}")
            print(f"[Qwen] Полный ответ:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
            raise

        if not images_out:
            raise Exception("Не получено изображений от API")

        if use_r2_url and uploaded_objects and s3 is not None:
            try:
                print("[Qwen] Удаляем временные файлы из R2")
                for obj in uploaded_objects:
                    s3.delete_object(Bucket=r2_bucket_name, Key=obj)
                    print(f"[Qwen R2] Удалён: {obj}")
            except Exception as del_err:
                print(f"[Qwen R2] Ошибка удаления: {str(del_err)}")

        total_time = time.time() - overall_start
        print(f"[Qwen] Завершено | общее время: {total_time:.1f} сек\n")

        return (torch.cat(images_out, dim=0),)

# ---------------------------------------------------
# Registration
# ---------------------------------------------------
NODE_CLASS_MAPPINGS = {"QwenImage2Pro": QwenImage2Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage2Pro": "Qwen Image 2.0 Pro"}