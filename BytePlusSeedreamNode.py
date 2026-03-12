import torch
import requests
import base64
import io
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
from botocore.client import Config
import boto3
from botocore.exceptions import ClientError

# =====================================================
# BYTEPLUS ENDPOINT
# =====================================================
SERVER_ENDPOINTS = {
    "asia": "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
}

# =====================================================
# Size map
# =====================================================
SIZE_MAP = {
    "1K (auto)": "1K",
    "1K (1024x1024 1:1)": "1024x1024",
    "1K (1280x720 16:9)": "1280x720",
    "1K (720x1280 9:16)": "720x1280",
    "1K (1152x864 4:3)": "1152x864",
    "1K (864x1152 3:4)": "864x1152",
    "2K (auto)": "2K",
    "2K (2048x2048 1:1)": "2048x2048",
    "2K (2560x1440 16:9)": "2560x1440",
    "2K (1440x2560 9:16)": "1440x2560",
    "2K (2304x1728 4:3)": "2304x1728",
    "2K (1728x2304 3:4)": "1728x2304",
    "4K (auto)": "4K",
    "4K (4096x4096 1:1)": "4096x4096",
    "4K (5504x3040 16:9)": "5504x3040",
    "4K (3040x5504 9:16)": "3040x5504",
    "4K (4704x3520 4:3)": "4704x3520",
    "4K (3520x4704 3:4)": "3520x4704",
    "4K (3648x4576 4:5)": "3648x4576",
    "4K (4576x3648 5:4)": "4576x3648",
    "Custom": "Custom",
}

def pil_to_tensor(pil_image: Image.Image):
    arr = np.array(pil_image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr)[None]

# =====================================================
# НОДА BytePlus Seedream 4.x + R2 signed URL (как в Qwen)
# =====================================================
class BytePlusSeedream4Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (
                    [
                        "seedream-4-0-250828",
                        "ep-20250918135640-cxht8",
                        "ep-20251203181040-v7thr",
                        "seedream-4-5-251128",
                    ],
                    {"default": "ep-20251203181040-v7thr"},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "input your prompt"}),
                "size": (list(SIZE_MAP.keys()), {"default": "4K (auto)"}),
                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 5504, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 5504, "step": 64}),
                "seed_value": ("INT", {"default": -1}),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                "watermark": (["true", "false"], {"default": "true"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "optimize_prompt_mode": (["", "standard", "fast"], {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
                "server": (["auto", "asia", "custom"], {"default": "auto"}),
                "custom_server_url": ("STRING", {"default": ""}),
                # R2 параметры — как в Qwen
                "upload_to_r2": ("BOOLEAN", {"default": False, "label_on": "R2 signed URL", "label_off": "Base64"}),
                "r2_access_key_id": ("STRING", {"default": ""}),
                "r2_secret_access_key": ("STRING", {"default": ""}),
                "r2_account_id": ("STRING", {"default": ""}),
                "r2_bucket_name": ("STRING", {"default": "cpu-comfyui-assets"}),
                "r2_endpoint": ("STRING", {"default": "https://9039edcc862f2346df7bd4673dce1982.r2.cloudflarestorage.com"}),
                "r2_public_domain": ("STRING", {"default": "https://cpu.storage.zncr.pro"}),
                "r2_signed_expiry": ("INT", {"default": 900, "min": 300, "max": 3600, "step": 60}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "usage_info")
    FUNCTION = "generate"
    CATEGORY = "ZenCreator/BytePlus"
    OUTPUT_IS_LIST = (True, False)

    def generate(
        self,
        api_key,
        model,
        prompt,
        size,
        custom_width,
        custom_height,
        seed_value,
        sequential_image_generation,
        max_images,
        watermark,
        response_format,
        optimize_prompt_mode,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image_url="",
        server="auto",
        custom_server_url="",
        upload_to_r2=False,
        r2_access_key_id="",
        r2_secret_access_key="",
        r2_account_id="",
        r2_bucket_name="cpu-comfyui-assets",
        r2_endpoint="https://9039edcc862f2346df7bd4673dce1982.r2.cloudflarestorage.com",
        r2_public_domain="https://cpu.storage.zncr.pro",
        r2_signed_expiry=900,
    ):
        overall_start = time.time()
        print(f"\n[BytePlus] Запуск | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Модель: {model}")

        # Сервер
        if server == "auto":
            api_urls = [SERVER_ENDPOINTS["asia"]]
        elif server == "asia":
            api_urls = [SERVER_ENDPOINTS["asia"]]
        elif server == "custom":
            if not custom_server_url:
                raise Exception("Custom server selected but URL not provided.")
            api_urls = [custom_server_url if custom_server_url.startswith(("http://", "https://")) else f"https://{custom_server_url}"]
        else:
            api_urls = [SERVER_ENDPOINTS["asia"]]
        print(f"[BytePlus] Final server list: {api_urls}")

        # Размер
        mapped = SIZE_MAP[size]
        size_value = f"{custom_width}x{custom_height}" if mapped == "Custom" else mapped
        print(f"[BytePlus] Разрешение: {size_value}")

        # Входные изображения
        image_tensors = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        print(f"[BytePlus] Входных изображений: {len(image_tensors)}")

        external_urls = [image_url.strip()] if image_url.strip() else []

        image_inputs = []
        uploaded_objects = []
        s3 = None

        if upload_to_r2 and image_tensors and r2_access_key_id.strip() and r2_secret_access_key.strip() and r2_account_id.strip() and r2_bucket_name.strip():
            print("[BytePlus] Режим R2 signed URL")
            try:
                s3 = boto3.client(
                    's3',
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_access_key_id.strip(),
                    aws_secret_access_key=r2_secret_access_key.strip(),
                    config=Config(signature_version='s3v4'),
                    region_name='auto'
                )

                for idx, tensor in enumerate(image_tensors, 1):
                    arr = np.clip(tensor[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG", optimize=True, quality=85)
                    buf.seek(0)

                    object_name = f"seedream4_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                    print(f"[BytePlus R2] Upload {idx}/{len(image_tensors)} → {object_name}")

                    upload_start = time.time()
                    s3.put_object(
                        Bucket=r2_bucket_name,
                        Key=object_name,
                        Body=buf.getvalue(),
                        ContentType='image/png'
                    )
                    print(f"[BytePlus R2] Upload OK ({time.time() - upload_start:.2f} сек)")

                    signed_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': r2_bucket_name, 'Key': object_name},
                        ExpiresIn=r2_signed_expiry
                    )
                    print(f"[BytePlus R2] Signed URL (expires in {r2_signed_expiry} сек): {signed_url}")
                    image_inputs.append(signed_url)
                    uploaded_objects.append(object_name)

            except Exception as e:
                print(f"[BytePlus R2] Ошибка R2: {str(e)} → fallback на base64")

        # Fallback base64
        if len(image_inputs) < len(image_tensors):
            print("[BytePlus] Fallback: base64")
            for idx, tensor in enumerate(image_tensors[len(image_inputs):], len(image_inputs) + 1):
                arr = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                image_inputs.append(f"data:image/png;base64,{b64}")

        image_inputs.extend(external_urls)

        # Payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format,
            "watermark": (watermark == "true"),
            "stream": False,
            "sequential_image_generation": sequential_image_generation,
            "sequential_image_generation_options": {"max_images": max_images},
        }
        if image_inputs:
            payload["image"] = image_inputs if len(image_inputs) > 1 else image_inputs[0]
        if optimize_prompt_mode in ["standard", "fast"]:
            payload["optimize_prompt_options"] = {"mode": optimize_prompt_mode}

        print("[BytePlus] Payload size:", len(json.dumps(payload)))
        print("[BytePlus] Отправка...")

        # Запрос
        session = requests.Session()
        session.headers.update({"Connection": "close"})
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = None
        last_error = None
        for url in api_urls:
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=(120, 600))
                print(f"[BytePlus] Status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"[BytePlus] ERROR contacting {url}: {str(e)}")
                last_error = str(e)
        if data is None:
            raise Exception(f"All servers failed. Last error: {last_error}")

        # Обработка ответа
        usage_info = str(data.get("usage", {})) + f" | size: {size_value}"
        images = []
        for item in data.get("data", []):
            if "error" in item:
                print(f"Generation error: {item.get('error', {}).get('message', 'Unknown')}")
                continue
            try:
                if response_format == "b64_json":
                    raw = base64.b64decode(item["b64_json"])
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                else:
                    img_url = item["url"]
                    print(f"[BytePlus] Скачивание результата: {img_url}")
                    r = session.get(img_url, timeout=60)
                    r.raise_for_status()
                    pil = Image.open(io.BytesIO(r.content)).convert("RGB")
                images.append(pil_to_tensor(pil))
            except Exception as e:
                print(f"Image processing error: {str(e)}")

        if not images:
            raise Exception("No valid images returned")

        # Удаление временных объектов после успешной генерации (как в Qwen)
        if upload_to_r2 and uploaded_objects and s3 is not None:
            try:
                print("[BytePlus] Удаляем временные файлы из R2")
                for obj in uploaded_objects:
                    s3.delete_object(Bucket=r2_bucket_name, Key=obj)
                    print(f"[BytePlus R2] Удалён: {obj}")
            except Exception as del_err:
                print(f"[BytePlus R2] Ошибка удаления: {str(del_err)}")

        total_time = time.time() - overall_start
        print(f"[BytePlus] Завершено | общее время: {total_time:.1f} сек\n")

        return images, usage_info

# =====================================================
# EXPORTS
# =====================================================
NODE_CLASS_MAPPINGS = {
    "BytePlusSeedream4Simple": BytePlusSeedream4Simple
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedream4Simple": "BytePlus Seedream 4.x"
}