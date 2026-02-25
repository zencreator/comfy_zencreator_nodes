import torch
import requests
import base64
import io
import numpy as np
from PIL import Image
import json

# =====================================================
# BYTEPLUS ENDPOINT
# =====================================================
SERVER_ENDPOINTS = {
    "asia": "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
}

# =====================================================
# Размеры (полный список из документации февраль 2026)
# =====================================================
SIZE_MAP_5 = {
    "2K (auto)": "2K",
    "3K (auto)": "3K",
    "Custom": "Custom",
    # 2K пресеты
    "2K 1:1     2048×2048": "2048x2048",
    "2K 4:3     2304×1728": "2304x1728",
    "2K 3:4     1728×2304": "1728x2304",
    "2K 16:9    2848×1600": "2848x1600",
    "2K 9:16    1600×2848": "1600x2848",
    "2K 3:2     2496×1664": "2496x1664",
    "2K 2:3     1664×2496": "1664x2496",
    "2K 21:9    3136×1344": "3136x1344",
    # 3K пресеты
    "3K 1:1     3072×3072": "3072x3072",
    "3K 4:3     3456×2592": "3456x2592",
    "3K 3:4     2592×3456": "2592x3456",
    "3K 16:9    4096×2304": "4096x2304",
    "3K 9:16    2304×4096": "2304x4096",
    "3K 2:3     2496×3744": "2496x3744",
    "3K 3:2     3744×2496": "3744x2496",
    "3K 21:9    4704×2016": "4704x2016",
}

def pil_to_tensor(pil_image: Image.Image):
    arr = np.array(pil_image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr)[None]

# =====================================================
# НОДА BytePlus Seedream 5.0 Lite + фиктивный seed
# =====================================================
class BytePlusSeedream5Lite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (
                    [
                        "ep-20260225210520-zkd6v",          # ваш Endpoint ID — замените на актуальный
                        "seedream-5-0-260128",
                        "seedream-5-0-lite-260128",
                    ],
                    {"default": "ep-20260225210520-zkd6v"},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape at sunset"}),
                "size": (list(SIZE_MAP_5.keys()), {"default": "2K (auto)"}),
                "custom_width": ("INT", {"default": 2048, "min": 1440, "max": 4704, "step": 8}),
                "custom_height": ("INT", {"default": 2048, "min": 1440, "max": 4704, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),  # ← фиктивный seed
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                "watermark": (["true", "false"], {"default": "true"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "optimize_prompt_mode": (["", "standard"], {"default": ""}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
                "server": (["auto", "asia", "custom"], {"default": "auto"}),
                "custom_server_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "usage_info")
    FUNCTION = "generate"
    CATEGORY = "ZenCreator/BytePlus"
    OUTPUT_IS_LIST = (True, False)

    def generate(
        self, api_key, model, prompt, size, custom_width, custom_height, seed,  # seed здесь, но не используется
        sequential_image_generation, max_images, watermark,
        response_format, output_format, optimize_prompt_mode,
        image1=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
        image11=None, image12=None, image13=None, image14=None,
        image_url="", server="auto", custom_server_url=""
    ):
        print(f"[Seedream5Lite] Model: {model} | Server: {server} | Seed (dummy): {seed}")

        # --------------------------------------------------
        # Выбор сервера
        # --------------------------------------------------
        if server in ["auto", "asia"]:
            api_urls = [SERVER_ENDPOINTS["asia"]]
        elif server == "custom":
            if not custom_server_url:
                raise Exception("Custom server selected but no URL provided.")
            if not custom_server_url.startswith(("http://", "https://")):
                custom_server_url = "https://" + custom_server_url
            api_urls = [custom_server_url]
        else:
            api_urls = [SERVER_ENDPOINTS["asia"]]

        # --------------------------------------------------
        # Размер + проверка Custom
        # --------------------------------------------------
        mapped = SIZE_MAP_5[size]
        if mapped == "Custom":
            width, height = custom_width, custom_height
            size_value = f"{width}x{height}"
            pixels = width * height
            if pixels < 3686400 or pixels > 10404496:
                raise ValueError(f"Total pixels {pixels} must be in [3,686,400 – 10,404,496]")
            aspect = max(width, height) / min(width, height)
            if aspect < 1/16 or aspect > 16:
                raise ValueError("Aspect ratio must be in [1/16, 16]")
        else:
            size_value = mapped

        # --------------------------------------------------
        # Референсные изображения
        # --------------------------------------------------
        image_inputs = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8,
                    image9, image10, image11, image12, image13, image14]:
            if img is not None:
                pil = Image.fromarray((img[0].cpu().numpy() * 255).astype(np.uint8))
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                image_inputs.append(f"data:image/png;base64,{b64}")

        if image_url.strip():
            image_inputs.append(image_url.strip())

        if len(image_inputs) > 14:
            raise ValueError("Maximum 14 reference images allowed")

        # --------------------------------------------------
        # Payload (без seed!)
        # --------------------------------------------------
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format,
            "output_format": output_format,
            "watermark": (watermark == "true"),
            "stream": False,
        }

        if sequential_image_generation == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {"max_images": max_images}

        if image_inputs:
            payload["image"] = image_inputs

        if optimize_prompt_mode == "standard":
            payload["optimize_prompt_options"] = {"mode": "standard"}

        print("[Seedream5Lite] Sending payload:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        # --------------------------------------------------
        # HTTP-запрос
        # --------------------------------------------------
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
                print(f"[Seedream5Lite] POST → {url}")
                resp = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(120, 600)
                )
                print(f"[Seedream5Lite] Status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"[Seedream5Lite] Error: {str(e)}")
                last_error = str(e)

        if data is None:
            raise Exception(f"All endpoints failed. Last error: {last_error}")

        # --------------------------------------------------
        # Обработка ответа
        # --------------------------------------------------
        usage = data.get("usage", {})
        usage_info = (
            f"Generated: {usage.get('generated_images', 0)} | "
            f"Output tokens: {usage.get('output_tokens', '?')} | "
            f"Total tokens: {usage.get('total_tokens', '?')} | "
            f"Size: {size_value} | Dummy seed: {seed}"
        )

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
                    r = session.get(img_url, timeout=120)
                    r.raise_for_status()
                    pil = Image.open(io.BytesIO(r.content)).convert("RGB")

                images.append(pil_to_tensor(pil))
            except Exception as e:
                print(f"Image processing error: {str(e)}")

        if not images:
            raise Exception("No valid images returned")

        return images, usage_info

# =====================================================
# Экспорт
# =====================================================
NODE_CLASS_MAPPINGS = {
    "BytePlusSeedream5Lite": BytePlusSeedream5Lite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedream5Lite": "BytePlus Seedream 5.0 Lite"
}