"""
BytePlus Seedream 4.0 Node for ComfyUI
Generates 1–15 images from BytePlus API
"""

import torch
import requests
import base64
import io
import numpy as np
from PIL import Image
import ssl
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager, Retry

API_URLS = [
    "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
    "https://ark.us-east.bytepluses.com/api/v3/images/generations",  # fallback endpoint
]

# Size mapping: UI display name → API value
SIZE_MAP = {
    # 1K
    "1K (auto)": "1K",
    "1K (1024x1024 1:1)": "1024x1024",
    "1K (1280x720 16:9)": "1280x720",
    "1K (720x1280 9:16)": "720x1280",
    "1K (1152x864 4:3)": "1152x864",
    "1K (864x1152 3:4)": "864x1152",

    # 2K
    "2K (auto)": "2K",
    "2K (2048x2048 1:1)": "2048x2048",
    "2K (2560x1440 16:9)": "2560x1440",
    "2K (1440x2560 9:16)": "1440x2560",
    "2K (2304x1728 4:3)": "2304x1728",
    "2K (1728x2304 3:4)": "1728x2304",

    # 4K
    "4K (auto)": "4K",
    "4K (4096x4096 1:1)": "4096x4096",
    "4K (5504x3040 16:9)": "5504x3040",
    "4K (3040x5504 9:16)": "3040x5504",
    "4K (4704x3520 4:3)": "4704x3520",
    "4K (3520x4704 3:4)": "3520x4704",
    "4K (3648x4576 4:5)": "3648x4576",
    "4K (4576x3648 5:4)": "4576x3648",

    # Custom
    "Custom": "Custom",
}


def pil_to_tensor(pil_image: Image.Image):
    """Convert PIL.Image to ComfyUI torch.Tensor format"""
    arr = np.array(pil_image).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr)[None,]  # add batch dim
    return tensor


class TLS12Adapter(HTTPAdapter):
    """Forces TLS 1.2 and adds retry support"""
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_version'] = ssl.PROTOCOL_TLSv1_2
        return super().init_poolmanager(*args, **kwargs)


class BytePlusSeedream4Simple:
    """
    ComfyUI Node: BytePlus Seedream 4.0
    Generates 1–15 images from BytePlus API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "model": (
                    ["seedream-4-0-250828", "ep-20250918135640-cxht8"],
                    {"default": "seedream-4-0-250828"}
                ),
                "prompt": ("STRING", {"multiline": True, "default": "input your prompt"}),
                "size": (list(SIZE_MAP.keys()), {"default": "4K (auto)"}),

                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 5504, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 5504, "step": 64}),

                "seed_value": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                "watermark": (["true", "false"], {"default": "true"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),

                # NEW — Backward-compatible prompt optimization mode
                "optimize_prompt_mode": (
                    ["", "standard", "fast"],  # "" for old workflows
                    {"default": ""}
                ),
            },

            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "usage_info",)
    FUNCTION = "generate"
    CATEGORY = "ZenCreator/BytePlus"
    OUTPUT_IS_LIST = (True, False)

    def generate(self, api_key, model, prompt, size, custom_width, custom_height,
                 seed_value, sequential_image_generation, max_images, watermark,
                 response_format, optimize_prompt_mode,
                 image1=None, image2=None, image3=None,
                 image4=None, image5=None, image_url=""):

        print(f"[ZenCreator/BytePlus] model={model}, prompt={prompt}, size={size}, seed={seed_value}")

        # --- BACKWARD COMPATIBILITY FIX ---
        # If old workflow passed "", fallback to "standard"
        if optimize_prompt_mode not in ["standard", "fast"]:
            optimize_prompt_mode = "standard"

        # --- Size mapping ---
        mapped = SIZE_MAP[size]
        if mapped == "Custom":
            size_value = f"{custom_width}x{custom_height}"
        else:
            size_value = mapped

        # --- Seed handling ---
        used_seed = None
        if model != "seedream-4-0-250828":
            used_seed = seed_value

        # --- Collect input images ---
        image_list = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        image_urls = []

        if image_list:
            for img in image_list:
                pil = Image.fromarray((img[0].cpu().numpy() * 255).astype(np.uint8))
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                image_urls.append("data:image/png;base64," + b64_str)

        if image_url:
            image_urls.append(image_url)

        # --- Payload ---
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format,
            "sequential_image_generation": sequential_image_generation,
            "sequential_image_generation_options": {"max_images": max_images},
            "watermark": (watermark == "true"),
            "stream": False,
        }

        if used_seed is not None:
            payload["seed"] = used_seed

        if image_urls:
            payload["image"] = image_urls

        # NEW — Prompt optimization for both models
        if model in ["seedream-4-0-250828", "ep-20250918135640-cxht8"]:
            payload["optimize_prompt_options"] = {
                "mode": optimize_prompt_mode
            }

        # --- Create session with retries & TLS1.2 ---
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        session = requests.Session()
        adapter = TLS12Adapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({"Connection": "close"})

        # --- Try main + fallback endpoints ---
        data = None
        last_error = None

        for url in API_URLS:
            try:
                print(f"[ZenCreator/BytePlus] Sending request to {url} (timeout 600s)...")
                resp = session.post(url, headers=headers, json=payload, timeout=(30, 600))
                print(f"[ZenCreator/BytePlus] Response status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.SSLError as e:
                print(f"[ZenCreator/BytePlus] SSL/TLS error on {url}: {e}")
                last_error = e
            except requests.exceptions.RequestException as e:
                print(f"[ZenCreator/BytePlus] Request error on {url}: {e}")
                last_error = e

        if data is None:
            raise Exception(f"BytePlus API request failed after retries: {str(last_error)}")

        print("=== API RESPONSE ===")
        print(data)

        usage_info = str(data.get("usage", {}))
        if used_seed is not None:
            usage_info += f" | used_seed: {used_seed}"
        usage_info += f" | size: {size}"

        # --- Convert all results ---
        tensors = []
        for item in data.get("data", []):
            if response_format == "b64_json":
                img_b64 = item["b64_json"]
                pil = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            else:
                img_url = item["url"]
                print(f"[ZenCreator/BytePlus] Fetching image: {img_url}")
                r = session.get(img_url, timeout=180)
                r.raise_for_status()
                pil = Image.open(io.BytesIO(r.content)).convert("RGB")

            tensors.append(pil_to_tensor(pil))

        if not tensors:
            return ([], "No images returned")

        return (tensors, usage_info)


NODE_CLASS_MAPPINGS = {
    "BytePlusSeedream4Simple": BytePlusSeedream4Simple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedream4Simple": "BytePlus Seedream 4.0"
}
