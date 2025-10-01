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

API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

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
    "4K (3840x2160 16:9)": "3840x2160",
    "4K (2160x3840 9:16)": "2160x3840",
    "4K (4096x3072 4:3)": "4096x3072",
    "4K (3072x4096 3:4)": "3072x4096",

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

                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city at sunset"}),

                # Size presets (UI-friendly names)
                "size": (list(SIZE_MAP.keys()), {"default": "2K (2048x2048 1:1)"}),

                "custom_width": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),

                # Seed (API accepts -1 or number 0…2147483647)
                "seed_value": ("INT", {"default": -1, "min": -1, "max": 2147483647}),

                "sequential_image_generation": (["disabled", "auto"], {"default": "disabled"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                "watermark": (["true", "false"], {"default": "true"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
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
                 seed_value,
                 sequential_image_generation, max_images, watermark, response_format,
                 image1=None, image2=None, image3=None, image4=None, image5=None, image_url=""):

        print(f"[ZenCreator/BytePlus] model={model}, prompt={prompt}, size={size}, seed={seed_value}")

        # --- Size mapping ---
        mapped = SIZE_MAP[size]
        if mapped == "Custom":
            size_value = f"{custom_width}x{custom_height}"
        else:
            size_value = mapped

        # --- Seed handling ---
        used_seed = None
        if model != "seedream-4-0-250828":  # seedream-4.0 doesn't support seed
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

        # --- API request ---
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

        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            data = resp.json()
            print("=== API RESPONSE ===")
            print(data)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[ZenCreator/BytePlus] API Error: {e}")
            raise Exception(f"BytePlus API request failed: {str(e)}")

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
                r = requests.get(img_url, timeout=90)
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
