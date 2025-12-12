import torch
import requests
import base64
import io
import numpy as np
from PIL import Image
import ssl
from requests.adapters import HTTPAdapter
from urllib3 import Retry


# ============================
#  Server endpoints
# ============================
SERVER_ENDPOINTS = {
    "asia": "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
    "us-east": "https://ark.us-east.bytepluses.com/api/v3/images/generations",
}


# ============================
#  Size map
# ============================
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


class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_version"] = ssl.PROTOCOL_TLSv1_2
        return super().init_poolmanager(*args, **kwargs)


def quick_health_check(url: str) -> bool:
    try:
        r = requests.head(url, timeout=1)
        return r.status_code < 500
    except:
        return False


# ============================================================
#  MAIN NODE (same name!)  —  backward compatible
# ============================================================

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
                    {"default": "seedream-4-0-250828"},
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

                "optimize_prompt_mode": (
                    ["", "standard", "fast"],
                    {"default": ""}
                ),
            },

            "optional": {
                # ------ old optional ------
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),

                # ------ NEW OPTIONS AT THE END FOR COMPATIBILITY ------
                "server": (
                    ["auto", "asia", "us-east", "custom"],
                    {"default": "auto"}            # safe fallback
                ),
                "custom_server_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "usage_info")
    FUNCTION = "generate"
    CATEGORY = "ZenCreator/BytePlus"
    OUTPUT_IS_LIST = (True, False)

    # ================================================================
    #                          MAIN FUNCTION
    # ================================================================
    def generate(
        self, api_key, model, prompt, size, custom_width, custom_height,
        seed_value, sequential_image_generation, max_images, watermark,
        response_format, optimize_prompt_mode,
        image1=None, image2=None, image3=None, image4=None, image5=None,
        image_url="",
        server="auto",                      # <—— NEW WITH DEFAULT
        custom_server_url=""                # <—— NEW WITH DEFAULT
    ):

        print(f"[BytePlus Node] server param: {server}, custom: {custom_server_url}")

        # =======================
        # Resolve API URL list
        # =======================

        api_urls = []

        if server == "auto":
            # Try healthy servers first
            for name, url in SERVER_ENDPOINTS.items():
                if quick_health_check(url):
                    api_urls.append(url)

            # If all failed health check → still use both
            if not api_urls:
                api_urls = list(SERVER_ENDPOINTS.values())

        elif server == "asia":
            api_urls = [SERVER_ENDPOINTS["asia"]]

        elif server == "us-east":
            api_urls = [SERVER_ENDPOINTS["us-east"]]

        elif server == "custom":
            if not custom_server_url:
                raise Exception("Custom server selected, but custom_server_url is empty.")
            if not custom_server_url.startswith("http"):
                custom_server_url = "https://" + custom_server_url
            api_urls = [custom_server_url]

        print(f"[BytePlus Node] Final API URL list: {api_urls}")

        # =======================
        # Size
        # =======================

        mapped = SIZE_MAP[size]
        size_value = f"{custom_width}x{custom_height}" if mapped == "Custom" else mapped

        # =======================
        # Prepare Img2Img inputs
        # =======================

        image_inputs = []
        for img in [image1, image2, image3, image4, image5]:
            if img is not None:
                pil = Image.fromarray((img[0].cpu().numpy() * 255).astype(np.uint8))
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                image_inputs.append(
                    "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                )

        if image_url:
            image_inputs.append(image_url)

        # =======================
        # Payload
        # =======================

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
            payload["image"] = image_inputs

        if optimize_prompt_mode in ["standard", "fast"]:
            payload["optimize_prompt_options"] = {"mode": optimize_prompt_mode}

        # =======================
        # HTTP session
        # =======================

        retries = Retry(
            total=5, backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )

        session = requests.Session()
        session.mount("https://", TLS12Adapter(max_retries=retries))
        session.headers.update({"Connection": "close"})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # =======================
        # Try sending request to each server
        # =======================

        data = None
        last_error = None

        for url in api_urls:
            try:
                print(f"[BytePlus Node] Trying server → {url}")
                resp = session.post(url, headers=headers, json=payload, timeout=(30, 600))
                print(f"[BytePlus Node] Server {url} returned {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"[BytePlus Node] FAILED server {url}: {e}")
                last_error = e

        if data is None:
            raise Exception(f"All servers failed. Last error: {last_error}")

        # =======================
        # Parse output
        # =======================

        usage_info = str(data.get("usage", {}))
        usage_info += f" | size: {size_value}"

        images = []

        for item in data.get("data", []):
            if response_format == "b64_json":
                img_data = base64.b64decode(item["b64_json"])
                pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img_url = item["url"]
                r = session.get(img_url, timeout=120)
                r.raise_for_status()
                pil = Image.open(io.BytesIO(r.content)).convert("RGB")

            images.append(pil_to_tensor(pil))

        return images, usage_info


# ============================================================
# NODE EXPORTS
# ============================================================

NODE_CLASS_MAPPINGS = {
    "BytePlusSeedream4Simple": BytePlusSeedream4Simple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedream4Simple": "BytePlus Seedream 4.0"
}
