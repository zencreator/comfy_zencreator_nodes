import requests
import io
import base64
import torch
import numpy as np
from PIL import Image
from xai_sdk import Client


class GrokImageNode:
    """
    Grok Image Generator for ComfyUI
    Optimized version
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False}),

                "model": (
                    ["grok-imagine-image"],
                ),

                "aspect_ratio": (
                    [
                        "1:1", "3:4", "4:3",
                        "9:16", "16:9",
                        "2:3", "3:2",
                        "9:19.5", "19.5:9",
                        "9:20", "20:9",
                        "1:2", "2:1",
                        "auto"
                    ],
                ),

                "resolution": (
                    ["1k"],
                ),

                "response_format": (
                    ["url", "b64_json"],
                ),

                # currently unused but in API
                "quality": (
                    ["low", "medium", "high"],
                ),

                "n": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"
    CATEGORY = "ZenCreator/Grok"

    # ==================================================
    # Main execution
    # ==================================================

    def run(self, prompt, api_key,
            model,
            aspect_ratio,
            resolution,
            response_format,
            quality,
            n):

        if not api_key:
            raise ValueError("API key is required")

        client = Client(api_key=api_key)

        args = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "image_format": (
                "base64" if response_format == "b64_json"
                else "url"
            ),
        }

        # Batch request
        if n > 1:
            result = client.image.sample_batch(n=n, **args)
            item = result[0]
        else:
            item = client.image.sample(**args)

        # ==============================
        # FAST IMAGE LOAD
        # ==============================
        if response_format == "b64_json":
            img_bytes = base64.b64decode(item.image)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            # stream download faster & memory safe
            r = requests.get(item.url, stream=True)
            r.raise_for_status()
            image = Image.open(r.raw)

        image = image.convert("RGB")

        # ==============================
        # Fast tensor conversion
        # ==============================
        arr = np.asarray(image, dtype=np.float32) / 255.0

        # ComfyUI format [B,H,W,C]
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "GrokImageNode": GrokImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageNode": "Grok Image Generator"
}
