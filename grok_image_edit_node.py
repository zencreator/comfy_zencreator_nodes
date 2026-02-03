import io
import base64
import requests
import torch
import numpy as np
from PIL import Image
from xai_sdk import Client


class GrokImageEditNode:
    """
    Grok Image Editor
    IMAGE + PROMPT → IMAGE
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False}),

                "model": (
                    ["grok-imagine-image"],
                ),

                "response_format": (
                    ["url", "b64_json"],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"
    CATEGORY = "ZenCreator/Grok"

    # ============================================
    # Main
    # ============================================

    def run(self, image, prompt, api_key,
            model, response_format):

        if not api_key:
            raise ValueError("API key required")

        client = Client(api_key=api_key)

        # ComfyUI tensor -> PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Encode to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        data_url = f"data:image/png;base64,{encoded}"

        args = {
            "model": model,
            "prompt": prompt,
            "image_url": data_url,
            "image_format": (
                "base64" if response_format == "b64_json"
                else "url"
            ),
        }

        result = client.image.sample(**args)

        # Receive result
        if response_format == "b64_json":
            img_bytes = base64.b64decode(result.image)
            image_out = Image.open(io.BytesIO(img_bytes))
        else:
            r = requests.get(result.url, stream=True)
            r.raise_for_status()
            image_out = Image.open(r.raw)

        image_out = image_out.convert("RGB")

        arr = np.asarray(image_out, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "GrokImageEditNode": GrokImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageEditNode": "Grok Image Edit"
}
