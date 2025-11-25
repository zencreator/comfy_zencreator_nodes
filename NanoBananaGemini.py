import os
import io
import json
import base64
import hashlib
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
import requests


# ============================================================
# Global / API
# ============================================================

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-3-pro-image-preview:generateContent"
)

SUPPORTED_ASPECTS = [
    "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9"
]

GLOBAL_GEMINI_CACHE: Dict[str, Any] = {}


# ============================================================
# Utility
# ============================================================

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(p: Image.Image) -> torch.Tensor:
    arr = np.array(p.convert("RGB")).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr)


def pick_nearest_aspect(w: int, h: int) -> str:
    target = w / h
    best = None
    best_diff = 9999

    for ar in SUPPORTED_ASPECTS:
        aw, ah = ar.split(":")
        r = float(aw) / float(ah)
        diff = abs(r - target)
        if diff < best_diff:
            best_diff = diff
            best = ar

    return best or "1:1"


# ============================================================
# Node
# ============================================================

class Gemini3ProImagePreviewNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image_size": (["1K", "2K", "4K"],),
                "aspect_ratio": (["auto"] + SUPPORTED_ASPECTS, {"default": "auto"}),
            },

            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "thinking_level": (["auto", "low", "high"], {"default": "auto"}),
                "include_thoughts": ("BOOLEAN", {"default": False}),
                "seed_mode": (["random", "fixed"], {"default": "random"}),
                "seed": ("INT", {"default": 0}),

                # up to 8 images
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thoughts")
    FUNCTION = "generate"
    CATEGORY = "Gemini / Gemini 3 Pro Image Preview"


    # -----------------------------------------------------------
    # Hashing helper
    # -----------------------------------------------------------
    def _hash_tensor(self, t: torch.Tensor) -> str:
        arr = t.cpu().numpy()
        return hashlib.md5(arr.tobytes()).hexdigest()


    # -----------------------------------------------------------
    # Encode images to inlineData
    # -----------------------------------------------------------
    def _encode_images(self, images: List[torch.Tensor]) -> List[Dict]:
        parts = []
        for t in images:
            pil = tensor_to_pil(t)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64
                }
            })
        return parts


    # -----------------------------------------------------------
    # Build payload
    # -----------------------------------------------------------
    def _build_payload(
        self,
        prompt: str,
        images: List[torch.Tensor],
        aspect_ratio: str,
        image_size: str,
        thinking_level: str,
        include_thoughts: bool
    ):
        parts = self._encode_images(images)
        parts.append({"text": prompt})

        if aspect_ratio == "auto":
            if images:
                pil0 = tensor_to_pil(images[0])
                ar = pick_nearest_aspect(pil0.width, pil0.height)
            else:
                ar = "1:1"
        else:
            ar = aspect_ratio

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": ar,
                    "imageSize": image_size
                }
            }
        }

        if thinking_level != "auto" or include_thoughts:
            thinking = {}
            if thinking_level in ["low", "high"]:
                thinking["thinking_level"] = thinking_level
            if include_thoughts:
                thinking["include_thoughts"] = True
            payload["thinkingConfig"] = thinking

        return payload


    # -----------------------------------------------------------
    # Extract outputs — FIXED VERSION (no KeyError)
    # -----------------------------------------------------------
    def _extract_outputs(self, js: dict, include_thoughts: bool):
        # Safety block
        if "promptFeedback" in js:
            raise Exception(f"Gemini Safety Block: {js['promptFeedback']}")

        candidates = js.get("candidates", [])
        if not candidates:
            raise Exception(f"Gemini: empty candidates. Raw response: {js}")

        first = candidates[0]
        content = first.get("content")
        if not isinstance(content, dict):
            raise Exception(f"Gemini: no 'content' field. Raw response: {js}")

        parts = content.get("parts", [])
        if not isinstance(parts, list) or not parts:
            raise Exception(f"Gemini: no 'parts' field in content. Raw: {js}")

        img_b64 = None
        thoughts = ""

        for p in parts:
            inline = p.get("inlineData") or p.get("inline_data")
            if inline and isinstance(inline, dict):
                mime = inline.get("mimeType") or inline.get("mime_type", "")
                if isinstance(mime, str) and mime.startswith("image/"):
                    img_b64 = inline.get("data") or img_b64

            if include_thoughts:
                if "thoughts" in p and isinstance(p["thoughts"], str):
                    thoughts += p["thoughts"]
                if "text" in p and isinstance(p["text"], str):
                    thoughts += "\n" + p["text"]

        if not img_b64:
            raise Exception(f"Gemini: no image in output. Raw: {js}")

        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        return img, thoughts


    # -----------------------------------------------------------
    # Main
    # -----------------------------------------------------------
    def generate(
        self,
        prompt,
        image_size,
        aspect_ratio,
        api_key="",
        thinking_level="auto",
        include_thoughts=False,
        seed_mode="random",
        seed=0,
        **kwargs
    ):
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise Exception("Missing API key")

        images = []
        for i in range(1, 9):
            t = kwargs.get(f"image{i}")
            if t is not None:
                images.append(t)

        if seed_mode == "random":
            import random
            seed = random.randint(1, 2**31 - 1)

        image_hashes = [self._hash_tensor(t) for t in images]

        cache_data = {
            "seed": seed,
            "prompt": prompt,
            "image_hashes": image_hashes,
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts
        }
        cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

        if seed_mode == "fixed" and cache_key in GLOBAL_GEMINI_CACHE:
            cached_img, cached_thoughts = GLOBAL_GEMINI_CACHE[cache_key]
            return cached_img.clone(), cached_thoughts

        payload = self._build_payload(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            thinking_level=thinking_level,
            include_thoughts=include_thoughts
        )

        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }

        r = requests.post(GEMINI_ENDPOINT, headers=headers, data=json.dumps(payload))

        if r.status_code != 200:
            try:
                err = r.json()
            except:
                err = r.text
            raise Exception(f"Gemini error {r.status_code}: {err}")

        js = r.json()

        pil_img, thoughts = self._extract_outputs(js, include_thoughts)
        tensor_img = pil_to_tensor(pil_img)

        if seed_mode == "fixed":
            GLOBAL_GEMINI_CACHE[cache_key] = (
                tensor_img.clone(),
                thoughts
            )

        return tensor_img, thoughts



# ============================================================
# Node registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "Gemini3ProImagePreviewNode": Gemini3ProImagePreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3ProImagePreviewNode": "Gemini 3 Pro Image Preview (Multi-Image, Seed)"
}
