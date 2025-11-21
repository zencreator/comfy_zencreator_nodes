import torch
import numpy as np
from PIL import Image
import io
import os
import requests
import fal_client
import json
import base64
import time

class FalGeminiEditNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "make a photo of the man driving the car down the california coastline"}),
                "fal_key": ("STRING", {"multiline": False, "default": "ENTER_YOUR_FAL_KEY_HERE"}),
                
                # Оставляем nano-banana как в доке. Если не работает - смените на fal-ai/flux/dev
                "endpoint": ("STRING", {"default": "fal-ai/nano-banana-pro/edit"}),
                
                "resolution": (["1K", "2K", "4K"], {"default": "4K"}),
                "aspect_ratio": (["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"], {"default": "auto"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "Fal.ai"

    def _upload_image(self, img_tensor):
        """
        Загружает картинку, конвертируя в JPEG для уменьшения размера 
        и предотвращения разрыва соединения (WinError 10054).
        """
        # 1. Конвертация в PIL
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 2. Сохраняем в JPEG вместо PNG!
        # Это уменьшает размер файла в ~10 раз, что критично для 4K загрузок.
        buffered = io.BytesIO()
        img = img.convert("RGB") # JPEG не поддерживает прозрачность, убираем альфа-канал
        img.save(buffered, format="JPEG", quality=95) 
        img_data = buffered.getvalue()
        
        print(f"[Fal.ai Upload] Image size: {len(img_data)/1024/1024:.2f} MB")

        # 3. Агрессивные повторные попытки (Retries)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Пытаемся загрузить (указываем image/jpeg)
                url = fal_client.upload(img_data, "image/jpeg")
                return url
            except Exception as e:
                wait_time = (attempt + 1) * 2 # 2с, 4с, 6с... увеличиваем время ожидания
                print(f"[Fal.ai Upload Error] Attempt {attempt+1}/{max_retries} failed. Retrying in {wait_time}s... Error: {e}")
                time.sleep(wait_time)
        
        raise RuntimeError(f"Failed to upload image after {max_retries} attempts. Network unstable.")

    def generate(self, prompt, fal_key, endpoint, resolution, aspect_ratio, num_images, output_format,
                 image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        
        # 1. Auth
        if fal_key and fal_key.strip() != "" and fal_key != "ENTER_YOUR_FAL_KEY_HERE":
            os.environ["FAL_KEY"] = fal_key
        
        if "FAL_KEY" not in os.environ:
            raise ValueError("FAL_KEY not found.")

        # 2. Upload Images
        potential_inputs = [image_1, image_2, image_3, image_4, image_5]
        image_urls = []

        print(f"--- Fal.ai: Sending to {endpoint} ---")
        
        for idx, img_batch in enumerate(potential_inputs):
            if img_batch is not None:
                for i in range(img_batch.shape[0]):
                    print(f"Processing input image #{idx+1}...")
                    url = self._upload_image(img_batch[i])
                    image_urls.append(url)
                    print(f"Image uploaded: {url}")

        # 3. Arguments
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "resolution": resolution,
            "image_urls": image_urls,
            "sync_mode": True
        }
        
        # Логгер сервера
        def on_queue_update(update):
            if hasattr(update, "logs") and update.logs:
                for log in update.logs:
                    print(f"[Fal.ai Server]: {log['message']}")

        # 4. Call API
        print(f"Status: Request sent. Waiting for result...")
        try:
            result = fal_client.subscribe(
                endpoint, 
                arguments, 
                with_logs=True, 
                on_queue_update=on_queue_update
            )
        except Exception as e:
            raise RuntimeError(f"Fal.ai API Error: {e}")

        # 5. Process Results
        output_tensors = []
        
        images_list = result.get("images", [])
        if not images_list:
             if "image" in result: images_list = [result["image"]]
             elif "outputs" in result: images_list = result["outputs"]

        if not images_list:
            print("FULL RESPONSE:", json.dumps(result, indent=2))
            raise RuntimeError("API finished but returned no images.")

        for img_info in images_list:
            img_path = ""
            if isinstance(img_info, dict) and "url" in img_info:
                img_path = img_info["url"]
            elif isinstance(img_info, str):
                img_path = img_info
            
            if not img_path: continue

            pil_img = None
            try:
                if img_path.startswith("data:image"):
                    base64_data = img_path.split(",", 1)[1]
                    image_bytes = base64.b64decode(base64_data)
                    pil_img = Image.open(io.BytesIO(image_bytes))
                elif img_path.startswith("http"):
                    # Retry logic for download
                    for _ in range(3):
                        try:
                            response = requests.get(img_path, timeout=60)
                            if response.status_code == 200:
                                pil_img = Image.open(io.BytesIO(response.content))
                                break
                        except:
                            time.sleep(2)

                if pil_img:
                    pil_img = pil_img.convert("RGB")
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np)[None,]
                    output_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error processing image: {e}")

        if not output_tensors:
            raise RuntimeError("No images generated.")

        return (torch.cat(output_tensors, dim=0),)

NODE_CLASS_MAPPINGS = { "FalGeminiEditNode": FalGeminiEditNode }
NODE_DISPLAY_NAME_MAPPINGS = { "FalGeminiEditNode": "Fal.ai Nano-Banana (Lightweight Upload)" }