"""
ZenCreator Custom Nodes for ComfyUI
BytePlus Seedream 4.0 & 5.0 Lite & Fal.ai & Gemini 3 Pro & Gemini 3.1 Flash & Grok
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ----------------------------
# BytePlus Seedream 4.x
# ----------------------------
try:
    from .BytePlusSeedreamNode import (
        NODE_CLASS_MAPPINGS as BytePlus_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as BytePlus_Names,
    )
    NODE_CLASS_MAPPINGS.update(BytePlus_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(BytePlus_Names)
except Exception as e:
    print("[ZenCreator] Failed to load BytePlusSeedreamNode (4.x):", e)

# ----------------------------
# BytePlus Seedream 5.0 Lite
# ----------------------------
try:
    from .BytePlusSeedream5Lite import (
        NODE_CLASS_MAPPINGS as Seedream5_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as Seedream5_Names,
    )
    NODE_CLASS_MAPPINGS.update(Seedream5_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(Seedream5_Names)
except Exception as e:
    print("[ZenCreator] Failed to load BytePlusSeedream5Lite (5.0 Lite):", e)

# ----------------------------
# Fal.ai
# ----------------------------
try:
    from .NanoBanabaFALAI import (
        NODE_CLASS_MAPPINGS as Fal_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as Fal_Names,
    )
    NODE_CLASS_MAPPINGS.update(Fal_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(Fal_Names)
except Exception as e:
    print("[ZenCreator] Failed to load NanoBanabaFALAI:", e)

# ----------------------------
# Gemini 3 Pro (Nano Banana Pro)
# ----------------------------
try:
    from .NanoBananaGemini import (
        NODE_CLASS_MAPPINGS as Gemini_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as Gemini_Names,
    )
    NODE_CLASS_MAPPINGS.update(Gemini_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(Gemini_Names)
except Exception as e:
    print("[ZenCreator] Failed to load NanoBananaGemini (3 Pro):", e)

# ----------------------------
# Gemini 3.1 Flash Image Preview (Nano Banana 2) ← НОВОЕ
# ----------------------------
try:
    from .NanoBananaGemini31Flash import (
        NODE_CLASS_MAPPINGS as Gemini31Flash_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as Gemini31Flash_Names,
    )
    NODE_CLASS_MAPPINGS.update(Gemini31Flash_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(Gemini31Flash_Names)
except Exception as e:
    print("[ZenCreator] Failed to load NanoBananaGemini31Flash (3.1 Flash):", e)

# ----------------------------
# Grok Unified
# ----------------------------
try:
    from .grok_unified_node import (
        NODE_CLASS_MAPPINGS as Grok_Classes,
        NODE_DISPLAY_NAME_MAPPINGS as Grok_Names,
    )
    NODE_CLASS_MAPPINGS.update(Grok_Classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(Grok_Names)
except Exception as e:
    print("[ZenCreator] Failed to load grok_unified_node:", e)

# ----------------------------
# Grok Image Generator
# ----------------------------
try:
    from .grok_image_node import (
        NODE_CLASS_MAPPINGS as GrokImageClasses,
        NODE_DISPLAY_NAME_MAPPINGS as GrokImageNames,
    )
    NODE_CLASS_MAPPINGS.update(GrokImageClasses)
    NODE_DISPLAY_NAME_MAPPINGS.update(GrokImageNames)
except Exception as e:
    print("[ZenCreator] Failed to load grok_image_node:", e)

# ----------------------------
# Grok Image Edit
# ----------------------------
try:
    from .grok_image_edit_node import (
        NODE_CLASS_MAPPINGS as GrokImageEditClasses,
        NODE_DISPLAY_NAME_MAPPINGS as GrokImageEditNames,
    )
    NODE_CLASS_MAPPINGS.update(GrokImageEditClasses)
    NODE_DISPLAY_NAME_MAPPINGS.update(GrokImageEditNames)
except Exception as e:
    print("[ZenCreator] Failed to load grok_image_edit_node:", e)

# ----------------------------
# Grok Video Generator
# ----------------------------
try:
    from .grok_video_node import (
        NODE_CLASS_MAPPINGS as GrokVideoClasses,
        NODE_DISPLAY_NAME_MAPPINGS as GrokVideoNames,
    )
    NODE_CLASS_MAPPINGS.update(GrokVideoClasses)
    NODE_DISPLAY_NAME_MAPPINGS.update(GrokVideoNames)
except Exception as e:
    print("[ZenCreator] Failed to load grok_video_node:", e)

# ----------------------------
# Grok Image → Video
# ----------------------------
try:
    from .grok_image_to_video_node import (
        NODE_CLASS_MAPPINGS as GrokI2VClasses,
        NODE_DISPLAY_NAME_MAPPINGS as GrokI2VNames,
    )
    NODE_CLASS_MAPPINGS.update(GrokI2VClasses)
    NODE_DISPLAY_NAME_MAPPINGS.update(GrokI2VNames)
except Exception as e:
    print("[ZenCreator] Failed to load grok_image_to_video_node:", e)

# ----------------------------
# Grok Video Edit
# ----------------------------
try:
    from .grok_video_edit_node import (
        NODE_CLASS_MAPPINGS as GrokVideoEditClasses,
        NODE_DISPLAY_NAME_MAPPINGS as GrokVideoEditNames,
    )
    NODE_CLASS_MAPPINGS.update(GrokVideoEditClasses)
    NODE_DISPLAY_NAME_MAPPINGS.update(GrokVideoEditNames)
except Exception as e:
    print("[ZenCreator] Failed to load grok_video_edit_node:", e)

# ----------------------------
# Export
# ----------------------------
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# Metadata
WEB_DIRECTORY = "./web"
__version__ = "1.7.0"   # Обновлено из-за добавления Gemini 3.1 Flash Image Preview