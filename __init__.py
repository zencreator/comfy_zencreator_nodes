"""
ZenCreator Custom Nodes for ComfyUI
BytePlus Seedream 4.0 & Fal.ai & Gemini 3 Pro
"""

# 1. Импортируем маппинги из первого файла (BytePlus)
from .BytePlusSeedreamNode import NODE_CLASS_MAPPINGS as BytePlus_Classes, NODE_DISPLAY_NAME_MAPPINGS as BytePlus_Names

# 2. Импортируем маппинги из второго файла (Fal.ai)
from .NanoBanabaFALAI import NODE_CLASS_MAPPINGS as Fal_Classes, NODE_DISPLAY_NAME_MAPPINGS as Fal_Names

# 3. Импортируем маппинги из НОВОГО файла (Gemini)
# Убедитесь, что имя файла (без .py) совпадает с тем, что вы создали на Шаге 1
from .NanoBananaGemini import NODE_CLASS_MAPPINGS as Gemini_Classes, NODE_DISPLAY_NAME_MAPPINGS as Gemini_Names

# 4. Объединяем все словари в общие переменные
NODE_CLASS_MAPPINGS = {
    **BytePlus_Classes, 
    **Fal_Classes, 
    **Gemini_Classes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BytePlus_Names, 
    **Fal_Names, 
    **Gemini_Names
}

# 5. Экспортируем объединенные словари
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Metadata
WEB_DIRECTORY = "./web"
__version__ = "1.2.0"
