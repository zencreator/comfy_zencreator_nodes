"""
ZenCreator Custom Nodes for ComfyUI
BytePlus Seedream 4.0 & Fal.ai Gemini
"""

# 1. Импортируем маппинги из первого файла (BytePlus) и даем им псевдонимы (as ...), 
# чтобы имена не конфликтовали
from .BytePlusSeedreamNode import NODE_CLASS_MAPPINGS as BytePlus_Classes, NODE_DISPLAY_NAME_MAPPINGS as BytePlus_Names

# 2. Импортируем маппинги из вашего нового файла (Fal.ai)
# ВАЖНО: Имя файла должно точно совпадать. У вас он называется NanoBanabaFALAI
from .NanoBanabaFALAI import NODE_CLASS_MAPPINGS as Fal_Classes, NODE_DISPLAY_NAME_MAPPINGS as Fal_Names

# 3. Объединяем словари в общие переменные
NODE_CLASS_MAPPINGS = {**BytePlus_Classes, **Fal_Classes}
NODE_DISPLAY_NAME_MAPPINGS = {**BytePlus_Names, **Fal_Names}

# 4. Экспортируем объединенные словари
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Metadata
WEB_DIRECTORY = "./web"
__version__ = "1.1.0"
