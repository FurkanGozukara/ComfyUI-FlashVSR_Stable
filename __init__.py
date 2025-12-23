# 1. 'nodes.py' faylından 'FlashVSR' class-ını çağırırıq
from .nodes import FlashVSR

# 2. Burada sol tərəfə "FlashVSR_Stable" yazırıq. 
# Bu, ComfyUI-ın node-u sistemdə tanıdığı unikal ID olacaq.
NODE_CLASS_MAPPINGS = {
    "FlashVSR_Stable": FlashVSR
}

# 3. Bu isə menyuda (Add Node menyusunda) görünən addır.
# İstədiyiniz kimi yaza bilərsiniz, mən mötərizədə Stable yazdım.
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSR_Stable": "FlashVSR (Stable)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
