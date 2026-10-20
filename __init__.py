import os

import folder_paths

# ComfyUI-RVC registers the same folder; adding it here keeps this node pack usable alone.
folder_paths.add_model_folder_path("rvc", os.path.join(folder_paths.models_dir, "rvc"))

from .nodes_rvc_trainer import NODE_CLASS_MAPPINGS as _TRAIN_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as _TRAIN_NODE_DISPLAY_NAME_MAPPINGS
from .nodes_rvc_inference import NODE_CLASS_MAPPINGS as _INFERENCE_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as _INFERENCE_NODE_DISPLAY_NAME_MAPPINGS


NODE_CLASS_MAPPINGS = {**_TRAIN_NODE_CLASS_MAPPINGS, **_INFERENCE_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_TRAIN_NODE_DISPLAY_NAME_MAPPINGS, **_INFERENCE_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
