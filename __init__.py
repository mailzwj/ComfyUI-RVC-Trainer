import os

import folder_paths

# ComfyUI-RVC registers the same folder; adding it here keeps this node pack usable alone.
folder_paths.add_model_folder_path("rvc", os.path.join(folder_paths.models_dir, "rvc"))

from .nodes_rvc_trainer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
