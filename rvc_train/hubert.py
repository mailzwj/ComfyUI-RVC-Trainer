"""HuBERT/ContentVec encoder used to produce RVC phone features.

Checkpoints are the converted ``hubert_base.safetensors`` files that ComfyUI-RVC loads,
so features extracted here match what that node pack feeds to a trained voice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.utils
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device


CONV_SPECS = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
              (512, 3, 2), (512, 2, 2), (512, 2, 2)]
N_LAYERS = 12
DIM = 768
HEADS = 12
FFN = 3072

FORMAT_KEY = "rvc_hubert.format_version"


class HubertSelfAttention(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.q_proj = operations.Linear(DIM, DIM, bias=True, device=device, dtype=dtype)
        self.k_proj = operations.Linear(DIM, DIM, bias=True, device=device, dtype=dtype)
        self.v_proj = operations.Linear(DIM, DIM, bias=True, device=device, dtype=dtype)
        self.out_proj = operations.Linear(DIM, DIM, bias=True, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        bsz, tlen, _ = x.shape
        q = self.q_proj(x).view(bsz, tlen, HEADS, -1).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tlen, HEADS, -1).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tlen, HEADS, -1).transpose(1, 2)
        aq, ak, av = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        return self.out_proj(attention(aq, ak, av, HEADS, mask=mask, skip_reshape=True, preferred_attention=self.comfy_attention))


class HubertEncoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = HubertSelfAttention(device=device, dtype=dtype, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(DIM, eps=1e-5, device=device, dtype=dtype)
        self.fc1 = operations.Linear(DIM, FFN, bias=True, device=device, dtype=dtype)
        self.fc2 = operations.Linear(FFN, DIM, bias=True, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(DIM, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        x = x + self.self_attn(x, mask)
        x = self.self_attn_layer_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(x)))
        return self.final_layer_norm(x)


class HubertEncoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        conv_layers = []
        for i, (dim, k, s) in enumerate(CONV_SPECS):
            conv = operations.Conv1d(1 if i == 0 else 512, dim, k, stride=s, bias=False, device=device, dtype=dtype)
            if i == 0:
                conv_layers.append(nn.Sequential(conv, nn.Identity(), operations.GroupNorm(512, 512, device=device, dtype=dtype)))
            else:
                conv_layers.append(nn.Sequential(conv))
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm = operations.LayerNorm(512, eps=1e-5, device=device, dtype=dtype)
        self.post_extract_proj = operations.Linear(512, DIM, bias=True, device=device, dtype=dtype)
        self.encoder = nn.Module()
        pos_conv = nn.Conv1d(DIM, DIM, 128, padding=64, groups=16, device=device, dtype=dtype)
        self.encoder.pos_conv = nn.Sequential(nn.utils.weight_norm(pos_conv, name="weight", dim=2))
        self.encoder.layer_norm = operations.LayerNorm(DIM, eps=1e-5, device=device, dtype=dtype)
        self.encoder.layers = nn.ModuleList(HubertEncoderLayer(device=device, dtype=dtype, operations=operations) for _ in range(N_LAYERS))
        self.final_proj = operations.Linear(DIM, 256, bias=True, device=device, dtype=dtype)

    def forward(self, wav):
        h = wav.unsqueeze(1)
        for conv_layer in self.feature_extractor.conv_layers:
            h = F.gelu(conv_layer(h))
        h = self.post_extract_proj(self.layer_norm(h.transpose(1, 2)))
        p = self.encoder.pos_conv[0](h.transpose(1, 2))[:, :, :-1]
        h = self.encoder.layer_norm(h + F.gelu(p.transpose(1, 2)))
        states = []
        for layer in self.encoder.layers:
            h = layer(h)
            states.append(h)
        return states


def load_hubert(path, device, dtype=torch.float32):
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    if metadata.get(FORMAT_KEY) != "1" or "post_extract_proj.weight" not in state_dict:
        raise ValueError(
            "%s is not a converted RVC Hubert checkpoint (expected %s=1). "
            "Use a hubert_base.safetensors converted for ComfyUI-RVC." % (path, FORMAT_KEY)
        )
    model = HubertEncoder(device=device, dtype=dtype, operations=comfy.ops.disable_weight_init)
    model.load_state_dict(state_dict, strict=True)
    # comfy.ops assigns checkpoint tensors as they were loaded, so with comfy-aimdo
    # (DynamicVRAM) the weights stay on CPU until an explicit move.
    return model.to(device=device, dtype=dtype).eval()


def extract_features(model, wav16, version):
    """Per-frame phone features for a 16 kHz mono waveform of shape (1, T)."""
    if version not in ("v1", "v2"):
        raise ValueError("Unsupported RVC feature version: %s" % version)
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        states = model(wav16.to(dtype=dtype))
        return model.final_proj(states[8]) if version == "v1" else states[11]
