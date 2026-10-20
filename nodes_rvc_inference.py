"""ComfyUI nodes for RVC voice conversion inference."""

import logging
import os
import re

import comfy.audio
import comfy.model_management
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

try:
    from .rvc_inference.index_loader import load_index
    from .rvc_inference.loader import load_hubert, load_pitch, load_voice
    from .rvc_inference.pipeline import RVCPipeline
except ImportError:  # pragma: no cover - absolute form when the pack is on sys.path
    from rvc_inference.index_loader import load_index
    from rvc_inference.loader import load_hubert, load_pitch, load_voice
    from rvc_inference.pipeline import RVCPipeline


LOGGER = logging.getLogger("RVC-Inference")

CATEGORY = "RVC/inference"
RVC_VOICE = io.Custom("RVC_VOICE")
RVC_INDEX = io.Custom("RVC_INDEX")
RVC_HUBERT = io.Custom("RVC_HUBERT")
RVC_PITCH = io.Custom("RVC_PITCH")

NONE_OPTION = "(none)"
PRETRAINED_NAME = re.compile(r"^(f0?[GDF])(\d+k)?\.pth$", re.IGNORECASE)


def _rvc_files():
    return folder_paths.get_filename_list("rvc")


def _is_pretrained(rel):
    if any(part.lower().startswith("pretrained") for part in rel.replace("/", os.sep).split(os.sep)):
        return True
    return bool(PRETRAINED_NAME.match(os.path.basename(rel)))


def _voice_options(files):
    return [rel for rel in files if rel.lower().endswith(".pth") and not _is_pretrained(rel)]


def _index_options(files):
    opts = [rel for rel in files if rel.lower().endswith((".index", ".npy"))]
    return [NONE_OPTION] + sorted(opts, key=os.path.basename)


def _base_model_options(files, kind):
    keys = ("hubert",) if kind == "hubert" else ("rmvpe", "fcpe")
    result = []
    for rel in files:
        if not rel.lower().endswith(".safetensors"):
            continue
        base = os.path.basename(rel).lower()
        if any(k in base for k in keys):
            result.append(rel)
    return sorted(result)


def _default_of(options, wanted):
    for option in options:
        if os.path.basename(option).lower() == wanted:
            return option
    return options[0] if options else wanted


def _resolve(name):
    return folder_paths.get_full_path_or_raise("rvc", name)


class RVCVoiceLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCVoiceLoader",
            display_name="RVC Voice Loader",
            category=CATEGORY,
            description="Load an RVC voice model (.pth) and optional retrieval index from models/rvc.",
            inputs=[
                io.Combo.Input("voice_pth", options=_voice_options(_rvc_files()), tooltip="RVC voice checkpoint (.pth)."),
                io.Combo.Input("index_file", options=_index_options(_rvc_files()), default=NONE_OPTION,
                               tooltip="Optional retrieval index (.index needs faiss-cpu; .npy works everywhere)."),
            ],
            outputs=[RVC_VOICE.Output(display_name="voice"), RVC_INDEX.Output(display_name="index")],
        )

    @classmethod
    def execute(cls, voice_pth, index_file=NONE_OPTION):
        device = comfy.model_management.get_torch_device()
        voice = load_voice(_resolve(voice_pth), device)
        index = load_index(_resolve(index_file)) if index_file != NONE_OPTION else None
        return io.NodeOutput(voice, index)


class RVCHubertPitchLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = _base_model_options(_rvc_files(), "hubert")
        pitch_files = _base_model_options(_rvc_files(), "pitch")
        return io.Schema(
            node_id="RVCHubertPitchLoader",
            display_name="RVC Hubert & Pitch Loader",
            category=CATEGORY,
            description="Load the ContentVec hubert encoder and the RMVPE/FCPE pitch model.",
            inputs=[
                io.Combo.Input("hubert_file", options=files, default=_default_of(files, "hubert_base.safetensors")),
                io.Combo.Input("pitch_file", options=pitch_files, default=_default_of(pitch_files, "rmvpe.safetensors")),
            ],
            outputs=[RVC_HUBERT.Output(display_name="hubert"), RVC_PITCH.Output(display_name="pitch")],
        )

    @classmethod
    def execute(cls, hubert_file, pitch_file):
        device = comfy.model_management.get_torch_device()
        hubert = load_hubert(_resolve(hubert_file), device)
        pitch = load_pitch(_resolve(pitch_file), device)
        return io.NodeOutput(hubert, pitch)


class RVCExecute(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCExecute",
            display_name="RVC Execute",
            category=CATEGORY,
            description="Convert the connected audio with the loaded voice.",
            inputs=[
                RVC_VOICE.Input("voice"),
                RVC_HUBERT.Input("hubert"),
                RVC_PITCH.Input("pitch"),
                io.Audio.Input("audio"),
                RVC_INDEX.Input("index", optional=True, tooltip="Optional retrieval index."),
                io.Combo.Input("f0_method", options=["rmvpe", "fcpe"], default="rmvpe"),
                io.Int.Input("f0_up_key", default=0, min=-12, max=12, step=1),
                io.Int.Input("speaker_id", default=0, min=0, max=109, step=1),
                io.Float.Input("index_rate", default=0.75, min=0.0, max=1.0, step=0.05),
                io.Float.Input("protect", default=0.33, min=0.0, max=0.5, step=0.01),
                io.Int.Input("filter_radius", default=3, min=0, max=10, step=1),
                io.Float.Input("rms_mix_rate", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("resample_sr", default=0, min=0, max=96000, step=1000,
                             tooltip="Output sample rate. 0 keeps the voice's own rate."),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, voice, hubert, pitch, audio, index=None, f0_method="rmvpe", f0_up_key=0,
                speaker_id=0, index_rate=0.75, protect=0.33, filter_radius=3,
                rms_mix_rate=1.0, resample_sr=0):
        load_device = comfy.model_management.get_torch_device()
        if_f0 = int(getattr(voice, "voice_if_f0", 1))
        version = voice.voice_version
        tgt_sr = int(voice.voice_tgt_sr)
        sid = max(0, min(int(speaker_id), int(getattr(voice, "voice_n_spk", 1)) - 1))
        if if_f0 == 1 and f0_method != pitch.pitch_kind:
            raise RuntimeError("ERROR: f0_method=%s but the pitch model is %s." % (f0_method, pitch.pitch_kind))

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sr != 16000:
            waveform = comfy.audio.resample(waveform, sr, 16000)

        cuda = load_device.type == "cuda"
        pipe = RVCPipeline(tgt_sr, **({"x_pad": 3, "x_query": 10, "x_center": 60, "x_max": 65} if cuda
                                      else {"x_pad": 1, "x_query": 6, "x_center": 38, "x_max": 41}))
        outs = []
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                for b in range(waveform.shape[0]):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    monos = waveform[b].reshape(-1).float().cpu()
                    out, out_sr = pipe.convert(hubert, voice, torch.tensor([[sid]]), pitch, index,
                                               index_rate, version, monos, int(f0_up_key), if_f0,
                                               float(protect), int(filter_radius), float(rms_mix_rate),
                                               int(resample_sr) or tgt_sr,
                                               _resample_fn, load_device)
                    outs.append(out.unsqueeze(0))
                    comfy.model_management.throw_exception_if_processing_interrupted()
        finally:
            comfy.model_management.soft_empty_cache()
        wave = torch.stack(outs) if len(outs) > 1 else outs[0].unsqueeze(0)
        return io.NodeOutput({"waveform": wave.unsqueeze(1) if wave.dim() == 2 else wave, "sample_rate": out_sr})


def _resample_fn(w, a, b):
    return comfy.audio.resample(w[None, None, :], a, b)[0, 0]


class RVCAudioSynthesis(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.Audio.Input("audio"), prefix="audio", min=1, max=100)
        return io.Schema(
            node_id="RVCAudioSynthesis",
            display_name="RVC Audio Synthesis",
            category=CATEGORY,
            description="Combine multiple audio clips: merge them or concatenate in order.",
            inputs=[
                io.Autogrow.Input("audios", template=template, tooltip="Clips to combine, in connection order."),
                io.Combo.Input("mode", options=["merge", "concat"], default="concat"),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audios, mode="concat"):
        names = sorted(audios, key=lambda n: int(re.search(r"(\d+)$", n).group(1)))
        clips = [audios[n] for n in names if audios[n] is not None]
        if not clips:
            raise ValueError("Connect at least one audio")

        target = max(c["sample_rate"] for c in clips)
        waves = [comfy.audio.resample(c["waveform"], c["sample_rate"], target) for c in clips]
        out_wave = torch.cat(waves, dim=-1) if mode == "concat" else _merge(waves)
        return io.NodeOutput({"waveform": out_wave, "sample_rate": target})


def _merge(waves):
    width = max(w.shape[-1] for w in waves)
    acc = None
    for w in waves:
        if w.shape[-1] != width:
            w = torch.nn.functional.pad(w, (0, width - w.shape[-1]))
        acc = w.float() if acc is None else acc + w.float()
    out = acc / len(waves)
    peak = out.abs().max() / 0.99
    if peak > 1:
        out = out / peak
    return out


NODE_CLASS_MAPPINGS = {
    "RVCVoiceLoader": RVCVoiceLoader,
    "RVCHubertPitchLoader": RVCHubertPitchLoader,
    "RVCExecute": RVCExecute,
    "RVCAudioSynthesis": RVCAudioSynthesis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVCVoiceLoader": "RVC Voice Loader",
    "RVCHubertPitchLoader": "RVC Hubert & Pitch Loader",
    "RVCExecute": "RVC Execute",
    "RVCAudioSynthesis": "RVC Audio Synthesis",
}
