"""Loaders for RVC voice .pth checkpoints and the content/pitch encoders."""

import os

import torch

# The trainer pack is loaded by ComfyUI under a hyphenated module name, so its
# top-level packages are NOT importable by absolute name at runtime; reach
# rvc_train through the package hierarchy, with an absolute fallback for
# direct-layout imports (pack dir legitimately on sys.path).
try:
    from rvc_train.models import sr2sr  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - pack loaded under a hyphenated name
    from ..rvc_train.models import sr2sr


class PitchModel:
    """Bundle of an RMVPE or FCPE model, tagged with its kind, for the node graph."""

    def __init__(self, kind, module):
        self.pitch_kind = kind
        self.module = module
        self.dtype = next(module.parameters()).dtype


def load_voice(path, device):
    """Validate an upstream RVC voice checkpoint and build its synthesizer.

    Rejects pretrained base models (G_*.pth / f0G48k.pth style) that carry no
    'weight'/'config' inference payload, with a message pointing at re-exporting.
    """
    try:
        from .synths import SYNTH_CLASSES
    except ImportError:  # pragma: no cover - absolute form when the pack is on sys.path
        from rvc_inference.synths import SYNTH_CLASSES

    if not os.path.isfile(path):
        raise FileNotFoundError("RVC voice model not found: %s" % path)
    cpt = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt
            or len(cpt["config"]) != 18):
        raise RuntimeError("ERROR: not an RVC voice checkpoint: %s" % path)
    version = cpt.get("version", "v1")
    if_f0 = int(cpt.get("f0", 1))
    cls = SYNTH_CLASSES.get((version, if_f0))
    if cls is None:
        raise RuntimeError("ERROR: unknown RVC voice version/f0: %s/%s" % (version, if_f0))
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    weight = {k: v for k, v in cpt["weight"].items() if "enc_q" not in k}
    try:
        model = cls(*config)
    except TypeError as e:
        raise RuntimeError("ERROR: unsupported RVC voice config in %s: %s" % (path, e))
    missing, unexpected = model.load_state_dict(weight, strict=False)
    if unexpected or missing:
        raise RuntimeError(
            "ERROR: voice checkpoint keys do not match the synthesizer layout: %s "
            "(missing=%s unexpected=%s)" % (path, list(missing)[:5], list(unexpected)[:5]))
    model = model.to(device).eval()
    model.voice_version = version
    model.voice_if_f0 = if_f0
    # Normalize the string sample-rate configs ("48k") to the numeric sr (48000).
    tgt = getattr(model, "voice_tgt_sr", config[-1])
    model.voice_tgt_sr = sr2sr[tgt] if isinstance(tgt, str) else int(tgt)
    model.voice_n_spk = config[-3]
    model.voice_path = path
    return model


def load_hubert(path, device):
    """Load the ContentVec hubert encoder via the trainer's loader."""
    try:
        from rvc_train.hubert import load_hubert as _load
    except ImportError:  # pragma: no cover - pack loaded under a hyphenated name
        from ..rvc_train.hubert import load_hubert as _load
    return _load(path, device)


def load_pitch(path, device):
    """Load rmvpe.safetensors or fcpe.safetensors; kind detected from the filename."""
    name = os.path.basename(path).lower()
    if "fcpe" in name:
        import comfy.utils

        try:
            from .fcpe import FCPitch
        except ImportError:  # pragma: no cover - absolute form when the pack is on sys.path
            from rvc_inference.fcpe import FCPitch
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = FCPitch(device=device, dtype=torch.float32, operations=torch.nn)
        model.load_state_dict(sd, strict=True)
        return PitchModel("fcpe", model.to(device).eval())
    if "rmvpe" in name:
        try:
            from rvc_train.rmvpe import load_rmvpe as _load
        except ImportError:  # pragma: no cover - pack loaded under a hyphenated name
            from ..rvc_train.rmvpe import load_rmvpe as _load
        return PitchModel("rmvpe", _load(path, device))
    raise RuntimeError(
        "ERROR: %s is not an RVC pitch checkpoint (expected 'rmvpe' or 'fcpe' in the filename)" % path)
