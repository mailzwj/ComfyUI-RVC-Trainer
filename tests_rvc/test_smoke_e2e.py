r"""End-to-end node-level smoke test with real RVC models.

Runs the four inference nodes' execute() classmethods directly (not through a
ComfyUI server), feeding the real checkpoints under models/rvc.

Run (PowerShell):
    $env:PYTHONPATH = "D:\Workspace\ComfyUI;D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC-Trainer"
    & "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc/test_smoke_e2e.py
"""
import os
import sys
import time

CUI = r"D:\Workspace\ComfyUI"
PACK = r"D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC-Trainer"

# The node classes resolve model paths through folder_paths; mirror what the
# pack's __init__.py registers so execute() works without a running server.
for p in (CUI, PACK):
    if p not in sys.path:
        sys.path.insert(0, p)

import folder_paths

folder_paths.add_model_folder_path("rvc", os.path.join(folder_paths.models_dir, "rvc"))


def main():
    import torch
    import comfy.model_management as mm
    from nodes_rvc_inference import (
        RVCVoiceLoader, RVCHubertPitchLoader, RVCExecute, RVCAudioSynthesis,
    )

    # io.NodeOutput probe (plan step 1): expose .result as the accessor
    from comfy_api.latest import io
    probe = io.NodeOutput(1, 2)
    assert probe.result == (1, 2), probe.result

    started = time.perf_counter()
    device = mm.get_torch_device()
    print("device:", device)

    # node 1: real voice + its retrieval index
    out1 = RVCVoiceLoader.execute("my-vocals-v2-48k.pth", "my-vocals-v2-48k.npy")
    voice, index = out1.result
    assert voice.voice_version == "v2", voice.voice_version
    assert voice.voice_tgt_sr == 48000, voice.voice_tgt_sr
    assert getattr(voice, "voice_n_spk", 1) == 109
    assert index is not None and index.kind == "feats", getattr(index, "kind", None)
    assert index.data.dim() == 2 and index.data.shape[1] == 768, index.data.shape
    print("PASS: RVCVoiceLoader: voice v2 @ %d Hz, n_spk %d; index feats %s"
          % (voice.voice_tgt_sr, voice.voice_n_spk, tuple(index.data.shape)))

    # node 2: real hubert + rmvpe
    out2 = RVCHubertPitchLoader.execute("hubert_base.safetensors", "rmvpe.safetensors")
    hubert, pitch = out2.result
    assert hubert is not None
    assert pitch is not None and pitch.pitch_kind == "rmvpe", pitch
    print("PASS: RVCHubertPitchLoader: hubert loaded, pitch kind %s" % pitch.pitch_kind)

    # source audio: 4-second 16 kHz mono sine
    # 2 s would fail here on a CUDA host: RVCExecute's CUDA window pads
    # x_pad=3 (3 * 16000 = 48000 samples) reflectively around the input, and
    # F.pad(mode="reflect") needs the pad to be smaller than the input, so
    # anything at or under 3 s at 16 kHz cannot pass the CUDA window. 4 s clears it.
    sr_in = 16000
    t = torch.arange(sr_in * 4, dtype=torch.float32) / sr_in
    source = (0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0).unsqueeze(0)
    audio = {"waveform": source, "sample_rate": sr_in}

    # node 3: full conversion through the node wrapper
    out3 = RVCExecute.execute(voice, hubert, pitch, audio, index,
                              f0_method="rmvpe", f0_up_key=0, speaker_id=0,
                              index_rate=0.75, protect=0.33, filter_radius=3,
                              rms_mix_rate=1.0, resample_sr=0)
    converted, = out3.result
    out = converted
    assert out["sample_rate"] == 48000, out["sample_rate"]
    assert out["waveform"].dtype == torch.float32, out["waveform"].dtype
    assert out["waveform"].shape[0] == 1 and out["waveform"].shape[1] == 1, out["waveform"].shape
    assert torch.isfinite(out["waveform"]).all()
    length_expected = 4 * 48000  # resampled 4 s at the voice's 48 kHz
    tolerance = 48000 // 20  # 50 ms
    observed = out["waveform"].shape[-1]
    assert abs(observed - length_expected) <= tolerance, (observed, length_expected)
    print("PASS: RVCExecute: M=%d samples @ %d Hz (expected ~%d)"
          % (observed, out["sample_rate"], length_expected))

    # node 4: concat the converted clip with itself, then merge the same pair
    bundle = RVCAudioSynthesis.execute(audios={"audio0": out, "audio1": out}, mode="concat")
    concat, = bundle.result
    assert concat["sample_rate"] == 48000
    assert torch.isfinite(concat["waveform"]).all()
    assert abs(concat["waveform"].shape[-1] - 2 * observed) <= 1000, concat["waveform"].shape

    merged, = RVCAudioSynthesis.execute(audios={"audio0": out, "audio1": out}, mode="merge").result
    # merge aligns and straight-sums takes (it does not join them): two
    # identical takes sum to exactly double amplitude; the 0.99 peak cap only
    # engages when that sum would clip (the converted take peaks ~0.37, so no
    # cap here and the merged waveform must be exactly 2x).
    assert merged["sample_rate"] == out["sample_rate"]
    assert merged["waveform"].shape == out["waveform"].shape, \
        (merged["waveform"].shape, out["waveform"].shape)
    sum_would_clip = 2.0 * out["waveform"].abs().max().item() > 0.99
    expected = out["waveform"] * (0.99 / out["waveform"].abs().max().item()) if sum_would_clip \
        else 2.0 * out["waveform"]
    assert torch.allclose(merged["waveform"], expected, atol=1e-5), \
        "merge of two identical takes must be their peak-capped sum"
    assert torch.isfinite(merged["waveform"]).all()
    print("PASS: RVCAudioSynthesis: concat %d samples, merge %d samples"
          % (concat["waveform"].shape[-1], merged["waveform"].shape[-1]))
    # write output wav to make manual listening possible
    import soundfile as sf
    OUT_DIR = CUI + r"\output\rvc_inference_smoke"
    os.makedirs(OUT_DIR, exist_ok=True)
    sf.write(OUT_DIR + r"\smoke.wav",
             out["waveform"].reshape(-1).numpy(), out["sample_rate"])

    elapsed = time.perf_counter() - started
    print("PASS: end-to-end smoke | runtime %.1fs | output at %s" % (elapsed, OUT_DIR))


if __name__ == "__main__":
    main()
