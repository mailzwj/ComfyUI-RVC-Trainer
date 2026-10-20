import os

CUI = r"D:\Workspace\ComfyUI"

VOICE_V2 = CUI + r"\models\rvc\my-vocals-v2-48k.pth"
VOICE_OTHER = CUI + r"\models\rvc\zi-lai-ye\zilaiye.pth"
VOICE_BAD = CUI + r"\models\rvc\pretrained_v2\f0G48k.pth"
HUBERT = CUI + r"\models\rvc\hubert_base.safetensors"
RMVPE = CUI + r"\models\rvc\rmvpe.safetensors"
FCPE = CUI + r"\models\rvc\fcpe.safetensors"


def main():
    import torch
    import comfy.model_management as mm
    from rvc_inference.loader import load_voice, load_pitch, load_hubert, PitchModel

    device = mm.get_torch_device()

    v = load_voice(VOICE_V2, device)
    assert v.voice_version == "v2" and isinstance(v.voice_tgt_sr, int) and v.voice_tgt_sr == 48000, (v.voice_version, v.voice_tgt_sr)
    n_spk = torch.load(VOICE_V2, map_location="cpu", weights_only=True)["weight"]["emb_g.weight"].shape[0]
    assert v.voice_n_spk == n_spk, (v.voice_n_spk, n_spk)  # probed: my-vocals-v2-48k.pth has 109 slots
    assert v.voice_if_f0 == 1

    v2 = load_voice(VOICE_OTHER, device)
    assert v2.voice_n_spk >= 1

    try:
        load_voice(VOICE_BAD, device)
        raise AssertionError("pretrained base model must be rejected")
    except RuntimeError as e:
        assert "not an RVC voice checkpoint" in str(e)

    h = load_hubert(HUBERT, device)
    p = load_pitch(RMVPE, device)
    assert isinstance(p, PitchModel) and p.pitch_kind == "rmvpe"
    p2 = load_pitch(FCPE, device)
    assert p2.pitch_kind == "fcpe"

    # pitch loaders sanity: one forward pass on a 1-second sine at 16k
    t = torch.arange(16000, dtype=torch.float32) / 16000
    wav = (0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)
    if wav.device.type != device.type:
        wav = wav.to(device)
    # Observed f0 shapes on the 1s/16k sine: rmvpe -> numpy (101,) flattened
    # frames (infer_f0 squeezes batch, returns .cpu().numpy()); fcpe -> torch
    # (1, 101) Mel frames. Both ~101 frames = 16000//160+1. Both medians land
    # near 220 Hz for the 220 Hz sine.
    from rvc_train.rmvpe import infer_f0
    with torch.inference_mode():
        f0_rmvpe = infer_f0(p.module, wav, thred=0.03)  # numpy (101,)
        f0_fcpe = p2.module(wav, threshold=0.006)       # torch (1, 101)
        for tag, f0 in (("rmvpe", f0_rmvpe), ("fcpe", f0_fcpe)):
            f0t = torch.as_tensor(f0)
            assert torch.isfinite(f0t).all(), tag
            med = float(f0t[f0t > 0].abs().median())
            assert abs(med - 220) < 30, (tag, med)
            print("PASS: %s f0 median %.2f Hz, shape %s" % (tag, med, tuple(f0t.shape)))
    print("PASS: loaders")

if __name__ == "__main__":
    main()
