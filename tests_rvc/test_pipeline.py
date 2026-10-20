import torch

CUI = r"D:\Workspace\ComfyUI"
HUBERT = CUI + r"\models\rvc\hubert_base.safetensors"
RMVPE = CUI + r"\models\rvc\rmvpe.safetensors"
VOICE = CUI + r"\models\rvc\my-vocals-v2-48k.pth"
NPY = CUI + r"\models\rvc\my-vocals-v2-48k.npy"


def _resample(w, sr_in, sr_out):
    return torch.nn.functional.interpolate(
        w[None, None, :], size=sr_out, mode="linear").reshape(-1)


def main():
    import numpy as np
    import comfy.model_management as mm
    from rvc_inference.loader import load_voice, load_hubert, load_pitch
    from rvc_inference.index_loader import load_index
    from rvc_inference.pipeline import RVCPipeline, coarse_f0, _f0_of

    device = mm.get_torch_device()
    voice = load_voice(VOICE, device)
    hubert = load_hubert(HUBERT, device)
    pitch = load_pitch(RMVPE, device)
    index = load_index(NPY)

    # unit: coarse_f0 binning sanity
    f0 = torch.tensor([0.0, 100.0, 200.0, 0.0, 300.0])
    coarse, f0bak = coarse_f0(f0, 0, 0)
    assert coarse.dtype == torch.int32, coarse.dtype
    assert ((coarse[1] >= 1) and (coarse[1] <= 255)).item()
    assert coarse.shape == f0.shape
    # gap interp: the unvoiced slots must be finite (interpolated or floor)
    assert torch.isfinite(f0bak).all()

    pipe = RVCPipeline(int(voice.voice_tgt_sr))
    sr_in = 16000
    length = sr_in * 3
    t = torch.arange(length, dtype=torch.float32) / sr_in
    wav = 0.1 * torch.sin(2 * torch.pi * 220 * t)

    with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        out, sr = pipe.convert(hubert, voice, torch.tensor([0]), pitch, index, 0.75, "v2",
                               wav, 0, 1, 0.33, 3, 1.0, 0,
                               resample_fn=_resample,
                               device=device)
    assert sr == 48000, sr
    expected = 3 * sr
    tolerance = sr // 20  # 50 ms
    assert abs(out.shape[0] - expected) <= tolerance, (out.shape[0], expected)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    # secondary: no-index path
    out2, _ = pipe.convert(hubert, voice, torch.tensor([0]), pitch, None, 0.0, "v2",
                           wav, 0, 1, 0.33, 3, 1.0, 0,
                           resample_fn=_resample,
                           device=device)
    assert abs(out2.shape[0] - expected) <= tolerance
    assert torch.isfinite(out2).all()

    # fcpe adapter sanity on real module shapes: same ~N//160+1 frame count as rmvpe
    from rvc_inference.loader import load_pitch as _lp
    fm = _lp(CUI + r"\models\rvc\fcpe.safetensors", device)
    probe = torch.zeros(3 * 16000 + 2 * pipe.t_pad, dtype=torch.float32, device=device)
    fp = _f0_of(fm, probe[None])
    fp_rmvpe = _f0_of(pitch, probe[None])
    p_len = probe.shape[0] // 160
    print("fcpe frames %d, rmvpe frames %d, padded p_len %d"
          % (fp.shape[-1], fp_rmvpe.shape[-1], p_len))
    assert abs(int(fp.shape[-1]) - (p_len + 1)) <= 1
    assert abs(int(fp_rmvpe.shape[-1]) - (p_len + 1)) <= 1
    assert torch.isfinite(fp.float()).all()

    print("PASS: pipeline convert end-to-end")

if __name__ == "__main__":
    main()
