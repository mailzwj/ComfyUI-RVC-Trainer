import torch

VOICE = r"D:\Workspace\ComfyUI\models\rvc\my-vocals-v2-48k.pth"

# From the checkpoint's config: 48 kHz RVC voices use hop 480 (see report for the
# upsampling-chain evidence: dec output = product(upsample_rates) * input frames).
OUT_SAMPLES_PER_FRAME = 480


def main():
    import comfy.utils  # noqa: F401 (ensures PYTHONPATH has ComfyUI)
    cpt = torch.load(VOICE, map_location="cpu", weights_only=True)
    assert len(cpt["config"]) == 18, len(cpt["config"])

    from rvc_inference.synths import SYNTH_CLASSES
    cls = SYNTH_CLASSES[(cpt["version"], int(cpt.get("f0", 1)))]
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    model = cls(*config)

    sd_keys = set(cpt["weight"].keys())
    mine = set(model.state_dict().keys())
    stripped = {k for k in sd_keys if "enc_q" not in k}
    assert stripped == mine, ["missing", sorted(stripped - mine)[:10], "extra", sorted(mine - stripped)[:10]]

    with torch.no_grad():
        model.load_state_dict({k: v for k, v in cpt["weight"].items() if "enc_q" not in k}, strict=True)
    assert model.voice_version == cpt["version"]
    assert model.voice_tgt_sr == cpt["config"][-1]

    T, D = 100, 768
    # phone is (B, T, D): TextEncoder's Linear applies over the last dim, so the
    # (1, D, T) layout in the task sketch crashes with the same shape error upstream would.
    phone = torch.randn(1, T, D) * 0.01
    length = torch.tensor([T])
    pitch = torch.randint(1, 230, (1, T))
    pitchf = torch.rand(1, T) * 300 + 100
    torch.manual_seed(0)
    # sid must be a 1-D speaker-index tensor (B,): emb_g(sid).unsqueeze(-1) then feeds
    # Conv1d as (B, gin_channels, 1); a 2-D [[0]] index would produce a 4-D g upstream too.
    out = model.infer(phone, length, pitch, pitchf, torch.tensor([0]))[0]
    assert out.dtype == torch.float32
    assert out.shape[0] == 1 and out.shape[1] == 1, out.shape
    # 48 kHz RVC voices use hop 480; infer without slices upsamples T frames to T*hop samples.
    assert out.shape[2] == OUT_SAMPLES_PER_FRAME * T, (out.shape, T)
    assert torch.isfinite(out).all()

    print("PASS: synths match checkpoint keys and infer runs")


if __name__ == "__main__":
    main()
