"""FCPE pitch estimator, vendored from ComfyUI-RVC's rvc/fcpe.py.

Layer constructions take an ``operations`` kwarg; callers pass ``torch.nn`` so
the module does not depend on comfy.ops at import time. The state_dict loader
fuses upstream's weight_norm parametrised output projection and renames the
stored mel filter bank key, so converted fcpe safetensors load strict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dims = (dim0, dim1)

    def forward(self, x):
        return x.transpose(*self.dims)


class _DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, padding, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv(x)


class _ConformerConvModule(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        inner_dim = dim * 2
        self.net = nn.Sequential(
            operations.LayerNorm(dim, device=device, dtype=dtype),
            _Transpose(1, 2),
            operations.Conv1d(dim, inner_dim * 2, 1, device=device, dtype=dtype),
            nn.GLU(dim=1),
            _DepthWiseConv1d(inner_dim, 31, 15, device=device, dtype=dtype, operations=operations),
            nn.SiLU(),
            operations.Conv1d(inner_dim, dim, 1, device=device, dtype=dtype),
            _Transpose(1, 2),
            nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class _EncoderLayer(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.conformer = _ConformerConvModule(dim, device=device, dtype=dtype, operations=operations)
        # Unused on the conv_only path, kept so checkpoint keys still load.
        self.norm = operations.LayerNorm(dim, device=device, dtype=dtype)

    def forward(self, x):
        return x + self.conformer(x)


class _Encoder(nn.Module):
    def __init__(self, n_layers, dim, conv_only, device=None, dtype=None, operations=None):
        super().__init__()
        assert conv_only, "only conv_only FCPE is supported"
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(dim, device=device, dtype=dtype, operations=operations)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class FCPitch(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, out_dims=360, hidden_dims=512, n_layers=6, conv_only=True, f0_min=32.7, f0_max=1975.5):
        super().__init__()
        assert conv_only, "only conv_only FCPE is supported"
        self.out_dims = out_dims
        self.threshold = 0.006
        self.input_stack = nn.Sequential(
            operations.Conv1d(128, hidden_dims, 3, 1, 1, device=device, dtype=dtype),
            operations.GroupNorm(4, hidden_dims, device=device, dtype=dtype),
            nn.LeakyReLU(),
            operations.Conv1d(hidden_dims, hidden_dims, 3, 1, 1, device=device, dtype=dtype),
        )
        self.net = _Encoder(n_layers, hidden_dims, conv_only, device=device, dtype=dtype, operations=operations)
        self.norm = operations.LayerNorm(hidden_dims, device=device, dtype=dtype)
        self.output_proj = operations.Linear(hidden_dims, out_dims, device=device, dtype=dtype)
        lo = 1200.0 * torch.log2(torch.tensor(f0_min / 10.0))
        hi = 1200.0 * torch.log2(torch.tensor(f0_max / 10.0))
        self.register_buffer("cent_table", torch.linspace(lo, hi, out_dims).to(device=device))
        self.register_buffer("gaussian_blurred_cent_mask", (1200.0 * torch.log2(torch.tensor([f0_max / 10.0])))[0].to(device=device))
        self.register_buffer("mel_basis", torch.empty(128, 513, device=device))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        sd = dict(state_dict)
        if "output_proj.weight_g" in sd and "output_proj.weight_v" in sd:
            g = sd.pop("output_proj.weight_g")
            v = sd.pop("output_proj.weight_v")
            # Same fused kernel upstream's weight_norm parametrization runs; naive g*v/||v|| differs by ~1e-7.
            sd["output_proj.weight"] = torch._weight_norm(v, g, 0)
        if "rvc.mel_basis" in sd and "mel_basis" not in sd:
            sd["mel_basis"] = sd.pop("rvc.mel_basis")
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def _wav_to_mel(self, wav, key_shift=0, speed=1):
        factor = 2.0 ** (key_shift / 12.0)
        win_new = int(round(1024 * factor))
        n_fft_new = int(round(1024 * factor))
        hop_new = int(round(160 * speed))
        window = torch.hann_window(win_new, device=wav.device, dtype=wav.dtype)
        pad_left = (win_new - hop_new) // 2
        pad_right = max((win_new - hop_new + 1) // 2, win_new - wav.shape[-1] - pad_left)
        mode = "reflect" if pad_right < wav.shape[-1] else "constant"
        y = F.pad(wav.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)
        spec = torch.stft(y, n_fft_new, hop_length=hop_new, win_length=win_new, window=window, center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        if key_shift != 0:
            if mag.shape[1] < 513:
                mag = F.pad(mag, (0, 0, 0, 513 - mag.shape[1]))
            mag = mag[:, :513, :] * 1024 / win_new
        mel = torch.matmul(self.mel_basis.to(device=mag.device, dtype=mag.dtype), mag)
        mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(-1, -2)
        n_frames = wav.shape[-1] // 160 + 1
        if n_frames > mel.shape[1]:
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < mel.shape[1]:
            mel = mel[:, :n_frames, :]
        return mel

    def forward(self, wav, key_shift=0, speed=1, threshold=None):
        if threshold is None:
            threshold = self.threshold
        mel = self._wav_to_mel(wav, key_shift, speed)
        x = self.input_stack(mel.transpose(-1, -2)).transpose(-1, -2)
        x = self.net(x)
        latent = torch.sigmoid(self.output_proj(self.norm(x)))
        batch, frames, _ = latent.shape
        cents = self.cent_table.to(device=latent.device, dtype=latent.dtype)[None, None, :].expand(batch, frames, -1)
        confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
        local_index = (torch.arange(9, device=latent.device) + (max_index - 4)).clamp(0, self.out_dims - 1)
        decoded = torch.sum(torch.gather(cents, -1, local_index) * torch.gather(latent, -1, local_index), dim=-1, keepdim=True) / torch.sum(torch.gather(latent, -1, local_index), dim=-1, keepdim=True)
        decoded = decoded.masked_fill(confidence <= threshold, float("-inf"))
        return (10.0 * torch.pow(2.0, decoded / 1200.0)).squeeze(-1)
