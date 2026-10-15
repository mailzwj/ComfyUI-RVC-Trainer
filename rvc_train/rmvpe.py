"""RMVPE pitch estimator used for RVC f0 extraction.

Uses the converted ``rmvpe.safetensors`` checkpoint that ComfyUI-RVC loads, including
its stored mel filter bank, so extracted pitch matches that node pack's pitch path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.utils


FORMAT_KEY = "rvc_rmvpe.format_version"
SAMPLE_RATE = 16000
HOP_LENGTH = 160


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers, device=None, dtype=None):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = nn.Sequential(
            operations.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False, device=device, dtype=dtype),
            operations.BatchNorm2d(out_channels, momentum=0.01, device=device, dtype=dtype),
            nn.ReLU(),
            operations.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False, device=device, dtype=dtype),
            operations.BatchNorm2d(out_channels, momentum=0.01, device=device, dtype=dtype),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = operations.Conv2d(in_channels, out_channels, (1, 1), device=device, dtype=dtype)

    def forward(self, x):
        if hasattr(self, "shortcut"):
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, device=device, dtype=dtype, operations=operations))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, device=device, dtype=dtype, operations=operations))
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if hasattr(self, "pool"):
            return x, self.pool(x)
        return x


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, device=device, dtype=dtype, operations=operations))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, device=device, dtype=dtype, operations=operations))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, device=None, dtype=None, operations=None):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            operations.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False, device=device, dtype=dtype),
            operations.BatchNorm2d(out_channels, momentum=0.01, device=device, dtype=dtype),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, device=device, dtype=dtype, operations=operations))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, device=device, dtype=dtype, operations=operations))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, device=device, dtype=dtype, operations=operations))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class RMVPEModel(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.unet = nn.Module()
        self.unet.encoder = nn.Module()
        self.unet.encoder.bn = operations.BatchNorm2d(1, momentum=0.01, device=device, dtype=dtype)
        self.unet.encoder.layers = nn.ModuleList(
            ResEncoderBlock(cin, cout, (2, 2), 4, device=device, dtype=dtype, operations=operations)
            for cin, cout in [(1, 16), (16, 32), (32, 64), (64, 128), (128, 256)])
        self.unet.intermediate = Intermediate(256, 512, 4, 4, device=device, dtype=dtype, operations=operations)
        self.unet.decoder = Decoder(512, 5, (2, 2), 4, device=device, dtype=dtype, operations=operations)
        self.cnn = operations.Conv2d(16, 3, (3, 3), padding=(1, 1), device=device, dtype=dtype)
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, 1, device=device, dtype=dtype),
            operations.Linear(512, 360, device=device, dtype=dtype),
            nn.Identity(),
            nn.Sigmoid(),
        )
        self.cents_mapping = torch.arange(360).float().mul_(20).add_(1997.3794084376191)

    def forward(self, mel):
        x = mel.transpose(-1, -2).unsqueeze(1)
        enc = self.unet.encoder
        x = enc.bn(x)
        concats = []
        for layer in enc.layers:
            t, x = layer(x)
            concats.append(t)
        x = self.unet.intermediate(x)
        x = self.unet.decoder(x, concats)
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        return self.fc(x)

    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        return self(mel)[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        center = hidden.argmax(-1)
        padded = F.pad(hidden, (4, 4))
        off = torch.arange(9, device=hidden.device)
        win = padded.gather(-1, (center.unsqueeze(-1) + off).clamp(0, padded.shape[-1] - 1))
        mapping = self.cents_mapping.to(hidden.device, hidden.dtype)
        win_cents = mapping[(center.unsqueeze(-1) + off - 4).clamp(0, 359)]
        cents = (win * win_cents).sum(-1) / win.sum(-1)
        conf = hidden.amax(-1)
        cents = cents.masked_fill(conf <= thred, 0)
        f0 = 10 * torch.pow(2.0, cents / 1200)
        return f0.masked_fill(cents == 0, 0)


def log_mel_spectrogram(audio, mel_basis, n_fft=1024, hop_length=HOP_LENGTH, win_length=1024, clamp=1e-5):
    window = torch.hann_window(win_length).to(audio.device, audio.dtype)
    fft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                     window=window, center=True, return_complex=True)
    magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
    mel = torch.matmul(mel_basis.to(audio.device, audio.dtype), magnitude)
    return torch.log(torch.clamp(mel, min=clamp))


def load_rmvpe(path, device, dtype=torch.float32):
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    if metadata.get(FORMAT_KEY) != "1" or "fc.1.weight" not in state_dict:
        raise ValueError(
            "%s is not a converted RVC RMVPE checkpoint (expected %s=1). "
            "Use an rmvpe.safetensors converted for ComfyUI-RVC." % (path, FORMAT_KEY)
        )
    mel_basis = state_dict.pop("rvc.mel_basis").to(dtype=torch.float32)
    model = RMVPEModel(device=device, dtype=dtype, operations=comfy.ops.disable_weight_init)
    model.load_state_dict(state_dict, strict=True)
    # comfy.ops assigns checkpoint tensors as they were loaded, so with comfy-aimdo
    # (DynamicVRAM) the weights stay on CPU until an explicit move.
    model = model.to(device=device, dtype=dtype).eval()
    model.mel_basis = mel_basis
    return model


def infer_f0(model, audio16k, thred=0.03):
    """Pitch in Hz per 160-sample frame for a (1, T) mono 16 kHz waveform."""
    dtype = next(model.parameters()).dtype
    mel = log_mel_spectrogram(audio16k.float(), model.mel_basis.float())
    with torch.inference_mode():
        hidden = model.mel2hidden(mel.to(dtype=dtype))
        f0 = model.decode(hidden.float(), thred=thred)
    return f0.squeeze(0).float().cpu().numpy()
