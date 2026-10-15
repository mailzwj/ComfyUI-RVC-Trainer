"""Spectrogram/mel math plus f0 and HuBERT feature extraction."""

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

from . import hubert, rmvpe


F0_MIN = 50.0
F0_MAX = 1100.0
F0_BIN = 256
AUDIO_SR = 16000
HOP_16K = 160


class MelProcessor:
    """Linear/mel spectrogram conversion for one model config."""

    def __init__(self, filter_length, num_mels, sampling_rate, fmin, fmax, hop_length, win_length, device):
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = torch.from_numpy(mel).to(device=device, dtype=torch.float32)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

    def spectrogram(self, y):
        pad = int((self.filter_length - self.hop_length) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device, dtype=y.dtype),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 2e-7)

    def spec_to_mel(self, spec):
        mel = torch.matmul(self.mel_basis.to(dtype=spec.dtype, device=spec.device), spec)
        return torch.log(torch.clamp(mel, min=2e-6))

    def mel(self, y):
        return self.spec_to_mel(self.spectrogram(y))


def fill_unvoiced(f0):
    """Interpolate unvoiced frames in place. Returns None for an entirely unvoiced clip."""
    voiced = f0 != 0
    if not voiced.any():
        return None
    if not voiced.all():
        f0[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], f0[voiced])
    return f0


def coarse_f0(f0):
    """Quantize Hz into the 1..255 pitch bins the model embedding uses."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + F0_MIN / 700)
    f0_mel_max = 1127 * np.log(1 + F0_MAX / 700)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * (F0_BIN - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
    return np.rint(f0_mel).astype(int)


def extract_f0(audio16k, method, pitch_model=None):
    """Pitch in Hz per 160 samples for a 1-D float32 numpy 16 kHz waveform."""
    if method == "rmvpe":
        device = next(pitch_model.parameters()).device
        wav = torch.from_numpy(np.ascontiguousarray(audio16k)).float().unsqueeze(0).to(device)
        return rmvpe.infer_f0(pitch_model, wav)
    if method == "pm":
        try:
            import parselmouth
        except ImportError as error:
            raise RuntimeError("The pm f0 method needs praat-parselmouth: pip install praat-parselmouth") from error
        p_len = audio16k.shape[0] // HOP_16K
        f0 = (
            parselmouth.Sound(audio16k, AUDIO_SR)
            .to_pitch_ac(
                time_step=HOP_16K / AUDIO_SR,
                voicing_threshold=0.6,
                pitch_floor=int(F0_MIN),
                pitch_ceiling=int(F0_MAX),
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return np.asarray(f0, dtype=np.float32)
    raise ValueError("Unsupported f0 method: %s" % method)


def extract_hubert(audio16k, version, hubert_model):
    """Per-frame phone features of shape (frames, 256) for v1 or (frames, 768) for v2."""
    dtype = next(hubert_model.parameters()).dtype
    device = next(hubert_model.parameters()).device
    wav = torch.from_numpy(np.ascontiguousarray(audio16k)).float().unsqueeze(0).to(device=device, dtype=dtype)
    feats = hubert.extract_features(hubert_model, wav, version)
    return feats.squeeze(0).float().cpu().numpy()
