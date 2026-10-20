"""Conversion pipeline for RVC voice inference.

Structural port of ComfyUI-RVC's rvc/pipeline.py (which itself distills
RVC-Project's infer/modules/vc/pipeline.py): 48 Hz highpass, energy-minima
splitting, per-segment vc() around the t_pad overlap, mel-binned coarse
pitch, RMS envelope mixing and peak normalization. Differences from that
reference are deliberate, to match the trainer-side building blocks:

- f0 is extracted here from the padded audio instead of being handed in
  precomputed: `_f0_of` adapts the two estimator transports (rmvpe returns
  a CPU numpy array via log-mel; fcpe takes the waveform tensor directly,
  FCPE normalizes key_shift/speed into -inf unvoiced rather than zeros).
- `coarse_f0` mirrors upstream get_f0's post-estimation steps, including
  the `filter_radius` median over the T axis that ComfyUI-RVC applies to
  both the coarse bins and the float f0 track.
- hubert features come from rvc_train.hubert.extract_features (the same
  ContentVec checkpoint the reference pack feeds), index blending from
  rvc_inference.index_loader.blend_with_index, and synthesis uses the
  sliced-decode path of rvc_inference.synths (skip_head/return_length2)
  instead of pre-sliced phone tensors.
"""

import functools

import numpy as np
import torch
import torch.nn.functional as F

from rvc_inference.index_loader import blend_with_index


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _highpass_coeffs():
    """Cached butter(N=5, Wn=48, btype='high', fs=16000) coefficients."""
    from scipy import signal  # lazy: scipy already ships with the pinned env

    return signal.butter(N=5, Wn=48, btype="high", fs=16000)


def highpass_48hz(x):
    """Butterworth highpass at 48 Hz on 16 kHz audio (filtfilt semantics).

    Coefficients come from `_highpass_coeffs`; scipy is imported lazily the
    way faiss is in index_loader, and the design is computed once.
    """
    b, a = _highpass_coeffs()
    from scipy import signal  # lazy: scipy already ships with the pinned env
    # scipy's backward-forward pass leaves a reversed (negative-stride) view;
    # copy so torch.from_numpy accepts it.
    y = np.ascontiguousarray(signal.filtfilt(b, a, x.double().cpu().numpy()))
    return torch.from_numpy(y).float().to(x.device)


def _interp_unvoiced(f0: torch.Tensor) -> torch.Tensor:
    """Fill 0 Hz slots by linear interpolation between voiced neighbours.

    Mirrors upstream get_f0's np.interp gap-fill (mel-binning then maps any
    still-empty edges to the floor bin, so the synthesizer never sees 0).
    """
    n = f0.shape[0]
    voiced = f0 != 0
    if not voiced.any() or not (~voiced).any():
        return f0
    idx = torch.arange(n, device=f0.device)
    filled = torch.from_numpy(
        np.interp(idx[~voiced].cpu().numpy(), idx[voiced].cpu().numpy(),
                  f0[voiced].cpu().numpy())).to(f0.device, f0.dtype)
    return f0.clone().index_copy_(0, idx[~voiced], filled)


def _median_axis_last(t: torch.Tensor, radius: int) -> torch.Tensor:
    """Median along the T axis with a (2*radius+1) window, edge-clamped.

    Equivalent to upstream's scipy medfilt over the 1-D coarse/pitchf frame
    tracks; padding repeats the boundary value so output length is T.
    """
    k = int(2 * radius + 1)
    was_1d = t.dim() == 1
    if was_1d:
        t = t.unsqueeze(0)
    if radius <= 0 or t.shape[-1] == 0:
        return t.squeeze(0) if was_1d else t
    padded = F.pad(t, (radius, radius), mode="replicate")
    out = padded.unfold(-1, k, 1).median(dim=-1).values
    return out.squeeze(0) if was_1d else out


def coarse_f0(f0_hz, f0_up_key=0, filter_radius=3):
    """Coarse pitch bins (1..255 int32) + float key-shifted f0 (both (T,)).

    Port of upstream get_f0's tail: gap interpolation, pow(2, key/12) shift,
    1127*log(1+f/700) mel mapping with 50..1100 Hz -> 1..255, then the
    medfilt(filter_radius * 2 + 1) ComfyUI-RVC applies before clipping; the
    median smooths the float track too (pitchf feeds harmonic UVR + protect).
    """
    f0 = f0_hz.clone() if isinstance(f0_hz, torch.Tensor) else torch.tensor(f0_hz)
    f0 = f0.float().reshape(-1).cpu()
    f0 = _interp_unvoiced(f0)
    f0 = f0 * pow(2, f0_up_key / 12)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel_min = 1127 * float(np.log(1 + 50 / 700))
    f0_mel_max = 1127 * float(np.log(1 + 1100 / 700))
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitchf = f0.clone()
    f0_mel = _median_axis_last(f0_mel, filter_radius)
    pitchf = _median_axis_last(pitchf, filter_radius)
    return f0_mel.round().int(), pitchf


def change_rms(data1, sr1, data2, sr2, rate):
    """Mix the source loudness envelope into the converted audio (pure torch)."""
    def rms(y, sr):
        y = y.double()
        window, hop = sr // 2 * 2, sr // 2
        padded = F.pad(y.unsqueeze(0), (window // 2, window // 2)).squeeze(0)
        win = padded.unfold(0, window, hop)
        return win.pow(2).mean(-1).sqrt()

    rms1 = rms(data1, sr1)
    rms2 = rms(data2, sr2)
    rms1 = F.interpolate(rms1.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze(0).squeeze(0)
    rms2 = F.interpolate(rms2.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze(0).squeeze(0)
    rms2 = torch.clamp(rms2, min=1e-6)
    return (data2 * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1))).float()


def _f0_of(pitch_model, wav16_batch):
    """f0 Hz per 160-sample frame; unified adapter over rmvpe/fcpe.

    Both branches run under an explicitly disabled autocast: f0 models are
    fp32-only, and a caller-side `torch.autocast("cuda", fp16)` (as the
    convert() wrapper of Task 6 adds) must not silently downcast their matmul
    or conv layers.
    """
    module = pitch_model.module
    if pitch_model.pitch_kind == "rmvpe":
        from rvc_train.rmvpe import infer_f0
        with torch.autocast(device_type=wav16_batch.device.type, enabled=False):
            f0 = infer_f0(module, wav16_batch, thred=0.03)
        return torch.as_tensor(f0).float().reshape(1, -1)
    # fcpe: module(wav[1, N] on the model's device, threshold=...) -> (1, T) Hz
    param = next(module.parameters(), None)
    if param is not None and param.device != wav16_batch.device:
        wav16_batch = wav16_batch.to(param.device)
    with torch.autocast(device_type=wav16_batch.device.type, enabled=False), torch.inference_mode():
        f0 = module(wav16_batch, threshold=0.006)
    return f0.float().reshape(1, -1)


class RVCPipeline:
    def __init__(self, tgt_sr: int, x_pad=1, x_query=6, x_center=38, x_max=41):
        self.tgt_sr = tgt_sr
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max

    def split_points(self, audio):
        """Quiet spots every t_center region, for crossfaded segment seams."""
        audio_pad = F.pad(audio.unsqueeze(0), (self.window // 2, self.window // 2), mode="reflect").squeeze(0)
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = torch.zeros_like(audio)
            # Python slice semantics: stop=i-window means "to len-window", so
            # every fold is full-length (upstream np.abs(audio_pad[i:i-self.window])) —
            # NOT audio_pad[i:i+window], which collides with the full-length lhs.
            for i in range(self.window):
                audio_sum += audio_pad[i:i - self.window].abs()
            for t in range(self.t_center, audio.shape[0], self.t_center):
                seg = audio_sum[t - self.t_query:t + self.t_query]
                opt_ts.append(t - self.t_query + int(seg.argmin().item()))
        return opt_ts

    def vc(self, hubert, net_g, sid, audio0, pitch, pitchf, index_vectors, index_rate, version, protect, device):
        from rvc_train.hubert import extract_features

        feats = extract_features(hubert, audio0.view(1, -1).to(device), version).to(device)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index_vectors is not None and index_rate != 0:
            blended = blend_with_index(feats[0].float().cpu(), index_vectors, index_rate)
            feats = blended.unsqueeze(0).to(device=device, dtype=feats.dtype)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len_t = torch.tensor([p_len], device=device).long()
        with torch.inference_mode():
            if pitch is not None and pitchf is not None:
                audio1 = net_g.infer(feats, p_len_t, pitch, pitchf, sid)[0]
            else:
                audio1 = net_g.infer(feats, p_len_t, sid)[0]
        return audio1[0, 0].float().cpu()

    def convert(
        self,
        hubert,
        net_g,
        sid_t,
        pitch_model,
        index,
        index_rate: float,
        version: str,
        audio,
        f0_up_key: int,
        if_f0: int,
        protect: float,
        filter_radius: int,
        rms_mix_rate: float,
        resample_sr: int,
        resample_fn,       # callable(waveform_1d, sr_in, sr_out) -> waveform_1d
        device,
    ):
        """Returns (waveform_1d float32 CPU, out_sr int)."""
        if if_f0 == 1 and pitch_model is None:
            raise RuntimeError("ERROR: if_f0 is 1 but no pitch model (rmvpe/fcpe) was provided")
        if if_f0 == 1 and net_g.voice_if_f0 != 1:
            raise RuntimeError(
                "ERROR: if_f0 is 1 but the voice checkpoint is not f0-capable: %s (voice f0=%s)"
                % (net_g.voice_path, net_g.voice_if_f0))
        if if_f0 == 0 and net_g.voice_if_f0 == 1:
            raise RuntimeError(
                "ERROR: if_f0 is 0 but the voice checkpoint is f0-capable: %s (voice f0=%s)"
                % (net_g.voice_path, net_g.voice_if_f0))
        if version != net_g.voice_version:
            raise RuntimeError(
                "ERROR: version %s does not match voice checkpoint version %s (%s)"
                % (version, net_g.voice_version, net_g.voice_path))
        sid_t = sid_t.to(device).long().reshape(-1)
        if sid_t.numel() and not (0 <= int(sid_t.max()) < net_g.voice_n_spk):
            raise RuntimeError(
                "ERROR: speaker id %d is out of range for %d speakers in %s"
                % (int(sid_t.max()), net_g.voice_n_spk, net_g.voice_path))
        audio = highpass_48hz(audio)
        opt_ts = self.split_points(audio)
        audio_pad = F.pad(audio.unsqueeze(0), (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        p_len = audio_pad.shape[0] // self.window
        pitch, pitchf = None, None
        if if_f0 == 1:
            f0_hz = _f0_of(pitch_model, audio_pad[None].to(device))
            # rmvpe/fcpe land on N//160 (+1) frames; the synthesizer's nsff0
            # slice (skip_head, return_length) needs an exact per-frame track
            # over the padded length — take p_len = N//160 as reference does,
            # so truncate the (+1)-frame track to p_len.
            pitch, pitchf = coarse_f0(f0_hz, f0_up_key, filter_radius)
            pitch = pitch.reshape(1, -1)[:, :p_len].to(device).long()
            pitchf = pitchf.reshape(1, -1)[:, :p_len].float().to(device)
        audio_opt = []
        s = 0
        t = None
        for t in opt_ts:
            t = t // self.window * self.window
            e = t + self.t_pad * 2 + self.window
            sl = slice(s // self.window, e // self.window)
            seg = self.vc(hubert, net_g, sid_t, audio_pad[s:e],
                          pitch[:, sl] if pitch is not None else None,
                          pitchf[:, sl] if pitchf is not None else None,
                          index, index_rate, version, protect, device)
            audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        sl = slice((t // self.window if t is not None else 0), None)
        seg = self.vc(hubert, net_g, sid_t, audio_pad[t:] if t is not None else audio_pad,
                      pitch[:, sl] if pitch is not None else None,
                      pitchf[:, sl] if pitchf is not None else None,
                      index, index_rate, version, protect, device)
        audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
        out = torch.cat(audio_opt)
        if rms_mix_rate != 1:
            out = change_rms(audio, 16000, out, self.tgt_sr, rms_mix_rate)
        if self.tgt_sr != resample_sr >= 16000:
            out = resample_fn(out, self.tgt_sr, resample_sr)
            sr = resample_sr
        else:
            sr = self.tgt_sr
        peak = out.abs().max() / 0.99
        if peak > 1:
            out = out / peak
        return out.float().cpu().squeeze(0), int(sr)
