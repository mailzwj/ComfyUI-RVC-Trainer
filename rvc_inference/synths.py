"""Inference synthesizers distilled from RVC-Project's SynthesizerTrnMs* modules.

Building blocks come from rvc_train.models (plain torch.nn operations so nothing
depends on comfy.ops at import time). Only the inference pieces missing from the
trainer are defined here: the plain Generator (non-NSF), and the four
SynthesizerTrn variants with their `infer` method (posterior-free sampling).
State-dict keys mirror upstream RVC voice checkpoints exactly.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

import rvc_train.models as M


def _to_int(x):
    """Accept int or 0/1-D torch tensor the way upstream RVC does."""
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


class _InferMixin:
    """Shared `infer` core for the four SynthesizerTrn variants.

    All variants differ only in which TextEncoder (enc_p) and vocoder (dec)
    they construct; sampling, the sliced decode path, and the speaker-embedding
    lookup are identical upstream, so they live here once. The mixin performs a
    single `emb_g(sid)` lookup and returns that g so subclasses reuse it in
    their dec calls rather than re-encoding the speaker.

    Note: the sliced decode path is all-or-nothing upstream — skip_head and
    return_length must be set together; if either is None the full-frames path
    runs and the slice window is ignored entirely.
    """

    def _infer_common(self, enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length):
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            head = _to_int(skip_head)
            length = _to_int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            return z, x_mask, g, (z, z_p, m_p, logs_p), head, length
        # Full-frames path. slice_head/slice_length stay None so callers skip
        # their own slicing (e.g. the nsff0 cut in the NSF variant below).
        m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        return z, x_mask, g, (z, z_p, m_p, logs_p), None, None


class Generator(nn.Module):
    """Non-NSF vocoder used by SynthesizerTrn*_nono (f0=0) voices."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, device=None, dtype=None):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3, device=device, dtype=dtype)
        resblock_cls = M.ResBlock1 if resblock == "1" else M.ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2, device=device, dtype=dtype)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d, device=device, dtype=dtype))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False, device=device, dtype=dtype)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1, device=device, dtype=dtype)

    def forward(self, x, g=None, n_res=None):
        if n_res is not None:
            n = _to_int(n_res)
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, M.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SynthesizerTrnMs256NSFsid(_InferMixin, nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=None, dtype=None):
        super().__init__()
        if isinstance(sr, str):
            sr = M.sr2sr[sr]
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = M.TextEncoder(256, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), device=device, dtype=dtype, operations=torch.nn)
        self.dec = M.GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, device=device, dtype=dtype, operations=torch.nn)
        self.flow = M.ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels, device=device, dtype=dtype, operations=torch.nn)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels, device=device, dtype=dtype)
        self.voice_version = "v1"
        self.voice_tgt_sr = sr

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=None, return_length=None, return_length2=None):
        z, x_mask, g, diagnostics, head, slice_length = self._infer_common(self.enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length)
        if head is not None:
            # Assumes the caller's pitchf window covers the padded length,
            # mirroring upstream RVC's inference which slices it the same way.
            # Do NOT clamp here — upstream parity is intentional (see plan note).
            nsff0 = nsff0[:, head : head + slice_length]
        o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        return o, x_mask, diagnostics

    def forward(self, *args, **kwargs):
        raise NotImplementedError("rvc_inference.synths carries inference only; use rvc_train.models for training")


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=None, dtype=None):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=device, dtype=dtype)
        del self.enc_p
        self.enc_p = M.TextEncoder(768, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), device=device, dtype=dtype, operations=torch.nn)
        self.voice_version = "v2"


class SynthesizerTrnMs256NSFsid_nono(_InferMixin, nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr=None, device=None, dtype=None):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = M.TextEncoder(256, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), f0=False, device=device, dtype=dtype, operations=torch.nn)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, device=device, dtype=dtype)
        self.flow = M.ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels, device=device, dtype=dtype, operations=torch.nn)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels, device=device, dtype=dtype)
        self.voice_version = "v1"
        self.voice_tgt_sr = sr

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, phone, phone_lengths, sid, skip_head=None, return_length=None, return_length2=None):
        z, x_mask, g, diagnostics, _, _ = self._infer_common(self.enc_p, phone, None, phone_lengths, sid, skip_head, return_length)
        o = self.dec(z * x_mask, g=g, n_res=return_length2)
        return o, x_mask, diagnostics

    def forward(self, *args, **kwargs):
        raise NotImplementedError("rvc_inference.synths carries inference only; use rvc_train.models for training")


class SynthesizerTrnMs768NSFsid_nono(SynthesizerTrnMs256NSFsid_nono):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr=None, device=None, dtype=None):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=device, dtype=dtype)
        del self.enc_p
        self.enc_p = M.TextEncoder(768, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), f0=False, device=device, dtype=dtype, operations=torch.nn)
        self.voice_version = "v2"


SYNTH_CLASSES = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
}
