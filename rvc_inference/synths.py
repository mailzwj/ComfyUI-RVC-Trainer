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
            n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)
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


class SynthesizerTrnMs256NSFsid(nn.Module):
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
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            length = int(return_length.item()) if isinstance(return_length, torch.Tensor) else int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            nsff0 = nsff0[:, head : head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("rvc_inference.synths carries inference only; use rvc_train.models for training")


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=None, dtype=None):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, device=device, dtype=dtype)
        del self.enc_p
        self.enc_p = M.TextEncoder(768, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), device=device, dtype=dtype, operations=torch.nn)
        self.voice_version = "v2"


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
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
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            length = int(return_length.item()) if isinstance(return_length, torch.Tensor) else int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)

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
