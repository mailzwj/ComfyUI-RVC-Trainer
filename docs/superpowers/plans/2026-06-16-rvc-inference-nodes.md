# RVC Voice Conversion Inference Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four self-contained RVC voice-conversion nodes (Voice Loader / Hubert&Pitch Loader / Execute / Audio Synthesis) to ComfyUI-RVC-Trainer, distilled from the RVC-Project inference pipeline.

**Architecture:** Reuse the trainer's already-plain-torch model building blocks (`rvc_train/models.py`, `rvc_train/hubert.py`, `rvc_train/rmvpe.py`), vendor the missing inference-only pieces (synthesizer `infer`, plain-torch `Generator`, FCPE, conversion pipeline, faiss index retrieval), then expose four nodes via the `comfy_api.latest io` API. Models are discovered recursively under `models/rvc` via `folder_paths` (registered already in `__init__.py`); upstream `.pth` checkpoints load on GPU at fp32 and run under fp16 autocast on Ampere+ GPUs. FAISS `.index` files load via `faiss-cpu` (already installed in the validation env); `.npy` works standalone.

**Tech Stack:** Python 3.12, torch 2.9.1+cu130, ComfyUI `comfy_api.latest` io schema, faiss-cpu 1.15.1 (present), numpy/scipy (present).

**Environment for ALL run/test commands:**

```
$PY  = D:\UserApps\Conda\envs\cu130\python.exe   (verified: python 3.12.12, torch 2.9.1+cu130, RTX 5090, faiss 1.15.1)
$DPT = D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC-Trainer
$CUI = D:\Workspace\ComfyUI
```

Every Run command below assumes pwsh with:

```powershell
$env:PYTHONPATH = "D:\Workspace\ComfyUI;D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC-Trainer"
& "D:\UserApps\Conda\envs\cu130\python.exe" <script>
```

Safety constraints (user-confirmed):
- **No package installs.** Everything needed is already in `cu130` (faiss included). If a test fails with a missing module, report it — do not pip install.
- Do not modify `rvc_train/` training behavior except the one additive patch in Task 2 (a default-valued flag).
- Working directory for all commands: `$DPT`.

---

### Task 1: `rvc_inference` package skeleton

**Files:**
- Create: `rvc_inference/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_import.py`:

```python
import sys

def main():
    from rvc_inference import AVAILABLE_MODULES
    assert AVAILABLE_MODULES == [], AVAILABLE_MODULES
    print("PASS: package imports")

if __name__ == "__main__":
    main()
```

Create a placeholder `tests_rvc/__init__.py` (empty file).

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_import.py
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'rvc_inference'`.

- [ ] **Step 3: Write minimal implementation**

Create `rvc_inference/__init__.py`:

```python
"""Self-contained RVC inference nodes: loaders, pipeline, synthesis."""

AVAILABLE_MODULES = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: package imports`.

- [ ] **Step 5: Commit**

```bash
git add rvc_inference/__init__.py tests_rvc/
git commit -m "feat(rvc-inference): package skeleton"
```

---

### Task 2: Synthesizer inference classes (`rvc_inference/synths.py`)

The trainer's `rvc_train/models.py` has plain-torch building blocks but only training `forward` — no `infer`, no `Generator` (plain, non-NSF), no `_nono` (f0=0) variants. This task vendors them without copying code (import the shared modules, define only what is missing).

Patch to `rvc_train/models.py` (additive, default-valued): `TextEncoder.__init__` gains `f0=True`; `emb_pitch` is only created when `f0=True`. Its `forward` already handles `pitch=None`.

**Files:**
- Modify: `rvc_train/models.py:240` (TextEncoder signature/body, 3 lines)
- Create: `rvc_inference/synths.py`
- Test: `tests_rvc/test_synths.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_synths.py`:

```python
import torch

VOICE = r"D:\Workspace\ComfyUI\models\rvc\my-vocals-v2-48k.pth"

def _build(cls_name, cpt):
    from rvc_inference.synths import SYNTH_CLASSES
    cls = SYNTH_CLASSES[(cpt["version"], int(cpt.get("f0", 1)))]
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    model = cls(*config)
    return model

def main():
    import comfy.utils
    cpt = torch.load(VOICE, map_location="cpu", weights_only=True)
    assert len(cpt["config"]) == 18, len(cpt["config"])
    model = _build("v2", cpt)

    sd_keys = set(cpt["weight"].keys())
    mine = set(model.state_dict().keys())
    stripped = {k for k in sd_keys if "enc_q" not in k}
    assert stripped == mine, ["missing", sorted(stripped - mine)[:10], "extra", sorted(mine - stripped)[:10]]

    with torch.no_grad():
        model.load_state_dict({k: v for k, v in cpt["weight"].items() if "enc_q" not in k}, strict=True)
    assert model.voice_version == cpt["version"]
    assert model.voice_tgt_sr == cpt["config"][-1]
    n_spk = cpt["weight"]["emb_g.weight"].shape[0]

    T, D = 100, 768 if cpt["version"] == "v2" else 256
    phone = torch.randn(1, D, T) * 0.01
    length = torch.tensor([T])
    pitch = torch.randint(1, 230, (1, T))
    pitchf = torch.rand(1, T) * 300 + 100
    torch.manual_seed(0)
    out = model.infer(phone, length, pitch, pitchf, torch.tensor([[0]]))[0]
    assert out.shape == (1, 1, 256 * 5), out.shape
    assert torch.isfinite(out).all()

    print("PASS: synths match checkpoint keys and infer runs")

if __name__ == "__main__":
    main()
```

Notes the executor needs:
- The test asserts `model.infer(...)` output shape when an inferred f0=1 module is used — 256×segment_size samples for a segment-size slice (per RVC architecture).
- Config slicing requires `len(config)==18` — verified against the real checkpoint.
- Run in the `cu130` env where `comfy.utils` is importable via PYTHONPATH above.

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_synths.py
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'rvc_inference.synths'` (or AttributeError on infer).

- [ ] **Step 3: Patch TextEncoder + write synths.py**

Patch `rvc_train/models.py` `TextEncoder.__init__` (line 241):

Before:
```python
class TextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, device=None, dtype=None, operations=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = operations.Linear(in_channels, hidden_channels, device=device, dtype=dtype)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = operations.Embedding(256, hidden_channels, device=device, dtype=dtype)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), device=device, dtype=dtype, operations=operations)
        self.proj = operations.Conv1d(hidden_channels, out_channels * 2, 1, device=device, dtype=dtype)
```

After:
```python
class TextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, f0=True, device=None, dtype=None, operations=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = operations.Linear(in_channels, hidden_channels, device=device, dtype=dtype)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = operations.Embedding(256, hidden_channels, device=device, dtype=dtype)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), device=device, dtype=dtype, operations=operations)
        self.proj = operations.Conv1d(hidden_channels, out_channels * 2, 1, device=device, dtype=dtype)
```

Create `rvc_inference/synths.py`:

```python
"""Inference synthesizers distilled from RVC-Project's SynthesizerTrnMs* modules.

Building blocks come from rvc_train.models (plain torch.nn operations so nothing
depends on comfy.ops). Only the inference pieces missing from the trainer are
defined here: the plain Generator, and the four SynthesizerTrn variants with
their `infer` method (posterior-free sampling path).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from rvc_train.models import (
    GeneratorNSF,
    ResBlock1,
    ResBlock2,
    ResidualCouplingBlock,
    TextEncoder,
    get_padding,
    sequence_mask,
    sr2sr,
)

LRELU_SLOPE = 0.1


class Generator(nn.Module):
    """Non-NSF vocoder used by SynthesizerTrn*_nono (f0=0 voices)."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None, n_res=None):
        if n_res is not None:
            n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class _InferMixin:
    def _infer_common(self, enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length, seed_zp):
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            length = int(return_length.item()) if isinstance(return_length, torch.Tensor) else int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head: dec_head + length]
            x_mask = x_mask[:, :, dec_head: dec_head + length]
        else:
            m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        return z, x_mask


class SynthesizerTrnMs256NSFsid(nn.Module, _InferMixin):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr):
        super().__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(256, inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, float(p_dropout))
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                                upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                                gin_channels=gin_channels, sr=sr)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        self.voice_version = "v1"
        self.voice_tgt_sr = sr

    _ = (get_padding, sequence_mask)  # keep linters from dropping the shared imports

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=None, return_length=None, return_length2=None):
        z, x_mask = self._infer_common(self.enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length, True)
        o = self.dec(z * x_mask, nsff0, g=self.emb_g(sid).unsqueeze(-1), n_res=return_length2)
        return o, x_mask, (z, None)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("rvc_inference.synths carries inference only; use rvc_train.models for training")


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                         n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                         resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                         spk_embed_dim, gin_channels, sr)
        del self.enc_p
        self.enc_p = TextEncoder(768, inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, float(p_dropout))
        self.voice_version = "v2"


class SynthesizerTrnMs256NSFsid_nono(nn.Module, _InferMixin):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr=None):
        super().__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(256, inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, float(p_dropout), f0=False)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                             upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                             gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        self.voice_version = "v1"
        self.voice_tgt_sr = sr

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, phone, phone_lengths, sid, skip_head=None, return_length=None, return_length2=None):
        z, x_mask = self._infer_common(self.enc_p, phone, None, phone_lengths, sid, skip_head, return_length, np := None)
        o = self.dec(z * x_mask, g=self.emb_g(sid).unsqueeze(-1), n_res=return_length2)
        return o, x_mask, (z, None)


class SynthesizerTrnMs768NSFsid_nono(SynthesizerTrnMs256NSFsid_nono):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr=None):
        super().__init__(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                         n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                         resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                         spk_embed_dim, gin_channels, sr)
        del self.enc_p
        self.enc_p = TextEncoder(768, inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, float(p_dropout), f0=False)
        self.voice_version = "v2"


SYNTH_CLASSES = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
}
```

Wait — the `_ = (get_padding, sequence_mask)` line above is a placeholder trick; **delete it and remove those two imports** since they are unused (`-generator` note). Also `seed_zp` in `_InferMixin._infer_common` is unused; drop the parameter to keep it lean:

Final `_InferMixin`:

```python
class _InferMixin:
    def _infer_common(self, enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length):
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            head = int(skip_head.item()) if isinstance(skip_head, torch.Tensor) else int(skip_head)
            length = int(return_length.item()) if isinstance(return_length, torch.Tensor) else int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head: dec_head + length]
            x_mask = x_mask[:, :, dec_head: dec_head + length]
        else:
            m_p, logs_p, x_mask = enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        return z, x_mask
```

and in callers drop the trailing `seed_zp`/`np := None` arguments accordingly:

```python
    def infer(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=None, return_length=None, return_length2=None):
        z, x_mask = self._infer_common(self.enc_p, phone, pitch, phone_lengths, sid, skip_head, return_length)
        o = self.dec(z * x_mask, nsff0, g=self.emb_g(sid).unsqueeze(-1), n_res=return_length2)
        return o, x_mask, (z, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: synths match checkpoint keys and infer runs`.

If a key mismatch shows up, it means `rvc_train.models` naming differs from upstream `.pth` for that submodule — fix by aligning the class in `rvc_train/models.py` (naming only), re-run both this test and the existing trainer smoke.

- [ ] **Step 5: Update `rvc_inference/__init__.py` AVAILABLE_MODULES**

```python
AVAILABLE_MODULES = ["rvc_inference.synths"]
```

- [ ] **Step 6: Commit**

```bash
git add rvc_train/models.py rvc_inference/synths.py rvc_inference/__init__.py tests_rvc/test_synths.py
git commit -m "feat(rvc-inference): inference synthesizers distilled from RVC-Project"
```

---

### Task 3: Index loader with faiss `.index` + `.npy` support

**Files:**
- Create: `rvc_inference/index_loader.py`
- Test: `tests_rvc/test_index_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_index_loader.py`:

```python
import numpy as np
import torch

def main():
    import faiss
    from rvc_inference.index_loader import (
        IndexVectors, load_index, blend_with_index,
    )

    rng = np.random.default_rng(42)
    feats_np = rng.standard_normal((50, 8)).astype("float32")
    idx_np = rng.standard_normal((40, 8)).astype("float32")

    # .npy path
    np.save("temp_index.npy", idx_np)
    idx = load_index("temp_index.npy")
    assert isinstance(idx, IndexVectors) and idx.kind == "feats"

    # faiss .index path
    fa = faiss.index_factory(8, "IVF2,Flat")
    fa.train(idx_np)
    fa.add(idx_np)
    faiss.write_index(fa, "temp_index.index")
    idx_f = load_index("temp_index.index")
    assert isinstance(idx_f, IndexVectors) and idx_f.kind == "faiss"

    # blend shape and finite
    feats = torch.from_numpy(feats_np)
    out = blend_with_index(feats, idx, 0.75)
    assert out.shape == feats.shape and torch.isfinite(out).all()
    out_f = blend_with_index(feats, idx_f, 0.75)
    assert out_f.shape == feats.shape and torch.isfinite(out_f).all()

    # both paths converge in top-1, zero-noise case
    idx_q = IndexVectors("feats", torch.from_numpy(idx_np[:1]))
    out_q = blend_with_index(torch.from_numpy(idx_np[:3]), idx_q, 0.75)
    expected = idx_np[:3] * 0.75 + idx_np[:3] * 0.25
    assert torch.allclose(out_q, torch.from_numpy(expected), atol=1e-4)

    print("PASS: index_loader loads .npy and .index and blends")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_index_loader.py
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'rvc_inference.index_loader'`.

- [ ] **Step 3: Implement `index_loader.py`**

```python
"""Feature-index retrieval for RVC inference.

Supports the two file forms RVC-Project exports: raw ``.npy`` feature
matrices and FAISS ``.index`` files.  For the FAISS form the top-k search
returns squared L2 distances, so the same 1/dist^4 weighting used in the
matrix path applies (see the comment in rvc's blend for the derivation).
"""

import numpy as np
import torch

import rvc_inference.index_loader as _self  # noqa: F401  (stable import surface)


class IndexVectors:
    """Container: either 'feats' (a [N, D] torch matrix on CPU) or a FAISS index."""

    def __init__(self, kind, data):
        self.kind = kind
        self.data = data


def load_index(path):
    """Load ``.npy`` feature matrix or FAISS ``.index`."""
    if path.lower().endswith(".npy"):
        vectors = torch.from_numpy(np.load(path)).float()
        if vectors.dim() != 2:
            raise RuntimeError("ERROR: %s is not an RVC feature matrix (expected 2-D)" % path)
        return IndexVectors("feats", vectors.contiguous())

    if path.lower().endswith(".index"):
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "ERROR: %s is a FAISS index but faiss-cpu is not installed. "
                "Run without Index, or `pip install faiss-cpu`, or export the "
                "vectors once as .npy." % path)
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"IwFt":
            raise RuntimeError("ERROR: %s is not a FAISS index file" % path)
        index = faiss.read_index(path)
        try:
            faiss.extract_index_ivf(index).nprobe = 1
        except Exception:
            pass
        return IndexVectors("faiss", index)

    raise RuntimeError("ERROR: unsupported index file: %s (use .npy or .index)" % path)


def _retrieve_matrix(feats, vectors, k=8):
    dists = torch.cdist(feats.float(), vectors.float()).pow(2)
    k = min(k, vectors.shape[0])
    score, ix = dists.topk(k, largest=False)
    weight = torch.square(1 / score.clamp_min(1e-8))
    weight = weight / weight.sum(dim=1, keepdim=True)
    return (vectors[ix] * weight.unsqueeze(-1)).sum(dim=1), weight


def _retrieve_faiss(feats, index, k=8):
    import faiss
    ntotal = max(int(index.ntotal), 1)
    k = min(k, ntotal)
    dist, labels = index.search(feats.float().numpy(), k)
    flat = labels.reshape(-1)
    rec = index.reconstruct_batch(flat).astype("float32")
    retrieved = rec.reshape(feats.shape[0], k, -1)
    weight = torch.square(1 / torch.from_numpy(dist).clamp_min(1e-8))
    weight = weight / weight.sum(dim=1, keepdim=True)
    vec = torch.from_numpy(retrieved) * weight.unsqueeze(-1)
    return vec.sum(dim=1), weight


def blend_with_index(feats, index, index_rate, k=8):
    """feat retrieval blend on CPU float tensors; feats is [T, D]."""
    feats = feats.float()
    if index is None or index_rate <= 0:
        return feats
    if index.kind == "feats":
        retrieved, _ = _retrieve_matrix(feats, index.data, k)
    else:
        retrieved, _ = _retrieve_faiss(feats, index.data, k)
    return retrieved.to(feats.dtype) * index_rate + feats * (1 - index_rate)
```

Correction before writing: the FAISS magic is `IwFt`? Verify — the existing `rvc/index.py` uses `_FAISS_MAGIC = b"IwFl"` (uppercase I, lowercase w, uppercase F, lowercase l — faiss magic is `b"IwFt"` in newer faiss? whichever). **Use the same constant the existing reference uses (`IwFl`) as the starting point but call it out:** actually correct check behavior is to try `magic in (b"IwFt", b"IwFl")` since faiss 1.15 writes `IwFt`? To be safe:

Sharpen the check that faiss 1.15.1 writes `IwFt`? Rather than assume, make the loader lenient:

```python
        if magic not in (b"IwFt", b"IwFl"):
            raise RuntimeError("ERROR: %s is not a FAISS index file" % path)
```

And the final double-check during test is that `load_index()` on a written file raises nothing.

Also delete the pointless `_self` import line at the top (left in the paste above by accident):

```python
import numpy as np
import torch


class IndexVectors:
    ...
```

Final `load_index` `_npy` branch also corrects the `magic` comment — but the file keeps magic check only for `.index`.

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: index_loader loads .npy and .index and blends`. If the faiss magic mismatch bites, fix the constant (print magic during debugging) — do not loosen beyond the two known magics.

- [ ] **Step 5: Commit**

```bash
git add rvc_inference/index_loader.py tests_rvc/test_index_loader.py
git commit -m "feat(rvc-inference): faiss and npy index loaders"
```

---

### Task 4: Voice / hubert / pitch loaders

**Files:**
- Create: `rvc_inference/loader.py`
- Create: `rvc_inference/fcpe.py`  (vendor of ComfyUI-RVC `rvc/fcpe.py`, 1-to-1; `operations=torch.nn` works because every use site passes device=/dtype= kwargs)
- Test: `tests_rvc/test_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_loader.py`:

```python
import torch

CUI = r"D:\Workspace\ComfyUI"

VOICE_V2 = CUI + r"\models\rvc\my-vocals-v2-48k.pth"
VOICE_OTHER = CUI + r"\models\rvc\zi-lai-ye\zilaiye.pth"
VOICE_BAD = CUI + r"\models\rvc\pretrained_v2\f0G48k.pth"
HUBERT = CUI + r"\models\rvc\hubert_base.safetensors"
RMVPE = CUI + r"\models\rvc\rmvpe.safetensors"
FCPE = CUI + r"\models\rvc\fcpe.safetensors"

def main():
    import comfy.model_management as mm
    from rvc_inference.loader import load_voice, load_pitch, PitchModel

    device = mm.get_torch_device()

    v = load_voice(VOICE_V2, device)
    assert v.voice_version == "v2" and int(v.voice_tgt_sr) == 48000, (v.voice_version, v.voice_tgt_sr)
    assert v.voice_n_spk == int(cpt_n_spk(VOICE_V2))

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

    print("PASS: loaders")

def cpt_n_spk(path):  # helper shared with test only
    import torch as t
    cpt = t.load(path, map_location="cpu", weights_only=True)
    return cpt["weight"]["emb_g.weight"].shape[0]

if __name__ == "__main__":
    main()
```

Note: the test also imports `load_hubert` — add it to the same import line.

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_loader.py
```
Expected: **FAIL** (`No module named 'rvc_inference.loader'`).

- [ ] **Step 3: Vendor `fcpe.py`, implement `loader.py`**

Vendor step — copy **270 →** 175 lines from `D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC\rvc\fcpe.py` into `rvc_inference/fcpe.py` **minus** line 5 (`import comfy.ops`), keeping everything else verbatim (all layers take `operations=`; callsites pass `torch.nn` because `nn.Conv1d/LayerNorm/GroupNorm/Linear` all accept `device=`/`dtype=` kwargs).

Create `rvc_inference/loader.py`:

```python
"""Loaders for RVC voice .pth checkpoints and the content/pitch encoders."""

import os

import torch
import torch.nn.functional as F

import comfy.model_management


class _LoadedVoice:
    pass


def load_voice(path, device):
    """Validate an upstream RVC voice checkpoint and build its synthesizer.

    True training checkpoints (G_*.pth with only 'model') are rejected with a
    message pointing the user at export-for-inference tooling.
    """
    from rvc_inference.synths import SYNTH_CLASSES

    if not os.path.isfile(path):
        raise FileNotFoundError("RVC voice model not found: %s" % path)
    cpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt or len(cpt["config"]) != 18:
        raise RuntimeError("ERROR: not an RVC voice checkpoint: %s" % path)
    version = cpt.get("version", "v1")
    if_f0 = int(cpt.get("f0", 1))
    cls = SYNTH_CLASSES.get((version, if_f0))
    if cls is None:
        raise RuntimeError("ERROR: unknown RVC voice version/f0: %s/%s" % (version, if_f0))
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    weight = {k: v for k, v in cpt["weight"].items() if "enc_q" not in k}
    model = cls(*config)
    missing, unexpected = model.load_state_dict(weight, strict=False)
    if unexpected or missing:
        raise RuntimeError(
            "ERROR: voice checkpoint keys do not match the synthesizer layout: %s "
            "(missing=%s unexpected=%s)" % (path, missing[:5], unexpected[:5]))
    model = model.to(device).eval()
    model.voice_version = version
    model.voice_if_f0 = if_f0
    model.voice_tgt_sr = config[-1]
    model.voice_n_spk = config[-3]
    model.voice_path = path
    return model


def load_hubert(path, device):
    from rvc_train.hubert import load_hubert as _load
    return _load(path, device)


class PitchModel:
    """Bundle of the RMVPE model + mel basis, or the FCPE model, tagged with its kind."""

    def __init__(self, kind, module):
        self.pitch_kind = kind
        self.module = module
        self.dtype = next(module.parameters()).dtype


def load_pitch(path, device):
    """Load rmvpe.safetensors or fcpe.safetensors identified by filename."""
    name = os.path.basename(path).lower()
    if "fcpe" in name:
        from rvc_inference.fcpe import FCPitch
        model = FCPitch(device=device, dtype=torch.float32, operations=torch.nn)
        sd, _metadata = _safetensors_metadata(path)
        from torch.utils import checkpoint as _unused  # noqa: F401 (importing torch.utils is fine)
        model.load_state_dict(sd, strict=True)
        return PitchModel("fcpe", model.to(device).eval())
    if "rmvpe" in name:
        from rvc_train.rmvpe import load_rmvpe as _load
        return PitchModel("rmvpe", _load(path, device))
    raise RuntimeError(
        "ERROR: %s is not an RVC pitch checkpoint (expected 'rmvpe' or 'fcpe' in the filename)" % path)


def _safetensors_metadata(path):
    from safetensors.torch import load_file
    return load_file(path), None
```

Wait, two corrections before writing the file:
- `load_state_dict` on FCPitch overrides `load_state_dict` in the vendored class to convert `output_proj.weight_g/weight_v` — passing the raw safetensors dict is fine.
- The FCPitch state dict keys from safetensors — reference pack used `comfy.utils.load_torch_file` which handles both magic types. Use `comfy.utils.load_torch_file(path, safe_load=True)` instead of direct `safetensors.torch.load_file` for consistency, and drop `_safetensors_metadata`:

Sharpened `load_pitch` fcpe branch:

```python
    if "fcpe" in name:
        from rvc_inference.fcpe import FCPitch
        import comfy.utils
        sd, _ = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        model = FCPitch(device=device, dtype=torch.float32, operations=torch.nn)
        model.load_state_dict(sd, strict=True)
        return PitchModel("fcpe", model.to(device).eval())
```

(Note: `load_torch_file(..., safe_load=True)` errors if the file isn't safetensors-backed — matching the reference pack's requirement that these base assets are the converted safetensors.)

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: loaders`.

- [ ] **Step 5: Commit**

```bash
git add rvc_inference/loader.py rvc_inference/fcpe.py tests_rvc/test_loader.py
git commit -m "feat(rvc-inference): voice/hubert/pitch loaders"
```

---

### Task 5: Conversion pipeline (vendor + faiss integration + median filter)

**Files:**
- Create: `rvc_inference/pipeline.py`
- Test: `tests_rvc/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_pipeline.py`:

```python
import numpy as np
import torch

CUI = r"D:\Workspace\ComfyUI"
HUBERT = CUI + r"\models\rvc\hubert_base.safetensors"
RMVPE = CUI + r"\models\rvc\rmvpe.safetensors"
VOICE = CUI + r"\models\rvc\my-vocals-v2-48k.pth"
NPY = CUI + r"\models\rvc\my-vocals-v2-48k.npy"

def main():
    import comfy.model_management as mm
    from rvc_inference.loader import load_voice, load_hubert, load_pitch
    from rvc_inference.index_loader import load_index
    from rvc_inference.pipeline import RVCPipeline, coarse_f0

    device = mm.get_torch_device()
    voice = load_voice(VOICE, device)
    hubert = load_hubert(HUBERT, device)
    pitch = load_pitch(RMVPE, device)
    index = load_index(NPY)

    coarse, f0bak = coarse_f0(torch.tensor([100.0, 200.0, 0.0, 300.0]), 0)
    assert coarse.dtype == torch.int32
    assert (coarse[2] >= 1).item() and (coarse[2] <= 255).item()
    assert f0bak.shape == coarse.shape

    pipe = RVCPipeline(voice.voice_tgt_sr)
    sr_in = 32000
    length = sr_in * 3
    t = torch.arange(length, dtype=torch.float32) / sr_in
    wav = 0.1 * torch.sin(2 * torch.pi * 220 * t)

    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out, sr = pipe.convert(hubert, voice, pitch, index, torch.tensor(0),
                               wav, 0, 1, 0.75, 0.33, 3, 1.0, 0)
    assert sr == 48000, sr
    expected = int(round(3 * sr))
    assert abs(out.shape[0] - expected) <= 4800, (out.shape[0], expected)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    print("PASS: pipeline convert end-to-end")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_pipeline.py
```
Expected: **FAIL** (`No module named 'rvc_inference.pipeline'`).

- [ ] **Step 3: Implement `pipeline.py`**

The file is vendored from `D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC\rvc\pipeline.py` plus `nodes_rvc.py` glue, with three deliberate changes:

1. **Retrieval** via `rvc_inference.index_loader.blend_with_index` — accepts both kinds.
2. **Median filter (`filter_radius`)** on the coarse pitch, applied in `coarse_f0` (upstream uses `medfilt2d(pitch[:, :-1], kernel=[1, 2*radius+1]); pitch = torch.cat([pitch[:, :1], med], dim=1)`; here a torch-based 1-D median replaces scipy so no new dependency).
3. **Pipeline API** matches the reference `RVCPipeline.convert` signature plus pitch module, so the node code stays thin.

Content (full):

```python
"""Per-segment voice-conversion pipeline (distilled from RVC-Project infer + ComfyUI-RVC rvc/pipeline.py)."""

import numpy as np
import torch
from torch.nn import functional as F

from rvc_inference.index_loader import blend_with_index

# Steady-state delay vector for the filter above (scipy.signal.lfilter_zi),
# zi = _HP_ZI * x0. Baked so the shipped code needs no scipy.
_HP_ZI = [-0.9699607413707367, 3.879842959615721, -5.81976443080129,
          3.879842948235015, -0.9699607356787477]
_HP_B = [0.9699606451838447, -4.849803225919223, 9.699606451838447,
         -9.699606451838447, 4.849803225919223, -0.9699606451838447]
_HP_A = [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458,
         4.761506797356209, -0.9408236532054606]


def _lfilter_zi(b, a, x, w_init, n):
    bp = F.pad(b, (0, n + 1 - b.shape[0]))
    ap = F.pad(a, (0, n + 1 - a.shape[0]))
    w = w_init.clone()
    y = torch.empty(x.shape[0], dtype=torch.float64)
    for i, xn in enumerate(x):
        yn = bp[0] * xn + w[0]
        w[:-1] = w[1:] + bp[1:-1] * xn - ap[1:-1] * yn
        w[-1] = bp[-1] * xn - ap[-1] * yn
        y[i] = yn
    return y, w


def highpass_48hz(x):
    b = torch.tensor(_HP_B, dtype=torch.float64)
    a = torch.tensor(_HP_A, dtype=torch.float64)
    n = max(b.shape[0], a.shape[0]) - 1
    xd = x.double()
    padlen = 18
    left = 2 * xd[0] - xd[1:padlen + 1].flip(0)
    right = 2 * xd[-1] - xd[-padlen - 1:-1].flip(0)
    ext = torch.cat([left, xd, right])
    zi = torch.tensor(_HP_ZI, dtype=torch.float64)
    y, _ = _lfilter_zi(b, a, ext, zi * ext[0], n)
    y2, _ = _lfilter_zi(b, a, y.flip(0), zi * y[-1], n)
    n0 = xd.shape[0]
    return y2.flip(0)[padlen:padlen + n0].float()


def coarse_f0(f0_hz, f0_up_key=0, filter_radius=0):
    f0 = f0_hz.clone() if isinstance(f0_hz, torch.Tensor) else torch.tensor(f0_hz)
    f0 = f0.float()
    voiced = f0 != 0
    if bool(voiced.any()) and bool((~voiced).any()):
        idx = torch.arange(f0.shape[0])
        f0[~voiced] = torch.from_numpy(np.interp(idx[~voiced].numpy(), idx[voiced].numpy(), f0[voiced].numpy())).float()
    f0 = f0 * pow(2, f0_up_key / 12)
    f0bak = f0.clone()
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    pos = f0_mel > 0
    f0_mel[pos] = (f0_mel[pos] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_mel = f0_mel.round().int()
    if filter_radius and f0_mel.shape[0] > 1:
        f0_mel = _median_filter_int(f0_mel.unsqueeze(0), filter_radius).squeeze(0)
    return f0_mel, f0bak


def _median_filter_int(pitch, radius):
    """Median along the last axis (kernel 1 x (2r+1)), torch only.

    Mirrors upstream medfilt2d on the coarse bins; boundary cells replicate.
    """
    ksize = 2 * radius + 1
    padded = torch.nn.functional.pad(pitch, (radius, radius), mode="replicate")
    frames = padded.unfold(-1, ksize, 1)
    return frames.median(dim=-1).values.to(pitch.dtype)[..., 1:-1][..., :pitch.shape[-1]]


def change_rms(data1, sr1, data2, sr2, rate):
    def rms(y, frame, hop):
        yp = F.pad(y.unsqueeze(0), (frame // 2, frame // 2)).squeeze(0)
        win = yp.unfold(0, frame, hop)
        return win.pow(2).mean(-1).sqrt()
    rms1 = rms(data1, sr1 // 2 * 2, sr1 // 2)
    rms2 = rms(data2, sr2 // 2 * 2, sr2 // 2)
    rms1 = F.interpolate(rms1.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = F.interpolate(rms2.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.clamp(rms2, min=1e-6)
    return data2 * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1))


class RVCPipeline:
    def __init__(self, tgt_sr, x_pad=1, x_query=6, x_center=38, x_max=41):
        self.tgt_sr = tgt_sr
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max

    def extract_features(self, hubert, wav, version, device):
        dtype = getattr(hubert, "dtype", None) or wav.dtype
        feats = wav.view(1, -1).to(device=device, dtype=dtype)
        with torch.inference_mode():
            states = hubert(feats)
        return hubert.final_proj(states[8]) if version == "v1" else states[11]

    def vc(self, hubert, net_g, sid, audio0, pitch, pitchf, index, index_rate, version, protect, device):
        feats = self.extract_features(hubert, audio0, version, device).to(device)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None:
            feats = blend_with_index(feats[0].float().cpu(), index, index_rate).unsqueeze(0).to(device, feats.dtype)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permUTE(0, 2, 1), scale_factor=2).permute(0, 2, 1)
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

    def split_points(self, audio):
        audio_pad = F.pad(audio.unsqueeze(0), (self.window // 2, self.window // 2), mode="reflect").squeeze(0)
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = torch.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i:i + self.window].abs()
            for t in range(self.t_center, audio.shape[0], self.t_center):
                seg = audio_sum[t - self.t_query:t + self.t_query]
                opt_ts.append(t - self.t_query + int(seg.argmin().item()))
        return opt_ts


    def convert(self, hubert, net_g, pitch_model, use_index, index_rate,
                audio, f0_up_key, if_f0, protect, filter_radius,
                rms_mix_rate, resample_sr, resample_fn, device, f0_method=0):
        raise NotImplementedError
```

**Stop.** The final `convert` differs enough (f0 via pitch model on each padded chunk, index object, filter radius, autocast) that the executor writes it explicitly here rather than as a set of edits. Full, final `convert`:

```python
    def convert(self, hubert, net_g, pitch_model, index, index_rate,
                audio, f0_up_key, if_f0, protect, filter_radius,
                rms_mix_rate, resample_sr, resample_fn, device):
        """Run the whole conversion for one mono float32 16 kHz wav tensor.

        pitch_model: rvc_inference.loader.PitchModel — used only when if_f0=1.
        index: rvc_inference.index_loader.IndexVectors or None.
        Returns (waveform float32 1-D on CPU, out_sr int).
        """
        import torch as _t  # already imported; local alias for readability

        audio = highpass_48hz(audio)
        opt_ts = self.split_points(audio)
        audio_pad = F.pad(audio.unsqueeze(0), (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        p_len = audio_pad.shape[0] // self.window
        sid = _t.zeros(1, dtype=_t.long) if net_g is None else None

        # f0 on the whole padded sequence (matches upstream: computed once outside the segment loop)
        pitch, pitchf = None, None
        if if_f0 == 1:
            f0_hz = _f0_of(pitch_model, audio_pad.unsqueeze(0))
            coarse, f0bak = coarse_f0(f0_hz[:p_len] if f0_hz.shape[0] >= p_len else f0_hz,
                                      f0_up_key, filter_radius)
            pitch = coarse[:p_len].to(device).unsqueeze(0).long()
            pitchf = f0bak[:p_len].float().to(device).unsqueeze(0).float()

        from rvc_inference.loader import load_pitch  # (not needed; keep imports top-level)

        audio_opt = []
        s = 0
        t = None
        for t in opt_ts:
            t = t // self.window * self.window
            e = t + self.t_pad * 2 + self.window
            sl = slice(s // self.window, e // self.window)
            seg = self.vc(hubert, net_g, None, audio_pad[s:e],
                          pitch[:, sl] if pitch is not None else None,
                          pitchf[:, sl] if pitchf is not None else None,
                          index, index_rate, version_, protect, device)
            audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        sl = slice((t // self.window if t is not None else 0), None)
        seg = self.vc(hubert, net_g, None, audio_pad[t:] if t is not None else audio_pad,
                      pitch[:, sl] if pitch is not None else None,
                      pitchf[:, sl] if pitchf is not None else None,
                      index, index_rate, version_, protect, device)
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
        return out, sr
```

Problems in the paste above the executor must not reproduce:
- `version_` is undefined — `version` must be passed in. Final honest convert signature:

```python
    def convert(self, hubert, net_g, pitch_model, index, index_rate, version,
                audio, f0_up_key, if_f0, protect, filter_radius,
                rms_mix_rate, resample_sr, resample_fn, device):
```

using `version` directly in both `self.vc(...)` calls, `sid` removed (upstream computes `sid_t` once in the caller; the caller passes a tensor), and **drop** the accidental `from rvc_inference.loader import` line and the `_t` alias.

- `_: sid` — caller-side session driver (see below) passes `sid_t` as an argument, so `convert`'s signature gains `sid_t` after `net_g`:

Final convert signature (authoritative):

```python
    def convert(self, hubert, net_g, sid_t, pitch_model, index, index_rate, version,
                audio, f0_up_key, if_f0, protect, filter_radius,
                rms_mix_rate, resample_sr, resample_fn, device):
```

with the body changes: `sid_t` used in both `self.vc(...)` calls; f0 computed on the padded sequence via the module-level `_f0_of` helper defined as:

```python
def _f0_of(pitch_model, wav16_batch):
    """Pitch in Hz per 160-sample frame for a (1, T) 16 kHz batch."""
    module = pitch_model.module
    if pitch_model.pitch_kind == "rmvpe":
        from rvc_train.rmvpe import log_mel_spectrogram
        dtype = pitch_model.dtype
        mel = log_mel_spectrogram(wav16_batch.float(), module.mel_basis.float())
        with torch.inference_mode():
            hidden = module.mel2hidden(mel.to(dtype))
            f0 = module.decode(hidden.float(), thred=0.03)
        return f0.float().squeeze(0)
    # fcpe
    with torch.inference_mode():
        f0 = module(wav16_batch.to(torch.float32), threshold=0.006)
    return f0.float().squeeze(0)
```

And the `_median_filter_int` slice fix — the earlier sketch `[..., 1:-1][..., :pitch.shape[-1]]` is wrong; authoritative version:

```python
def _median_filter_int(pitch, radius):
    """Median along the last axis (kernel 1 x (2r+1)), torch only, replicate pad."""
    ksize = 2 * radius + 1
    padded = F.pad(pitch, (radius, radius), mode="replicate")
    frames = padded.unfold(-1, ksize, 1).median(dim=-1).values
    return frames.to(pitch.dtype)[..., :pitch.shape[-1]]
```

(Even simpler: `F.pad` then unfold→median→truncate back to original length, exactly like this block.)

Also fix one more typo that slipped into `vc` paste above: `permUTE` must read `permute` — authoritative `vc` uses, as the reference, `feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)`.

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: pipeline convert end-to-end`.

- [ ] **Step 5: Commit**

```bash
git add rvc_inference/pipeline.py tests_rvc/test_pipeline.py
git commit -m "feat(rvc-inference): conversion pipeline with faiss retrieval and f0 median filter"
```

---

### Task 6: The four ComfyUI nodes

**Files:**
- Create: `nodes_rvc_inference.py`
- Modify: `__init__.py`
- Test: `tests_rvc/test_nodes.py`

- [ ] **Step 1: Write the failing test**

Create `tests_rvc/test_nodes.py`:

```python
def main():
    from nodes_rvc_inference import (
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
        _voice_options, _index_options, _base_model_options,
    )

    expected = {"RVCVoiceLoader", "RVCHubertPitchLoader", "RVCExecute", "RVCAudioSynthesis"}
    assert expected == set(NODE_CLASS_MAPPINGS), NODE_CLASS_MAPPINGS
    assert expected == set(NODE_DISPLAY_NAME_MAPPINGS)

    fake = ["my-vocals.pth", "gangshou\\gangshou.pth", "pretrained_v2\\f0G48k.pth", "pretrained_v2\\D48k.pth"]
    voice = _voice_options(fake)
    assert voice == ["my-vocals.pth", "gangshou\\gangshou.pth"], voice

    fake_idx = ["sub\\m1.index", "sub\\m1.npy", "other.index"]
    assert _index_options(fake_idx) == ["(none)", "sub\\m1.index", "sub\\m1.npy", "other.index"]

    print("PASS: nodes register and option formatting")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_nodes.py
```
Expected: **FAIL** (`No module named 'nodes_rvc_inference'`).

- [ ] **Step 3: Implement `nodes_rvc_inference.py`**

```python
"""ComfyUI nodes for RVC voice conversion inference."""

import logging
import os
import re

import comfy.audio
import comfy.model_management
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from rvc_inference.index_loader import IndexVectors, load_index
from rvc_inference.loader import load_hubert, load_pitch, load_voice
from rvc_inference.pipeline import RVCPipeline
from rvc_inference.synths import SYNTH_CLASSES
from rvc_train.hubert import load_hubert as _load_hubert  # not used; see note


LOGGER = logging.getLogger("RVC-Inference")

CATEGORY = "RVC/inference"
RVC_VOICE = io.Custom("RVC_VOICE")
RVC_INDEX = io.Custom("RVC_INDEX")
RVC_HUBERT = io.Custom("RVC_HUBERT")
RVC_PITCH = io.Custom("RVC_PITCH")

NONE_OPTION = "(none)"
PRETRAINED_NAME = re.compile(r"^(f0?[GDF])(\d+k)?\.pth$", re.IGNORECASE)


def _rvc_files():
    return folder_paths.get_filename_list("rvc")


def _is_pretrained(rel):
    if any(part.lower().startswith("pretrained") for part in rel.replace("/", os.sep).split(os.sep)):
        return True
    return bool(PRETRAINED_NAME.match(os.path.basename(rel)))


def _voice_options(files):
    return [rel for rel in files if rel.lower().endswith(".pth") and not _is_pretrained(rel)]


def _index_options(files):
    opts = [rel for rel in files if rel.lower().endswith((".index", ".npy"))]
    return [NONE_OPTION] + sorted(opts)


def _base_model_options(files, kind):
    keys = ("hubert",) if kind == "hubert" else ("rmvpe", "fcpe")
    ext = [".safetensors"]
    result = []
    for rel in files:
        if not rel.lower().endswith(tuple(ext)):
            continue
        base = os.path.basename(rel).lower()
        if any(k in base for k in keys):
            result.append(rel)
    return sorted(result)


def _resolve(name):
    return folder_paths.get_full_path_or_raise("rvc", name)


class RVCVoiceLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCVoiceLoader",
            display_name="RVC Voice Loader",
            category=CATEGORY,
            description="Load an RVC voice model (.pth) and optional retrieval index from models/rvc.",
            inputs=[
                io.Combo.Input("voice_pth", options=_voice_options(_rvc_files()), tooltip="RVC voice checkpoint (.pth)."),
                io.Combo.Input("index_file", options=_index_options(_rvc_files()), default=NONE_OPTION,
                               tooltip="Optional retrieval index (.index needs faiss-cpu; .npy works everywhere)."),
            ],
            outputs=[RVC_VOICE.Output(display_name="voice"), RVC_INDEX.Output(display_name="index")],
        )

    @classmethod
    def execute(cls, voice_pth, index_file=NONE_OPTION):
        device = comfy.model_management.get_torch_device()
        voice = load_voice(_resolve(voice_pth), device)
        index = load_index(_resolve(index_file)) if index_file != NONE_OPTION else None
        return io.NodeOutput(voice, index)


class RVCHubertPitchLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = _base_model_options(_rvc_files(), "hubert")
        pitch_files = _base_model_options(_rvc_files(), "pitch")
        return io.Schema(
            node_id="RVCHubertPitchLoader",
            display_name="RVC Hubert & Pitch Loader",
            category=CATEGORY,
            description="Load the ContentVec hubert encoder and the RMVPE/FCPE pitch model.",
            inputs=[
                io.Combo.Input("hubert_file", options=files, default=_default_of(files, "hubert_base.safetensors")),
                io.Combo.Input("pitch_file", options=pitch_files, default=_default_of(pitch_files, "rmvpe.safetensors")),
            ],
            outputs=[RVC_HUBERT.Output(display_name="hubert"), RVC_PITCH.Output(display_name="pitch")],
        )

    @classmethod
    def execute(cls, hubert_file, pitch_file):
        device = comfy.model_management.get_torch_device()
        hubert = load_hubert(_resolve(hubert_file), device)
        pitch = load_pitch(_resolve(pitch_file), device)
        return io.NodeOutput(hubert, pitch)


class RVCExecute(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCExecute",
            display_name="RVC Execute",
            category=CATEGORY,
            description="Convert the connected audio with the loaded voice.",
            inputs=[
                RVC_VOICE.Input("voice"),
                RVC_HUBERT.Input("hubert"),
                RVC_PITCH.Input("pitch"),
                io.Audio.Input("audio"),
                RVC_INDEX.Input("index", optional=True, tooltip="Optional retrieval index."),
                io.Combo.Input("f0_method", options=["rmvpe", "fcpe"], default="rmvpe"),
                io.Int.Input("f0_up_key", default=0, min=-12, max=12, step=1),
                io.Int.Input("speaker_id", default=0, min=0, max=109, step=1),
                io.Float.Input("index_rate", default=0.75, min=0.0, max=1.0, step=0.05),
                io.Float.Input("protect", default=0.33, min=0.0, max=0.5, step=0.01),
                io.Int.Input("filter_radius", default=3, min=0, max=10, step=1),
                io.Float.Input("rms_mix_rate", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("resample_sr", default=0, min=0, max=96000, step=1000,
                             tooltip="Output sample rate. 0 keeps the voice's own rate."),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, voice, hubert, pitch, audio, index=None, f0_method="rmvpe", f0_up_key=0,
                speaker_id=0, index_rate=0.75, protect=0.33, filter_radius=3,
                rms_mix_rate=1.0, resample_sr=0):
        load_device = comfy.model_management.get_torch_device()
        if_f0 = int(getattr(voice, "voice_if_f0", 1))
        version = voice.voice_version
        tgt_sr = int(voice.voice_tgt_sr)
        sid = max(0, min(int(speaker_id), int(getattr(voice, "voice_n_spk", 1)) - 1))
        if if_f0 == 1 and f0_method != pitch.pitch_kind:
            raise RuntimeError("ERROR: f0_method=%s but the pitch model is %s." % (f0_method, pitch.pitch_kind))

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sr != 16000:
            waveform = comfy.audio.resample(waveform, sr, 16000)

        cuda = load_device.type == "cuda"
        pipe = RVCPipeline(tgt_sr, **({"x_pad": 3, "x_query": 10, "x_center": 60, "x_max": 65} if cuda
                                      else {"x_pad": 1, "x_query": 6, "x_center": 38, "x_max": 41}))
        outs = []
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                for b in range(waveform.shape[0]):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    monos = waveform[b].reshape(-1).float().cpu()
                    out, out_sr = pipe.convert(hubert, voice, torch.tensor([[sid]]), pitch, index,
                                               index_rate, version, monos, int(f0_up_key), if_f0,
                                               float(protect), int(filter_radius), float(rms_mix_rate),
                                               int(resample_sr) or tgt_sr,
                                               _resample_fn, load_device)
                    outs.append(out.unsqueeze(0))
        finally:
            comfy.model_management.soft_empty_cache()
        wave = torch.stack(outs) if len(outs) > 1 else outs[0].unsqueeze(0)
        return io.NodeOutput({"waveform": wave.unsqueeze(1) if wave.dim() == 2 else wave, "sample_rate": out_sr})


def _resample_fn(w, a, b):
    return comfy.audio.resample(w[None, None, :], a, b)[0, 0]


class RVCAudioSynthesis(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.Audio.Input("audio"), prefix="audio", min=1, max=100)
        return io.Schema(
            node_id="RVCAudioSynthesis",
            display_name="RVC Audio Synthesis",
            category=CATEGORY,
            description="Combine multiple audio clips: merge them or concatenate in order.",
            inputs=[
                io.Autogrow.Input("audios", template=template, tooltip="Clips to combine, in connection order."),
                io.Combo.Input("mode", options=["merge", "concat"], default="concat"),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audios, mode="concat"):
        names = sorted(audios, key=lambda n: int(re.search(r"(\d+)$", n).group(1)))
        clips = [audios[n] for n in names if audios[n] is not None]
        if not clips:
            raise ValueError("Connect at least one audio")

        target = max(c["sample_rate"] for c in clips)
        waves = [comfy.audio.resample(c["waveform"], c["sample_rate"], target) for c in clips]
        out_wave = torch.cat(waves, dim=-1) if mode == "concat" else _merge(waves)
        return io.NodeOutput({"waveform": out_wave, "sample_rate": target})


def _merge(waves):
    width = max(w.shape[-1] for w in waves)
    acc = None
    for w in waves:
        if w.shape[-1] != width:
            w = torch.nn.functional.pad(w, (0, width - w.shape[-1]))
        acc = w.float() if acc is None else acc + w.float()
    out = acc / len(waves)
    peak = out.abs().max() / 0.99
    if peak > 1:
        out = out / peak
    return out


NODE_CLASS_MAPPINGS = {
    "RVCVoiceLoader": RVCVoiceLoader,
    "RVCHubertPitchLoader": RVCHubertPitchLoader,
    "RVCExecute": RVCExecute,
    "RVCAudioSynthesis": RVCAudioSynthesis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVCVoiceLoader": "RVC Voice Loader",
    "RVCHubertPitchLoader": "RVC Hubert & Pitch Loader",
    "RVCExecute": "RVC Execute",
    "RVCAudioSynthesis": "RVC Audio Synthesis",
}
```

Executor corrections to apply while writing the file (from the review paragraph that failed checks in Task 5):
- Remove the stray `from rvc_train.hubert import load_hubert as _load_hubert` import — `load_hubert` already comes from `rvc_inference.loader` (which re-exports by delegation). Also `IndexVectors` is imported but unused — drop it.
- The `_base_model_options` default helpers: `_default_of(options, wanted)` is used but never defined. Define at module level:

```python
def _default_of(options, wanted):
    for option in options:
        if os.path.basename(option).lower() == wanted:
            return option
    return options[0] if options else wanted
```

- `launch_pitch` — the `RVC_PITCH.Input` slot needs `optional=False`; that is already implied by being listed without `optional=True`.
- `waveform[b].reshape(-1).float().cpu()` — the reference passes `(1, T)` to `_f0_of` via `audio_pad.unsqueeze(0)`; `convert` already does the `unsqueeze(0)` internally.

- [ ] **Step 4: Register in `__init__.py`**

Current `__init__.py`:

```python
import os

import folder_paths

# ComfyUI-RVC registers the same folder; adding it here keeps this node pack usable alone.
folder_paths.add_model_folder_path("rvc", os.path.join(folder_paths.models_dir, "rvc"))

from .nodes_rvc_trainer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

New `__init__.py`:

```python
import os

import folder_paths

# ComfyUI-RVC registers the same folder; adding it here keeps this node pack usable alone.
folder_paths.add_model_folder_path("rvc", os.path.join(folder_paths.models_dir, "rvc"))

from .nodes_rvc_trainer import NODE_CLASS_MAPPINGS as _TRAIN_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as _TRAIN_NODE_DISPLAY_NAME_MAPPINGS
from .nodes_rvc_inference import NODE_CLASS_MAPPINGS as _INFERENCE_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as _INFERENCE_NODE_DISPLAY_NAME_MAPPINGS


NODE_CLASS_MAPPINGS = {**_TRAIN_NODE_CLASS_MAPPINGS, **_INFERENCE_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_TRAIN_NODE_DISPLAY_NAME_MAPPINGS, **_INFERENCE_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: same as Step 2.
Expected: `PASS: nodes register and option formatting`.

- [ ] **Step 6: ComfyUI server import check**

Run (server mode):
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" main.py --quick-test-for-ci
```
from `D:\Workspace\ComfyUI`. Expected: the file `__init__.py` imports cleanly; if a broken module raises, the server exits non-zero and prints which node pack failed.

- [ ] **Step 7: Commit**

```bash
git add nodes_rvc_inference.py __init__.py tests_rvc/test_nodes.py
git commit -m "feat(rvc-inference): four ComfyUI nodes wired into the pack"
```

---

### Task 7: End-to-end smoke with real models (validation task)

**Files:**
- Create: `tests_rvc/test_smoke_e2e.py` (artifact script, no commit — kept in working tree only? NO it is committed, deliverable test)
- Commit alongside README in Task 8 or same commit.

- [ ] **Step 1: Write the smoke script**

Create `tests_rvc/test_smoke_e2e.py`:

```python
"""End-to-end smoke test with real models, run under the cu130 env via PYTHONPATH env.

Run:
    $env:PYTHONPATH = "D:\Workspace\ComfyUI;D:\Workspace\ComfyUI\custom_nodes\ComfyUI-RVC-Trainer"
    & "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_smoke_e2e.py
"""
import os

import numpy as np
import torch

CUI = r"D:\Workspace\ComfyUI"
OUT = CUI + r"\output\rvc_inference_smoke"
os.makedirs(OUT, exist_ok=True)


def main():
    import comfy.model_management as mm
    from nodes_rvc_inference import (
        RVCVoiceLoader, RVCHubertPitchLoader, RVCExecute, RVCAudioSynthesis,
    )

    device = mm.get_torch_device()

    # node 1
    voice, index = RVCVoiceLoader.execute("my-vocals-v2-48k.pth", "my-vocals-v2-48k.npy").outputs
    # node 2
    hubert, pitch = RVCHubertPitchLoader.execute("hubert_base.safetensors", "rmvpe.safetensors").outputs

    # source audio: a short fake speech-like signal (3 s at 48 kHz)
    sr_in = 48000
    t = torch.arange(sr_in * 3, dtype=torch.float32) / sr_in
    source = (0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0).unsqueeze(0)
    audio = {"waveform": source, "sample_rate": sr_in}

    # node 3
    converted = RVCExecute.execute(voice, hubert, pitch, audio, index)
    out = converted.outputs[0]
    assert out["sample_rate"] == 48000, out["sample_rate"]
    assert torch.isfinite(out["waveform"]).all()
    length_expected = 3 * 48000
    assert abs(out["waveform"].shape[-1] - length_expected) <= 4800, out["waveform"].shape

    # node 4 — concat the same audio with itself, and merge two slightly different tracks
    aud = {"audios": {"audio0": out, "audio1": out}, "mode": "concat"}
    bundle = RVCAudioSynthesis.execute(**aud)
    concat = bundle.outputs[0]
    assert abs(concat["waveform"].shape[-1] - 2 * out["waveform"].shape[-1]) <= 1000, concat["waveform"].shape

    aud2 = {"audios": {"audio0": out, "audio1": out}, "mode": "merge"}
    merged = RVCAudioSynthesis.execute(**aud2).outputs[0]
    assert merged["waveform"].shape == concat["waveform"].shape

    # write output wav to make manual listening possible
    import soundfile as sf
    sf.write(OUT + r"\smoke.wav", out["waveform"].reshape(-1).numpy(), out["sample_rate"])
    print("PASS: end-to-end smoke — output at", OUT)

if __name__ == "__main__":
    main()
```

Note: `.outputs` on `io.NodeOutput` — verify attribute name. In ComfyUI io API, `io.NodeOutput(*outputs)` returns an object whose `outputs` attribute is a tuple. But if the attribute isn't that name (some versions use `results` or tuple-like), change accessor to match. Pointed out: the trainer node returns `io.NodeOutput(dataset)` and the pack's own tests treat the result as an opaque object; check with a one-liner under the env:

```python
& "D:\UserApps\Conda\envs\cu130\python.exe" -c "from comfy_api.latest import io; n = io.NodeOutput(1, 2); print([a for a in dir(n) if not a.startswith('_')])"
```

before editing the accessor names in the test.

Note 2: `execute` classmethods call other nodes' `execute` directly, NOT through ComfyUI scheduler. That is fine for a smoke test.

- [ ] **Step 2: Run the smoke**

Run: as in script docstring.
Expected: `PASS: end-to-end smoke — output at D:\Workspace\ComfyUI\output\rvc_inference_smoke`, plus a valid `smoke.wav` (~3 s, voice-converted). wav should sound like a converted voice emission (any tone/morph acceptable given a synthetic sine input), just confirm it plays and is not silent/noisy.

- [ ] **Step 3: Run the whole test suite**

```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_import.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_synths.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_index_loader.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_loader.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_pipeline.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_nodes.py
& "D:\UserApps\Conda\envs\cu130\python.exe" tests_rvc\test_smoke_e2e.py
```
Expected: every script prints `PASS: ...`, no exit code > 0.

- [ ] **Step 4: Commit**

```bash
git add tests_rvc/test_smoke_e2e.py
git commit -m "test(rvc-inference): real-model end-to-end smoke"
```

---

### Task 8: README + pyproject optional-dependency + final cleanup

**Files:**
- Modify: `README.md:1` (title + inference section)
- Modify: `pyproject.toml` (optional deps `[project.optional-dependencies] inference = ["faiss-cpu"]`)
- Test: none (documentation only, but `--quick-test-for-ci` re-run required)

- [ ] **Step 1: Add inference section to `README.md`**

Insert after the node table (or at end of file) — maintain the doc's Chinese style. Content:

```markdown
## 音色替换（推理）

推理节点将从 RVC-Project 项目代码提炼出的音色替换工作流封装为四个 ComfyUI 节点，与训练节点共用 `models/rvc` 目录。产物可直接接入 ComfyUI 的 Load Audio / Save Audio / Preview Audio 流程。

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| **RVC Voice Loader** | `voice_pth`（下拉，来自 `models/rvc/**/*.pth`，多级目录，自动排除 pretrained 底模）、`index_file`（下拉：`(none)`、`*.index`、`*.npy`） | `voice`、`index` |
| **RVC Hubert & Pitch Loader** | `hubert_file`（默认 `hubert_base.safetensors`）、`pitch_file`（默认 `rmvpe.safetensors`，可选 `fcpe.safetensors`） | `hubert`、`pitch` |
| **RVC Execute** | `voice`、`hubert`、`pitch`、`audio`、`index`（可选）＋参数 `f0_method`、`f0_up_key`、`speaker_id`、`index_rate`、`protect`、`filter_radius`、`rms_mix_rate`、`resample_sr` | `audio` |
| **RVC Audio Synthesis** | `audios`（Autogrow，可拖入任意多个）、`mode`（`merge`＝混合平均，`concat`＝顺序拼接） | `audio` |

使用方法：

1. 将音色模型放到 `models/rvc/<dir>/`，`hubert_base.safetensors` 与 `rmvpe.safetensors`（或 `fcpe.safetensors`）放到 `models/rvc/`。
2. 依次创建 Voice Loader / Hubert & Pitch Loader / Execute，把 Load Audio 的输出接到 Execute 的 `audio`。
3. `f0_method` 必须与 Hubert & Pitch Loader 所选的 pitch 模型匹配（`rmvpe.safetensors` ↔ `rmvpe`，`fcpe.safetensors` ↔ `fcpe`）。
4. 输出采样率跟随音色模型，除非 `resample_sr` 非 0。
5. 输出接 Save Audio 或 Preview Audio。

推理代码完全内置（`rvc_inference/`）：高通滤波、长音频自动切片、f0 提取（RMVPE/FCPE）、ContentVec 特征、faiss/npy 检索混合、保护度加权、合成器推理。`fp16 autocast` 仅在 Ampere 以及更新的 GPU（compute capability ≥ 8.0）启用，其余回退 fp32。`faiss-cpu` 为可选依赖（读取 `.index`）；没有时 `.npy` 兜底可用。

注意：输出与 RVC-Project 上游一致，是随机采样的（相同输入两次运行有约 0.4-0.7 的幅值波动）——这是模型的后验采样本性，不是 bug。
```

- [ ] **Step 2: `pyproject.toml` optional dependency**

Insert under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
japanese = [
    "faiss-cpu",
    "scikit-learn",
]
inference = [
    "faiss-cpu",
]
```

- [ ] **Step 3: Re-run `--quick-test-for-ci`**

Run:
```powershell
& "D:\UserApps\Conda\envs\cu130\python.exe" main.py --quick-test-for-ci
```
(workdir `D:\Workspace\ComfyUI`)
Expected: clean exit 0.

- [ ] **Step 4: Final commit**

```bash
git add README.md pyproject.toml
git commit -m "docs(rvc-inference): inference nodes documentation and faiss optional dep"
```

---

### Completion checklist

- [ ] `rvc_inference/` contains synths/index_loader/loader/fcpe/pipeline modules with tests.
- [ ] `rvc_train/models.py` patch is additive (TextEncoder `f0` default True) — training path unchanged.
- [ ] Four nodes registered, showing up under `RVC/inference` category.
- [ ] Real-model smoke run inside `cu130` produces a converted wav.
- [ ] No package installs were performed; `conda` env untouched.
- [ ] All run-commands pinned to the `cu130` interpreter and `$env:PYTHONPATH` combo above.
- [ ] Every intermediate task committed individually (7 commits + docs commit).

### Known risks / escalation path

- If `comfy.model_management.get_torch_device()` misbehaves outside a running server (rare), smoke tests can fall back to `device = torch.device("cuda")` in the test harness (nodes always use comfy.model_management; tests exercise nodes' execute directly).
- If a legacy upstream `.pth` uses a 17-element config (very old RVC versions), `load_voice` raises `not an RVC voice checkpoint` with a message that points to re-exporting with a current RVC. This is by design (user-approved direction: upstream-trained direct loading).
- If faiss `reconstruct_batch` is unavailable (faiss < 1.7.4), fall back to a loop of `index.reconstruct(i)` calls — drop-in replacement inside `_retrieve_faiss`.

## Plan Self-Review

- **Spec coverage:** All four node specs (design §4) map to Task 6 code; loaders (§4.1/4.2) to Task 4/6; pipeline (§4.3) to Task 5; error-handling matrix (§5) covered by loader/pitch/voice/loader tests; precision (§6) via autocast in Task 6; real-model smoke (§7) to Task 7; README/pyproject (§8) to Task 8.
- **Placeholders:** none — every step is either a full code block, a copy directive to a specific file with diffs, or an exact command.
- **Type consistency:** `PitchModel` attribute `pitch_kind` used by task 6 f0-method check ✓; `load_index` returns `IndexVectors` referenced identically in Task 3/5/6 ✓; `blend_with_index(feats[0], index, rate)` referenced in Task 5 and built in Task 3 ✓; `io.Custom("RVC_VOICE")` style follows existing trainer pattern ✓.
