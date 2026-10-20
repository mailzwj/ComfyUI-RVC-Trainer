# RVC Voice Conversion (音色替换) Inference Nodes — Design

Date: 2026-06-16  
Status: Approved  
Scope: add four inference nodes to the existing ComfyUI-RVC-Trainer package, distilled from the [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) workflow.

---

## 1. Goals

Expose the upstream RVC inference workflow as four ComfyUI nodes:

1. **RVC Voice Loader** — pick a voice `.pth` + optional feature index (`.index`/`.npy`) from `models/rvc/**`.
2. **RVC Hubert & Pitch Loader** — pick `hubert_base.safetensors` (default) and `rmvpe.safetensors` (default) / `fcpe.safetensors` from `models/rvc/**`.
3. **RVC Execute** — run the actual voice conversion pipeline on a connected `AUDIO`.
4. **RVC Audio Synthesis** — merge or concat multiple `AUDIO` inputs.

Non-goals: training changes, new GUI panels, external repo dependencies at runtime for the base install.

## 2. Decisions (user-confirmed)

| Decision | Choice |
|---|---|
| Model source | load upstream-trained `.pth` directly (convert layout at load time) |
| FAISS index | read native `.index` with `faiss-cpu` (optional dependency), `.npy` fallback |
| Inference core | self-contained inside this repo (port from installed ComfyUI-RVC pack reference) |
| f0 methods | RMVPE + FCPE |
| Model discovery | ComfyUI `folder_paths` mechanism, folder key `rvc` |
| Package | join ComfyUI-RVC-Trainer, category `RVC/inference` |
| Params | full upstream parameter set |
| Synthesis input | `io.Autogrow` audio slots |
| Output sample rate | voice model's native rate (`resample_sr=0`) |
| Precision | fp16 autocast on Ampere+ GPU, fp32 on CPU/older |
| Node API | new `comfy_api.latest io` style (matches existing trainer nodes) |
| Excluded from voice list | pretrained base models (`pretrained_v2/…` G/D/f0G/f0D) |
| Validation | real-model smoke test inside `conda activate cu130`; do not pollute the env |

## 3. File layout

```
custom_nodes/ComfyUI-RVC-Trainer/
├── rvc_inference/
│   ├── __init__.py
│   ├── synths.py        # SynthesizerTrnMs256NSFsid / _nono / Ms768NSFsid / _nono
│   ├── hubert.py        # ContentVec Hubert encoder (fp16/fp32 load + forward)
│   ├── rmvpe.py         # RMVPE mel→f0 model (mel2hidden + decode)
│   ├── fcpe.py          # FCPE fast pitch model
│   ├── pipeline.py      # highpass48 → slice → f0+features → infer → rms-mix → normalize
│   ├── index_loader.py  # .index (faiss) / .npy / .pt loading + retrieval blend
│   └── loader.py        # voice .pth validation & conversion; hubert/pitch safetensors load
├── nodes_rvc_inference.py  # four io.ComfyNode definitions
└── __init__.py             # add nodes_rvc_inference to existing mappings
```

- `rvc_train/` (training nodes) stays untouched.
- `__init__.py` already registers `models/rvc` under folder key `rvc`; new nodes reuse it.
- Model discovery uses `folder_paths.get_filename_list("rvc")` recursive listing, shown as `subdir/file.ext`.

## 4. Node specs

### Node 1 — RVC Voice Loader (`RVCVoiceLoader`)

**Inputs**
- `voice_pth` COMBO — all `models/rvc/**/*.pth`, multi-level, `pretrained base` files (paths containing a `pretrained` path segment or beginning with `f0G/f0D/G/D`) excluded.
- `index_file` COMBO — `(none)` + all `models/rvc/**/*.index` and `*.npy`.

**Outputs** — `voice` (`RVC_VOICE`), `index` (`RVC_INDEX`, passes a `(none)`-safe value).

**Voice load steps**
1. `folder_paths.get_full_path_or_raise("rvc", relative)`.
2. `torch.load(weights_only=True)`.
3. Validate dict keys `weight`, `config` (18 elements), `version` (`v1`/`v2`), `f0` (0/1); select synthesizer class via `SYNTH_CLASSES[(version, f0)]`.
4. Strip `enc_q.*` weights, expand `emb_g` to the voice's speaker count from `config[-3]`.
5. Load to GPU with fp16 autocast when `comfy.model_management` supports it, fp32 otherwise.

**Index load steps**
- `.index`: read with `faiss.read_index`; requires `faiss-cpu`; set `nprobe=1` by default. If faiss is missing raise a clear RuntimeError telling the user to `pip install faiss-cpu` or pick a `.npy`.
- `.npy`: `np.load` → torch float32 [N, D] matrix.
- Index result carries `kind` (`faiss` or `feats`) so downstream retrieval knows which path to use.

### Node 2 — RVC Hubert & Pitch Loader (`RVCHubertPitchLoader`)

**Inputs**
- `hubert_file` COMBO — default `hubert_base.safetensors`; lists `models/rvc/**/*.safetensors` whose name hints hubert.
- `pitch_file` COMBO — default `rmvpe.safetensors`; lists safetensors whose name hints `rmvpe` or `fcpe`.

**Outputs** — `hubert` (`RVC_HUBERT`), `pitch` (`RVC_PITCH`).

**Load steps**
- Hubert: `safetensors.torch.load_file` → `HubertEncoder`; keeps module names `post_extract_proj`/`encoder`/`proj` as in `rvc_train/hubert.py`, loaded fp16 autocast on capable GPUs.
- Pitch: filename determines class (`rmvpe` → `RMVPEModel`, `fcpe` → `FCPitch`); both loaded with the same autocast policy.
- Clear error naming the offending file if the filename matches neither `hubert` nor `rmvpe`/`fcpe` hints; the fix is to rename the file or pick the matching entry from the dropdown.

### Node 3 — RVC Execute (`RVCExecute`)

**Inputs**
- `voice` `RVC_VOICE`, `hubert` `RVC_HUBERT`, `pitch` `RVC_PITCH`, `audio` `AUDIO` — required.
- `index` `RVC_INDEX` — optional.

**Config params**

| Param | Default | Range | Notes |
|---|---|---|---|
| `f0_method` | `rmvpe` | rmvpe/fcpe | must match the loaded pitch model |
| `f0_up_key` | 0 | −12…+12 | semitone shift |
| `index_rate` | 0.75 | 0…1 | retrieval blend; only used when index connected |
| `protect` | 0.33 | 0…0.5 | consonant protection; 0.5 disables blend |
| `filter_radius` | 3 | 0…10 | median smoothing on f0 |
| `rms_mix_rate` | 1.0 | 0…1 | 0 = fully copy input envelope, 1 = off |
| `resample_sr` | 0 | 0…96000 | 0 = voice native |
| `speaker_id` | 0 | 0…109 auto-clamped | multi-speaker voices |

**Conversion pipeline (mirrors upstream `infer/lib/rvc`)**
1. Convert input to mono, resample to 16 kHz via `comfy.audio.resample`.
2. 48 Hz high-pass filter (`pipeline.highpass_48hz`).
3. Long-audio smart slicing using CUDA-aware `x_pad/x_query/x_center/x_max`.
4. Per-segment: f0 estimate (RMVPE or FCPE) → median smoothing (`filter_radius`) → `coarse_f0` quantization (1–255 bins) + `pitchf` Hz track; Hubert features; faiss/npy retrieval blend at `index_rate`; `protect` weighting; synthesizer `infer`; collect per-segment output.
5. Concatenate segments, RMS-mix against original at `rms_mix_rate`, peak-normalize to 0.99, optional resample.
6. fp16 autocast on Ampere+, fp32 fallback.

**Output** — `audio: AUDIO` (`waveform` [B, 1, N] float32, `sample_rate` = voice target sr unless `resample_sr>0`).

### Node 4 — RVC Audio Synthesis (`RVCAudioSynthesis`)

**Inputs**
- `audios` Autogrow (`io.Autogrow.Input`, template prefix `audio`, min 1, max 100), user-ordered by connection.
- `mode` COMBO — `merge` | `concat`.

**Behavior**
- All inputs resample to the highest sample rate present.
- `merge`: elementwise mean of all tracks, peak-normalize to 0.99.
- `concat`: sequential `torch.cat` along time axis, no crossfade.
- Output preserves the chosen common sample rate and shape [B, 1, N].

**Output** — `audio: AUDIO`.

## 5. Error handling

| Case | Behavior |
|---|---|
| `.pth` missing `weight`/`config`/`version`/`f0` | RuntimeError `not an RVC voice checkpoint: <name>` |
| `.index` read but faiss missing | RuntimeError pointing to `pip install faiss-cpu` or use `.npy` |
| `.npy` wrong shape (not 2-D) | RuntimeError `not an RVC feature matrix` |
| f0 method ≠ loaded pitch model | Preflight check before any GPU work, suggest loading the other loader |
| `speaker_id` out of range | Clamp silently to `n_spk-1` |
| Stereo / non-16k input | Auto-downmix + resample (upstream behavior) |
| Audio with zero length | RuntimeError before inference |
| Windows sandbox safety | relative-only paths allowed via `folder_paths`, no absolute escapes |

## 6. Precision & performance

- `comfy.model_management.unet_dtype` + `unet_manual_cast` decide fp16 vs fp32.
- fp16 autocast active on Ampere+ GPUs (compute capability ≥ 8.0), fp32 on CPU/older.
- All segment outputs concatenated in fp32.

## 7. Tests

- Real-model smoke test inside `conda activate cu130`:
  - Load `my-vocals-v2-48k.pth` + `.npy/.index` + `hubert_base.safetensors` + `rmvpe.safetensors`.
  - Generate a short `AUDIO` batch through the four nodes end-to-end via a raw (non-GUI) script that instantiates the nodes directly.
  - Save output WAV, compare duration with input (same ±50 ms).
- Verify all four nodes appear in ComfyUI web UI and connect cleanly.
- `.npy` path is the only one requiring faiss; if faiss-cpu is absent in `cu130`, skip reading `.index` and still run a feature-blend test with `.npy` only.
- No pollution of conda env: if faiss must be tested, install to `--target` temp dir then clean up after test.
- No changes to training-node paths.

## 8. Risks

- Upstream `.pth` layout variety (old WebUI versions may use 17- vs 18-element config): handle 18 elements as the primary target; if a checkpoint carries a different element count, raise a clear error naming the count found rather than guessing.
- `faiss-cpu` wheel availability on Windows: optional dependency, users without it can still use `.npy`.
- Existing `ComfyUI-RVC` nodes coexist in the same ComfyUI install; node ids here will not collide (`RVCVoiceLoader`, `RVCHubertPitchLoader`, `RVCExecute`, `RVCAudioSynthesis`).
