# ComfyUI-RVC-Trainer

在 ComfyUI 内训练 RVC（Retrieval-based Voice Conversion）音色模型。训练代码使用VibeCoding参考[RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)项目实现：
静音切片、高通滤波、f0 提取、HuBERT 特征、对抗训练，以及导出的 `assets/weights/*.pth` 格式。

导出的音色可直接被 [ComfyUI-RVC](../ComfyUI-RVC) 加载——训练出的权重按该节点包期望的相同模块布局和 18 元素 config 列表保存。

## 模型链接

### RVC推理模型权重和预训练模型

- [fcpe.safetensors](https://huggingface.co/Aero-Ex/ComfyUI-RVC/resolve/main/rvc/fcpe.safetensors)
- [hubert_base.safetensors](https://huggingface.co/Aero-Ex/ComfyUI-RVC/resolve/main/rvc/hubert_base.safetensors)
- [rmvpe.safetensors](https://huggingface.co/Aero-Ex/ComfyUI-RVC/resolve/main/rvc/rmvpe.safetensors)
- [pretrained](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained)
- [pretrained_v2](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2)

网盘直达：https://pan.quark.cn/s/299ebe4a22ff

### 模型存放位置

`ComfyUI/models/rvc/`

## 音色训练

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| **RVC Train Dataset** | `audio_folder`、`version`、`sample_rate`、`f0_method`、`experiment`、`overwrite` | `dataset` |
| **RVC Train Audio List** | `audios`（autogrow 音频接口）、`version`、`sample_rate`、`f0_method`、`experiment` | `dataset` |
| **RVC Train Model** | `dataset`、`model_name`、`epochs`、`batch_size`、`save_every_epoch`、`learning_rate`、`pretrained`、`resume` | `log`、`model_path` |

`dataset` 是自定义的 `RVC_DATASET` 类型，携带准备好的实验目录、采样率、版本和切片记录。

- **version / sample_rate** —— `v1` 或 `v2` 特征，`32k` / `40k` / `48k` 输出。`40k` 仅 `v1` 可用。该组合从 `rvc_train/configs` 中选取打包配置，与 RVC-Project 的`configs/` 完全一致。
- **f0_method** —— `rmvpe`（默认）或 `pm`（需要 `praat-parselmouth`）。
- **RVC Train Dataset** 首次遍历文件夹，之后的运行复用已有的切片、特征和频谱缓存；已存在切片的音频源会被跳过。勾选 `overwrite` 可重建。
- **RVC Train Audio List** 把连接的音频写到磁盘并始终重建，所以接口里看到的就是数据集。
- **pretrained** —— 预训练模型，用于微调（**选择规则见「pretrained 参数说明」**）。不选则从随机权重开始训练，小数据集训不出可用音色。同时接受 `{"weight": ...}` 和 `{"model": ...}` 两种检查点格式，单说话人 `emb_g` 嵌入会自动扩展到 109 个说话人。
- **resume** —— 从实验 `checkpoints` 文件夹中最新的 `G_*.pth` / `D_*.pth` 继续训练，并恢复优化器和轮次计数。

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

推理代码完全内置（`rvc_inference/`）：高通滤波、长音频自动切片、f0 提取（RMVPE/FCPE）、ContentVec 特征、faiss/npy 检索混合、保护度加权、合成器推理。`fp16 autocast` 在任何 CUDA GPU 上启用，无 CUDA 时回退 fp32。`faiss-cpu` 为可选依赖（读取 `.index`）；没有时 `.npy` 兜底可用。

注意：输出与 RVC-Project 上游一致，是随机采样的（相同输入两次运行有约 0.4-0.7 的幅值波动）——这是模型的后验采样本性，不是 bug。

## pretrained 参数说明

* 可以选择 `models/rvc` 目录下的任意 `*.pth` 模型（包括你之前训练导出的音色模型），**用它等于"从某个已有音色继续微调"**；
* **练全新声音时选官方 `f0G*.pth` 底座即可**。

### 必须与 dataset 的版本 + 采样率匹配

以你数据集节点（RVC Train Dataset / Audio List）选的 version/sample_rate 为准：

| dataset 配置 | pretrained 应选 |
| --- | --- |
| v2 + 48k | `f0G48k.pth`（来自 `pretrained_v2/`） |
| v2 + 32k | `f0G32k.pth`（来自 `pretrained_v2/`） |
| v1 + 40k | `f0G40k.pth`（来自 `pretrained/`） |
| v1 + 32k / 48k | `f0G32k.pth` / `f0G48k.pth`（来自 `pretrained/`） |

两个注意点：

- **务必选 f0 变体**（`f0G*.pth`），不是无 f0 的 `G*.pth` —— 本插件只训练 f0 引导模型，[`_load_pretrained`](rvc_train/train.py#L289-L290) 遇到非 f0 文件会直接报错。
- **版本不匹配会报错**（[L287](rvc_train/train.py#L287) 校验 `version`），但官方 `f0G*.pth` 不带采样率元数据，**采样率匹配靠你自己按文件名保证**——选错不会报错，只会训出坏模型。v2 没有 40k 配置，所以 v2 只配 32k/48k 两个文件。

## 文件

所有产物写在 `<ComfyUI output>/rvc_train/<experiment>/<version>_<采样率>/` 下：

```
0_gt_wavs/       模型采样率下的切片音频（线性频谱的来源）
1_16k_wavs/      同样的切片重采样到 16 kHz
2a_f0/           量化 f0 bins（1..255），用于音高嵌入
2b-f0nsf/        NSF 声源用的 Hz 值 f0
3_feature256/    HuBERT 第 8 层 + 最终投影特征（v1）
3_feature768/    HuBERT 第 11 层特征（v2）
checkpoints/     G_<epoch>.pth 和 D_<epoch>.pth，含优化器状态
```

`G_<epoch>.pth` 保存生成器（含仅训练用的 `enc_q` 后验编码器）及其 AdamW 状态，`D_<epoch>.pth` 保存周期判别器及其 AdamW 状态。它们只用于恢复训练——可推理的模型是导出的音色文件。每对都很大（48k 下约 0.4 GB + 0.8 GB），保存新检查点时会自动删除更早的，只保留最近 5 对。

导出的音色模型写入第一个 `rvc` 模型文件夹（通常是 `models/rvc/<model_name>.pth`），训练完即可在 ComfyUI-RVC 的 **Load RVC Voice** 节点中直接看到。

同一次运行还会用数据集的 HuBERT 特征构建检索索引：`<model_name>.npy`（原始向量，供ComfyUI-RVC 的 **Load RVC Index** 读取）和 `<model_name>.index`（FAISS `IVF<n>,Flat`、`nprobe 1`，RVC-Project 等工具期望的格式）。没有 faiss 时只写 `.npy`，日志会说明。超过 20 万帧的特征集会按 RVC-Project 的做法聚类到 1 万个中心，保证文件可加载。

## 依赖

- ComfyUI-RVC 使用的转换后检查点：`models/rvc/hubert_base.safetensors` 和`models/rvc/rmvpe.safetensors`（按 `rvc_hubert.format_version` /`rvc_rmvpe.format_version` 元数据识别，其他文件名按关键词匹配）。
- `torch`、`librosa`、`scipy`、`soundfile` —— 使用pip install挨个安装或`pip install -r requirements.txt`统一安装。
- 可选：`faiss` 和 `scikit-learn` 用于检索索引（`.index`，以及超大特征集的 1 万中心聚类）。没有它们训练照常进行，`.npy` 向量照常写出。
- 强烈建议 CUDA GPU。Ampere 及更新的 GPU（compute capability ≥ 8.0）使用 fp16 autocast，其余用 fp32。

## 快速上手

1. **RVC Train Dataset**：填入音频文件夹路径，选 `v2` + `48k`，f0 用 `rmvpe`，`experiment` 填一个名字，执行后完成切片、音高与 HuBERT 特征提取。也可以改用 **RVC Train Audio List**，把音频直接连到 autogrow 接口上。
2. **RVC Train Model**：接上 `dataset`，设置 `epochs`、`batch_size`（默认 4）、`save_every_epoch`；`pretrained` 按「pretrained 参数说明」选择（如 v2/48k 用 `f0G48k.pth`）——不选则从随机权重开始训练，小数据集训不出可用音色；执行后输出训练日志和模型路径。
3. 训练产物：`output/rvc_train/<experiment>/<version>_<采样率>/`（切片、特征、检查点），导出的音色模型写入 `models/rvc/<model_name>.pth`，可直接用 ComfyUI-RVC 的**Load RVC Voice** 加载。
4. 中断后把 `resume` 设为 true 可继续训练；改采样率或版本请改 `experiment` 名字，或勾选 `overwrite` 重建。

## 说明

- 训练在 ComfyUI 进程内运行并阻塞队列；保持窗口打开，用常规取消按钮停止。已写出的检查点会保留，`resume` 可从中继续。
- 只训练 f0 引导模型（`f0 = 1`），即当前 RVC 音色使用的形式。
- 训练日志以文本返回，同时镜像到 ComfyUI 控制台。
- **RVC Train Model** 用已完成轮次 / 总轮次驱动 ComfyUI 进度条，恢复的运行从上次停止处开始。
