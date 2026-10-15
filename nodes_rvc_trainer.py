"""ComfyUI nodes that prepare an RVC dataset and train a voice model."""

import logging
import os
import re

import folder_paths
import numpy as np
from comfy_api.latest import io
from scipy.io import wavfile

import comfy.model_management
import comfy.utils

from .rvc_train import hubert, index, preprocess, rmvpe
from .rvc_train.train import TrainSettings, TrainingLog, train


LOGGER = logging.getLogger("RVC-Trainer")

CATEGORY = "RVC/training"
RVC_DATASET = io.Custom("RVC_DATASET")

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".aiff", ".wma")
HUBERT_FILES = ("hubert_base.safetensors",)
RMVPE_FILES = ("rmvpe.safetensors",)
VERSION_OPTIONS = ["v2", "v1"]
SAMPLE_RATE_OPTIONS = ["48k", "40k", "32k"]
F0_METHOD_OPTIONS = ["rmvpe", "pm"]
NONE_OPTION = "(none)"


def _safe_component(value, label):
    name = re.sub(r"[^0-9A-Za-z_.\-]+", "_", os.path.basename(value.strip().strip('"'))).strip("._-")
    if not name:
        raise ValueError("%s must contain letters or digits" % label)
    return name


def _contained(base, name):
    path = os.path.abspath(os.path.join(base, name))
    if os.path.commonpath([os.path.abspath(base), path]) != os.path.abspath(base):
        raise ValueError("Refusing to write outside %s: %s" % (base, path))
    return path


def _dataset_root():
    root = os.path.join(folder_paths.get_output_directory(), "rvc_train")
    os.makedirs(root, exist_ok=True)
    return root


def _voice_output_dir():
    folders = folder_paths.get_folder_paths("rvc")
    directory = folders[0] if folders else os.path.join(folder_paths.models_dir, "rvc")
    os.makedirs(directory, exist_ok=True)
    return directory


def _rvc_model_file(kind):
    """Locate a converted hubert/rmvpe checkpoint in ComfyUI's rvc model folders."""
    wanted_names = HUBERT_FILES if kind == "hubert" else RMVPE_FILES
    files = folder_paths.get_filename_list("rvc")
    for wanted in wanted_names:
        for candidate in files:
            if os.path.basename(candidate).lower() == wanted:
                return folder_paths.get_full_path_or_raise("rvc", candidate)
    for candidate in files:
        if kind in os.path.basename(candidate).lower() and candidate.lower().endswith(".safetensors"):
            return folder_paths.get_full_path_or_raise("rvc", candidate)
    raise ValueError(
        "No converted %s checkpoint found. Expected one of %s in %s"
        % (kind, ", ".join(wanted_names), ", ".join(folder_paths.get_folder_paths("rvc")))
    )


def _pretrained_options():
    options = [NONE_OPTION]
    options.extend(sorted(name for name in folder_paths.get_filename_list("rvc") if name.lower().endswith(".pth")))
    return options


def _folder_sources(audio_folder):
    directory = os.path.abspath(os.path.expanduser(audio_folder.strip().strip('"')))
    if not os.path.isdir(directory):
        raise ValueError("Audio folder not found: %s" % directory)
    names = sorted(
        name
        for name in os.listdir(directory)
        if name.lower().endswith(AUDIO_EXTENSIONS) and os.path.isfile(os.path.join(directory, name))
    )
    if not names:
        raise ValueError("No audio files (%s) found in %s" % (", ".join(AUDIO_EXTENSIONS), directory))
    return [(preprocess.source_key(index, name), os.path.join(directory, name)) for index, name in enumerate(names)]


def _clip_sources(clips, raw_dir):
    """Write connected AUDIO inputs to wav files and return (key, path) sources."""
    os.makedirs(raw_dir, exist_ok=True)
    for name in os.listdir(raw_dir):
        os.remove(os.path.join(raw_dir, name))

    sources = []
    for index, audio in enumerate(clips):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        for batch in range(waveform.shape[0]):
            key = "%s_%s_clip" % (index, batch)
            path = os.path.join(raw_dir, key + ".wav")
            wavfile.write(path, sample_rate, waveform[batch].detach().float().cpu().numpy().astype(np.float32))
            sources.append((key, path))
    return sources


def _prepare(sources, root, options):
    os.makedirs(root, exist_ok=True)
    device = comfy.model_management.get_torch_device()
    LOGGER.info(
        "Preparing %s audio file(s) | version %s | sample rate %s | f0 %s",
        len(sources),
        options.version,
        options.sample_rate,
        options.f0_method,
    )
    pitch_model = rmvpe.load_rmvpe(_rvc_model_file("rmvpe"), device) if options.f0_method == "rmvpe" else None
    hubert_model = hubert.load_hubert(_rvc_model_file("hubert"), device)
    try:
        return preprocess.prepare_dataset(sources, root, options, pitch_model, hubert_model, LOGGER.info)
    finally:
        del pitch_model, hubert_model
        comfy.model_management.soft_empty_cache()


class RVCTrainDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCTrainDataset",
            display_name="RVC Train Dataset",
            category=CATEGORY,
            description="Slice a folder of audio into training clips and extract f0 and Hubert features.",
            inputs=[
                io.String.Input("audio_folder", default="", tooltip="Folder containing the source audio files."),
                io.Combo.Input("version", options=VERSION_OPTIONS, default="v2", tooltip="RVC feature and model version."),
                io.Combo.Input("sample_rate", options=SAMPLE_RATE_OPTIONS, default="48k", tooltip="Model sample rate. 40k is available for v1 only."),
                io.Combo.Input("f0_method", options=F0_METHOD_OPTIONS, default="rmvpe", tooltip="Pitch estimator used to build f0 features."),
                io.String.Input("experiment", default="my_voice", tooltip="Folder that holds this training set."),
                io.Boolean.Input("overwrite", default=False, tooltip="Delete existing slices and features for this version and sample rate first."),
            ],
            outputs=[RVC_DATASET.Output(display_name="dataset")],
        )

    @classmethod
    def execute(cls, audio_folder, version, sample_rate, f0_method, experiment, overwrite):
        sources = _folder_sources(audio_folder)
        options = preprocess.DatasetOptions(sample_rate, version, f0_method, overwrite)
        root = _contained(_dataset_root(), _safe_component(experiment, "experiment"))
        return io.NodeOutput(_prepare(sources, root, options))


class RVCTrainAudioList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.Audio.Input("audio"), prefix="audio", min=0, max=100)
        return io.Schema(
            node_id="RVCTrainAudioList",
            display_name="RVC Train Audio List",
            category=CATEGORY,
            description="Build a training set from individually connected audio clips.",
            inputs=[
                io.Autogrow.Input("audios", template=template, tooltip="Clips to train on. Batches are split into separate clips."),
                io.Combo.Input("version", options=VERSION_OPTIONS, default="v2", tooltip="RVC feature and model version."),
                io.Combo.Input("sample_rate", options=SAMPLE_RATE_OPTIONS, default="48k", tooltip="Model sample rate. 40k is available for v1 only."),
                io.Combo.Input("f0_method", options=F0_METHOD_OPTIONS, default="rmvpe", tooltip="Pitch estimator used to build f0 features."),
                io.String.Input("experiment", default="my_voice", tooltip="Folder that holds this training set."),
            ],
            outputs=[RVC_DATASET.Output(display_name="dataset")],
        )

    @classmethod
    def execute(cls, audios, version, sample_rate, f0_method, experiment):
        clips = [audios[name] for name in sorted(audios, key=lambda name: int(re.search(r"(\d+)$", name).group(1))) if audios[name] is not None]
        if not clips:
            raise ValueError("Connect at least one audio clip")
        root = _contained(_dataset_root(), _safe_component(experiment, "experiment"))
        options = preprocess.DatasetOptions(sample_rate, version, f0_method, True)
        sources = _clip_sources(clips, os.path.join(root, "%s_%s" % (version, sample_rate), "raw_wavs"))
        return io.NodeOutput(_prepare(sources, root, options))


class RVCTrainModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RVCTrainModel",
            display_name="RVC Train Model",
            category=CATEGORY,
            description="Train an RVC voice model from a prepared dataset and export it to the rvc models folder.",
            inputs=[
                RVC_DATASET.Input("dataset", tooltip="Dataset produced by an RVC Train Dataset or RVC Train Audio List node."),
                io.String.Input("model_name", default="my_voice", tooltip="File name of the exported voice model."),
                io.Int.Input("epochs", default=200, min=1, max=10000, tooltip="Total number of epochs to train."),
                io.Int.Input("batch_size", default=4, min=1, max=64, tooltip="Training batch size."),
                io.Int.Input("save_every_epoch", default=10, min=1, max=1000, tooltip="Save a checkpoint and export the voice model every N epochs."),
                io.Float.Input("learning_rate", default=1e-4, min=1e-6, max=1e-2, step=1e-6, tooltip="AdamW learning rate."),
                io.Combo.Input("pretrained", options=_pretrained_options(), default=NONE_OPTION, tooltip="Pretrained RVC .pth to fine-tune from; must match version and sample rate (e.g. f0G48k.pth). Without one the generator trains from random weights and produces unusable audio."),
                io.Boolean.Input("resume", default=False, tooltip="Continue from the newest checkpoint in the dataset folder."),
            ],
            outputs=[
                io.String.Output(display_name="log"),
                io.String.Output(display_name="model_path"),
            ],
        )

    @classmethod
    def execute(cls, dataset, model_name, epochs, batch_size, save_every_epoch, learning_rate, pretrained, resume):
        name = _safe_component(model_name, "model name")
        voice_dir = _voice_output_dir()
        weights_path = _contained(voice_dir, name + ".pth")
        vectors_path = _contained(voice_dir, name + ".npy")
        index_path = _contained(voice_dir, name + ".index")
        settings = TrainSettings(
            pretrained_path=None if pretrained == NONE_OPTION else folder_paths.get_full_path_or_raise("rvc", pretrained),
            epochs=epochs,
            batch_size=batch_size,
            save_every_epoch=save_every_epoch,
            learning_rate=learning_rate,
            checkpoint_dir=os.path.join(dataset.exp_dir, "checkpoints"),
            weights_path=weights_path,
            resume=resume,
        )
        log = TrainingLog()
        log(
            "数据集：%s | 切片：%s | 轮次：%s | 批大小：%s | 学习率：%s | 预训练：%s | 恢复：%s"
            % (dataset.exp_dir, len(dataset.records), epochs, batch_size, learning_rate, pretrained, resume)
        )
        progress = comfy.utils.ProgressBar(epochs)
        train(dataset, settings, log, progress.update_absolute)
        progress.update_absolute(epochs)
        log("音色模型：%s" % weights_path)
        index.build_index(dataset, vectors_path, index_path, log)
        return io.NodeOutput(log.text(), weights_path)


NODE_CLASS_MAPPINGS = {
    "RVCTrainDataset": RVCTrainDataset,
    "RVCTrainAudioList": RVCTrainAudioList,
    "RVCTrainModel": RVCTrainModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVCTrainDataset": "RVC Train Dataset",
    "RVCTrainAudioList": "RVC Train Audio List",
    "RVCTrainModel": "RVC Train Model",
}
