"""Model configs and training precision policy for RVC training."""

import json
import os

import torch

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

MODEL_CONFIG_FILES = (
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/32k.json",
    "v2/48k.json",
)


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            setattr(self, key, value)

    def __repr__(self):
        return self.__dict__.__repr__()


def model_config_files(version):
    """Bundled sample rate configs available for a model version, e.g. ["32k", "48k"]."""
    return [path.split("/")[1] for path in MODEL_CONFIG_FILES if path.split("/")[0] == version]


def load_config(version, sample_rate):
    """Load the bundled config for a version/sample rate pair ("v2", "48k")."""
    name = "%s/%s.json" % (version, sample_rate)
    if name not in MODEL_CONFIG_FILES:
        raise ValueError("Unsupported RVC version/sample rate: %s/%s" % (version, sample_rate))
    with open(os.path.join(CONFIGS_DIR, name), "r", encoding="utf8") as config_file:
        return HParams(**json.load(config_file))


def generator_arguments(hps):
    """Constructor arguments for the generator matching this config."""
    return {
        "spec_channels": hps.data.filter_length // 2 + 1,
        "segment_size": hps.train.segment_size // hps.data.hop_length,
        "sr": hps.data.sampling_rate,
        "inter_channels": hps.model.inter_channels,
        "hidden_channels": hps.model.hidden_channels,
        "filter_channels": hps.model.filter_channels,
        "n_heads": hps.model.n_heads,
        "n_layers": hps.model.n_layers,
        "kernel_size": hps.model.kernel_size,
        "p_dropout": hps.model.p_dropout,
        "resblock": hps.model.resblock,
        "resblock_kernel_sizes": hps.model.resblock_kernel_sizes,
        "resblock_dilation_sizes": hps.model.resblock_dilation_sizes,
        "upsample_rates": hps.model.upsample_rates,
        "upsample_initial_channel": hps.model.upsample_initial_channel,
        "upsample_kernel_sizes": hps.model.upsample_kernel_sizes,
        "spk_embed_dim": hps.model.spk_embed_dim,
        "gin_channels": hps.model.gin_channels,
    }


def voice_config(hps):
    """The 18 element config list stored inside an exported RVC voice model."""
    return [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]


def training_dtype(device):
    """fp16 autocast on Ampere and newer GPUs, fp32 everywhere else."""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.float16 if major >= 8 else torch.float32
