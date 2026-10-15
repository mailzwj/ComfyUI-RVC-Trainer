"""Experiment preparation: highpass, silence slicing, normalisation and features."""

import os
import re
import shutil
import traceback

import librosa
import numpy as np
import soundfile
from scipy import signal
from scipy.io import wavfile

from . import config, features
from .dataset import RVCDataset
from .slicer import Slicer


PER = 3.7
OVERLAP = 0.3
TAIL = PER + OVERLAP
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGHPASS_HZ = 48

GT_DIR = "0_gt_wavs"
WAV16_DIR = "1_16k_wavs"
F0_DIR = "2a_f0"
F0NSF_DIR = "2b-f0nsf"
FEATURE_V1_DIR = "3_feature256"
FEATURE_V2_DIR = "3_feature768"


class DatasetOptions:
    def __init__(self, sample_rate, version, f0_method, overwrite):
        self.sample_rate = sample_rate
        self.version = version
        self.f0_method = f0_method
        self.overwrite = overwrite


def experiment_path(root, options):
    return os.path.join(root, "%s_%s" % (options.version, options.sample_rate))


def source_key(index, name):
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.splitext(os.path.basename(name))[0]).strip("_")[:48]
    return "%s_%s" % (index, stem) if stem else str(index)


def prepare_dataset(sources, root, options, pitch_model, hubert_model, log):
    """Slice and featurize sources into a training ready experiment directory."""
    hps = config.load_config(options.version, options.sample_rate)
    sample_rate = hps.data.sampling_rate
    exp_dir = experiment_path(root, options)
    feature_dir = FEATURE_V1_DIR if options.version == "v1" else FEATURE_V2_DIR
    generated = (GT_DIR, WAV16_DIR, F0_DIR, F0NSF_DIR, feature_dir)

    if options.overwrite:
        for name in generated:
            shutil.rmtree(os.path.join(exp_dir, name), ignore_errors=True)
    for name in generated:
        os.makedirs(os.path.join(exp_dir, name), exist_ok=True)
    log("数据集目录：%s" % exp_dir)

    gt_dir = os.path.join(exp_dir, GT_DIR)
    wav16_dir = os.path.join(exp_dir, WAV16_DIR)
    f0_dir = os.path.join(exp_dir, F0_DIR)
    f0nsf_dir = os.path.join(exp_dir, F0NSF_DIR)
    features_dir = os.path.join(exp_dir, feature_dir)

    _slice_sources(sources, gt_dir, wav16_dir, sample_rate, options.overwrite, log)
    _extract_features(wav16_dir, f0_dir, f0nsf_dir, features_dir, options, pitch_model, hubert_model, log)

    records = []
    for name in sorted(os.listdir(wav16_dir)):
        if not name.endswith(".wav"):
            continue
        paths = (
            os.path.join(gt_dir, name),
            os.path.join(features_dir, os.path.splitext(name)[0] + ".npy"),
            os.path.join(f0_dir, name + ".npy"),
            os.path.join(f0nsf_dir, name + ".npy"),
        )
        if all(os.path.isfile(path) for path in paths):
            records.append([paths[0], paths[1], paths[2], paths[3], 0])
    if not records:
        raise RuntimeError("No usable training slices were produced in %s" % exp_dir)
    log("可用训练切片：%s" % len(records))
    return RVCDataset(exp_dir, options.sample_rate, options.version, records)


def _slice_sources(sources, gt_dir, wav16_dir, sample_rate, overwrite, log):
    high_b, high_a = signal.butter(N=5, Wn=HIGHPASS_HZ, btype="high", fs=sample_rate)
    slicer = Slicer(sr=sample_rate, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    step = int(sample_rate * (PER - OVERLAP))
    per_samples = int(PER * sample_rate)
    tail_samples = TAIL * sample_rate

    for index, (key, path) in enumerate(sources):
        if not overwrite and any(name.startswith(key + "_") for name in os.listdir(gt_dir)):
            log("[切分] 已存在，跳过：%s" % os.path.basename(path))
            continue
        try:
            audio = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)[0]
            audio = signal.lfilter(high_b, high_a, audio)

            written = 0
            for chunk in slicer.slice(audio):
                pieces = []
                start = 0
                while len(chunk[start:]) > tail_samples:
                    pieces.append(chunk[start : start + per_samples])
                    start += step
                pieces.append(chunk[start:])
                for piece in pieces:
                    if _write_slice(piece, gt_dir, wav16_dir, key, written, sample_rate):
                        written += 1
            log("[切分] %s/%s | %s | %s 个切片" % (index + 1, len(sources), os.path.basename(path), written))
        except Exception:
            log("[切分][失败] %s\n%s" % (path, traceback.format_exc()))


def _write_slice(audio, gt_dir, wav16_dir, key, index, sample_rate):
    peak = float(np.abs(audio).max())
    if not np.isfinite(peak) or peak <= 0 or peak > 2.5:
        return False
    audio = (audio / peak * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio
    wavfile.write(os.path.join(gt_dir, "%s_%s.wav" % (key, index)), sample_rate, audio.astype(np.float32))
    audio16 = librosa.resample(audio, orig_sr=sample_rate, target_sr=features.AUDIO_SR).astype(np.float32)
    wavfile.write(os.path.join(wav16_dir, "%s_%s.wav" % (key, index)), features.AUDIO_SR, audio16)
    return True


def _extract_features(wav16_dir, f0_dir, f0nsf_dir, features_dir, options, pitch_model, hubert_model, log):
    names = [name for name in sorted(os.listdir(wav16_dir)) if name.endswith(".wav")]
    total = len(names)
    done = 0
    skipped = 0
    for index, name in enumerate(names):
        f0_path = os.path.join(f0_dir, name + ".npy")
        f0nsf_path = os.path.join(f0nsf_dir, name + ".npy")
        feature_path = os.path.join(features_dir, os.path.splitext(name)[0] + ".npy")
        need_f0 = not (os.path.isfile(f0_path) and os.path.isfile(f0nsf_path))
        need_feature = not os.path.isfile(feature_path)
        if not need_f0 and not need_feature:
            skipped += 1
            continue
        try:
            audio16 = soundfile.read(os.path.join(wav16_dir, name), dtype="float32")[0]
            if need_f0:
                pitch = features.fill_unvoiced(np.asarray(features.extract_f0(audio16, options.f0_method, pitch_model), dtype=np.float32))
                if pitch is None:
                    log("[特征] 音高全为0，跳过：%s" % name)
                    continue
                np.save(f0nsf_path, pitch, allow_pickle=False)
                np.save(f0_path, features.coarse_f0(pitch), allow_pickle=False)
            if need_feature:
                feats = features.extract_hubert(audio16, options.version, hubert_model)
                if np.isnan(feats).sum() != 0:
                    log("[特征][失败] %s 包含 NaN" % name)
                    continue
                np.save(feature_path, feats, allow_pickle=False)
            done += 1
            if done % 20 == 0 or index + 1 == total:
                log("[特征] 进度：%s/%s | 已跳过：%s" % (index + 1, total, skipped))
        except Exception:
            log("[特征][失败] %s\n%s" % (name, traceback.format_exc()))
    log("[特征] 完成：新处理 %s | 已存在 %s" % (done, skipped))
