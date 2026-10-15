"""RVC training loop: adversarial fine-tuning of the generator and voice export."""

import datetime
import glob
import logging
import os
import pickle
import re
import time
import zipfile
from collections import OrderedDict

import torch
from torch.nn import functional as F
import torch.utils.data

import comfy.model_management

from . import config, features, losses, models
from .dataset import AudioFeatureDataset, BucketSampler, TextAudioCollate


BOUNDARIES = [100, 200, 300, 400, 500, 600, 700, 800, 900]
KEEP_CHECKPOINTS = 5
LOGGER = logging.getLogger("RVC-Trainer")


class TrainingLog:
    """Collects the lines a training node returns while forwarding them to the console."""

    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message))
        LOGGER.info(message)

    def text(self):
        return "\n".join(self.lines)


class TrainSettings:
    def __init__(self, pretrained_path, epochs, batch_size, save_every_epoch, learning_rate, checkpoint_dir, weights_path, resume):
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_every_epoch = save_every_epoch
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.weights_path = weights_path
        self.resume = resume


def train(data, settings, log, progress):
    """progress(completed_epochs, total_epochs) reports training progress to the caller."""
    # ComfyUI executes nodes under torch.inference_mode, and parameters created
    # there stay inference tensors that an optimizer cannot update.
    with torch.inference_mode(False):
        device = comfy.model_management.get_torch_device()
        dtype = config.training_dtype(device)
        is_half = dtype == torch.float16
        hps = config.load_config(data.version, data.sample_rate)
        log("设备：%s | 精度：%s | 版本：%s | 采样率：%s" % (device, dtype, data.version, hps.data.sampling_rate))
        torch.manual_seed(hps.train.seed)

        mel = features.MelProcessor(
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            hps.data.hop_length,
            hps.data.win_length,
            device="cpu",
        )
        dataset = AudioFeatureDataset(data.records, mel, hps.data.sampling_rate, hps.data.hop_length, hps.data.filter_length, hps.data.win_length)
        sampler = BucketSampler(dataset, settings.batch_size, BOUNDARIES)
        if len(sampler) == 0:
            raise RuntimeError("No training batches were formed; every slice is outside the sampler length buckets")
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=TextAudioCollate(),
        )
        log("训练切片：%s | 每轮批次数：%s" % (len(dataset), len(sampler)))

        net_g = models.pick_synthesizer(data.version)(**config.generator_arguments(hps)).to(device)
        net_d = models.pick_discriminator(data.version)(hps.model.use_spectral_norm).to(device)
        optim_g = torch.optim.AdamW(net_g.parameters(), settings.learning_rate, betas=tuple(hps.train.betas), eps=hps.train.eps)
        optim_d = torch.optim.AdamW(net_d.parameters(), settings.learning_rate, betas=tuple(hps.train.betas), eps=hps.train.eps)

        start_epoch = 1
        global_step = 0
        if settings.resume:
            start_epoch, global_step = _resume_checkpoints(net_g, net_d, optim_g, optim_d, settings, len(loader), log)
        if start_epoch == 1 and settings.pretrained_path:
            log("预训练生成器：%s" % os.path.basename(settings.pretrained_path))
            _load_pretrained(net_g, settings.pretrained_path, hps, data.version)
        elif start_epoch == 1:
            log("警告：未提供预训练模型，从随机权重开始训练；小数据集通常得不到可用音色")
        if start_epoch > settings.epochs:
            log("检查点已达到 %s 轮，无需继续训练" % (start_epoch - 1))
            return

        scaler = torch.amp.GradScaler(device.type, enabled=is_half)
        log("开始训练：第 %s 至 %s 轮" % (start_epoch, settings.epochs))
        progress(start_epoch - 1, settings.epochs)

        try:
            for epoch in range(start_epoch, settings.epochs + 1):
                _set_learning_rate(optim_g, optim_d, settings.learning_rate * hps.train.lr_decay ** (epoch - 1))
                global_step = _train_epoch(
                    net_g, net_d, optim_g, optim_d, scaler, loader, sampler, mel, hps, device, is_half, global_step, epoch, log
                )
                progress(epoch, settings.epochs)
                if epoch % settings.save_every_epoch == 0 or epoch == settings.epochs:
                    _save_checkpoints(net_g, net_d, optim_g, optim_d, hps, settings, epoch, log)
                    _export_voice(net_g, hps, data.version, epoch, settings, log)
        finally:
            del net_g, net_d, optim_g, optim_d, scaler, loader, dataset
            comfy.model_management.soft_empty_cache()


def _set_learning_rate(optim_g, optim_d, learning_rate):
    for optimizer in (optim_g, optim_d):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def _train_epoch(net_g, net_d, optim_g, optim_d, scaler, loader, sampler, mel, hps, device, is_half, global_step, epoch, log):
    net_g.train()
    net_d.train()
    sampler.set_epoch(epoch)
    segment_frames = hps.train.segment_size // hps.data.hop_length
    epoch_start = time.time()

    for batch_idx, info in enumerate(loader):
        comfy.model_management.throw_exception_if_processing_interrupted()
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _wave_lengths, sid = [
            tensor.to(device, non_blocking=True) for tensor in info
        ]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_half):
            y_hat, ids_slice, _x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
            )
            spec_mel = mel.spec_to_mel(spec)
            y_mel = models.slice_segments(spec_mel, ids_slice, segment_frames)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                y_hat_mel = mel.mel(y_hat.float().squeeze(1))
            if is_half:
                y_hat_mel = y_hat_mel.half()
            wave = models.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with torch.amp.autocast(device_type=device.type, enabled=False):
                loss_disc, _, _ = losses.discriminator_loss(y_d_hat_r, y_d_hat_g)

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        models.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_half):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                loss_fm = losses.feature_loss(fmap_r, fmap_g)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_gen, _ = losses.generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        models.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps.train.log_interval == 0 or batch_idx + 1 == len(loader):
            log(
                "轮次 %s [%.0f%%] 步 %s | lr %.2e | d=%.3f g=%.3f fm=%.3f mel=%.3f kl=%.3f"
                % (
                    epoch,
                    100.0 * (batch_idx + 1) / len(loader),
                    global_step,
                    optim_g.param_groups[0]["lr"],
                    loss_disc.item(),
                    loss_gen.item(),
                    loss_fm.item(),
                    min(loss_mel.item(), 75),
                    min(loss_kl.item(), 9),
                )
            )
        global_step += 1

    log("轮次 %s 完成，用时 %s" % (epoch, datetime.timedelta(seconds=int(time.time() - epoch_start))))
    return global_step


def _save_checkpoints(net_g, net_d, optim_g, optim_d, hps, settings, epoch, log):
    os.makedirs(settings.checkpoint_dir, exist_ok=True)
    for name, network, optimizer in (("G", net_g, optim_g), ("D", net_d, optim_d)):
        torch.save(
            {
                "model": network.state_dict(),
                "iteration": epoch,
                "optimizer": optimizer.state_dict(),
                "learning_rate": settings.learning_rate,
            },
            os.path.join(settings.checkpoint_dir, "%s_%s.pth" % (name, epoch)),
        )
    _prune_checkpoints(settings.checkpoint_dir, log)
    log("已保存检查点：%s/G_%s.pth、D_%s.pth" % (settings.checkpoint_dir, epoch, epoch))


def _prune_checkpoints(directory, log):
    epochs = sorted(int(re.search(r"(\d+)", os.path.basename(path)).group(1)) for path in glob.glob(os.path.join(directory, "G_*.pth")))
    pruned = 0
    for old_epoch in epochs[:-KEEP_CHECKPOINTS]:
        for name in ("G", "D"):
            stale = os.path.join(directory, "%s_%s.pth" % (name, old_epoch))
            if os.path.exists(stale):
                os.remove(stale)
        pruned += 1
    if pruned:
        log("已删除 %s 轮旧检查点，保留最近 %s 轮" % (pruned, KEEP_CHECKPOINTS))


def _export_voice(net_g, hps, version, epoch, settings, log):
    weights = OrderedDict()
    for key, value in net_g.state_dict().items():
        if "enc_q" in key:
            continue
        weights[key] = value.half() if value.is_floating_point() else value
    payload = OrderedDict()
    payload["weight"] = weights
    payload["config"] = config.voice_config(hps)
    payload["info"] = "%sepoch" % epoch
    payload["sr"] = hps.data.sampling_rate
    payload["f0"] = 1
    payload["version"] = version
    os.makedirs(os.path.dirname(settings.weights_path), exist_ok=True)
    torch.save(payload, settings.weights_path)
    log("已导出音色模型：%s" % settings.weights_path)


def _resume_checkpoints(net_g, net_d, optim_g, optim_d, settings, batches, log):
    g_path = _latest_checkpoint(settings.checkpoint_dir, "G_*.pth")
    d_path = _latest_checkpoint(settings.checkpoint_dir, "D_*.pth")
    if g_path is None or d_path is None:
        log("未找到可恢复的检查点，从头开始训练")
        return 1, 0
    g_state = torch.load(g_path, map_location="cpu", weights_only=True)
    d_state = torch.load(d_path, map_location="cpu", weights_only=True)
    net_g.load_state_dict(g_state["model"], strict=True)
    net_d.load_state_dict(d_state["model"], strict=True)
    optim_g.load_state_dict(g_state["optimizer"])
    optim_d.load_state_dict(d_state["optimizer"])
    epoch = int(g_state["iteration"])
    log("已恢复检查点 %s（第 %s 轮）" % (os.path.basename(g_path), epoch))
    return epoch + 1, epoch * batches


def _latest_checkpoint(directory, pattern):
    paths = glob.glob(os.path.join(directory, pattern))
    if not paths:
        return None
    paths.sort(key=lambda path: int(re.search(r"(\d+)", os.path.basename(path)).group(1)))
    return paths[-1]


def _load_pretrained(net_g, path, hps, version):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, zipfile.BadZipFile, EOFError):
        raise ValueError("预训练文件 %s 不是有效的 PyTorch .pth 检查点（可能下载不完整或保存成了网页）；请重新下载完整文件" % os.path.basename(path))
    state = checkpoint.get("weight")
    if state is None:
        state = checkpoint.get("model")
    if state is None:
        raise ValueError("%s is not an RVC voice checkpoint" % path)

    if checkpoint.get("version") not in (None, version):
        raise ValueError("pretrained %s is %s but the model is %s" % (os.path.basename(path), checkpoint["version"], version))
    if not checkpoint.get("f0", 1):
        raise ValueError("pretrained %s was trained without f0, which this trainer does not support" % os.path.basename(path))
    pretrained_sr = checkpoint.get("sr", checkpoint.get("config", [None] * 18)[-1])
    if pretrained_sr is not None and int(models.sr2sr.get(str(pretrained_sr), pretrained_sr)) != hps.data.sampling_rate:
        raise ValueError(
            "pretrained %s is %s Hz but the model is %s Hz" % (os.path.basename(path), pretrained_sr, hps.data.sampling_rate)
        )

    target = net_g.state_dict()
    loaded = {}
    for key, value in target.items():
        source = state.get(key)
        if source is None:
            continue
        if source.shape == value.shape:
            loaded[key] = source
        elif key == "emb_g.weight" and source.dim() == value.dim() and source.shape[1:] == value.shape[1:]:
            merged = value.clone()
            rows = min(source.shape[0], value.shape[0])
            merged[:rows].copy_(source[:rows])
            loaded[key] = merged
        else:
            raise ValueError("pretrained %s has %s with shape %s but the model needs %s" % (os.path.basename(path), key, tuple(source.shape), tuple(value.shape)))
    missing = sorted(key for key in target if key not in loaded and not key.startswith("enc_q."))
    if missing:
        raise ValueError("pretrained %s does not provide %s" % (os.path.basename(path), ", ".join(missing[:5])))
    net_g.load_state_dict(loaded, strict=False)
