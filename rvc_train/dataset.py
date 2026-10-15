"""Dataset, collate function and length bucketing for RVC training."""

import os
import traceback

import numpy as np
import torch
import torch.utils.data
from scipy.io.wavfile import read


class RVCDataset:
    """Prepared training set: an experiment directory plus its sample records."""

    def __init__(self, exp_dir, sample_rate, version, records):
        self.exp_dir = exp_dir
        self.sample_rate = sample_rate
        self.version = version
        self.records = records

    @property
    def feature_paths(self):
        return [record[1] for record in self.records]


class AudioFeatureDataset(torch.utils.data.Dataset):
    """Loads 16 kHz features, f0 and linear spectrograms for one training slice."""

    def __init__(self, records, mel, sampling_rate, hop_length, filter_length, win_length):
        self.records = records
        self.mel = mel
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.filter_length = filter_length
        self.win_length = win_length
        self.lengths = [os.path.getsize(record[0]) // (3 * self.hop_length) for record in records]

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)
        return (
            torch.FloatTensor(phone[:n_num, :]),
            torch.LongTensor(pitch[:n_num]),
            torch.FloatTensor(pitchf[:n_num]),
        )

    def get_audio(self, filename):
        sampling_rate, data = read(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("%s SR doesn't match target %s SR" % (sampling_rate, self.sampling_rate))
        audio_norm = torch.FloatTensor(data.astype(np.float32)).unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                return torch.load(spec_filename, weights_only=True), audio_norm
            except Exception:
                print("[RVC-Trainer] Recomputing unreadable spectrogram cache %s\n%s" % (spec_filename, traceback.format_exc()))
        spec = self.mel.spectrogram(audio_norm)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        wav_path, phone_path, pitch_path, pitchf_path, sid = self.records[index]
        phone, pitch, pitchf = self.get_labels(phone_path, pitch_path, pitchf_path)
        spec, wav = self.get_audio(wav_path)
        sid = self.get_sid(sid)

        len_min = min(phone.size(0), spec.size(-1))
        if phone.size(0) != spec.size(-1):
            spec = spec[:, :len_min]
            wav = wav[:, : len_min * self.hop_length]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]
        return spec, wav, phone, pitch, pitchf, sid

    def __len__(self):
        return len(self.records)


class TextAudioCollate:
    """Zero-pads a batch of (spec, wave, phone, pitch, pitchf, sid) tuples."""

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max(x[0].size(1) for x in batch)
        max_wave_len = max(x[1].size(1) for x in batch)
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.zeros(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.zeros(len(batch), 1, max_wave_len)

        max_phone_len = max(x[2].size(0) for x in batch)
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.zeros(len(batch), max_phone_len, batch[0][2].shape[1])
        pitch_padded = torch.zeros(len(batch), max_phone_len, dtype=torch.long)
        pitchf_padded = torch.zeros(len(batch), max_phone_len)
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class BucketSampler(torch.utils.data.Sampler):
    """Batches samples of similar length. Samples outside the boundaries are dropped."""

    def __init__(self, dataset, batch_size, boundaries, shuffle=True):
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets, self.num_samples_per_bucket = self._create_buckets()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for bucket in buckets:
            rem = (self.batch_size - (len(bucket) % self.batch_size)) % self.batch_size
            num_samples_per_bucket.append(len(bucket) + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.shuffle:
            indices = [torch.randperm(len(bucket), generator=g).tolist() for bucket in self.buckets]
        else:
            indices = [list(range(len(bucket))) for bucket in self.buckets]

        batches = []
        for i, bucket in enumerate(self.buckets):
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            rem = self.num_samples_per_bucket[i] - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: rem % len_bucket]
            for j in range(len(ids_bucket) // self.batch_size):
                batches.append([bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]])

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        return iter(batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            if x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            return self._bisect(x, mid + 1, hi)
        return -1

    def __len__(self):
        return sum(self.num_samples_per_bucket) // self.batch_size
