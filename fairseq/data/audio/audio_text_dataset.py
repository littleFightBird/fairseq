# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import get_fbank

logger = logging.getLogger(__name__)

def load_paired_data(manifest_path, max_keep, min_keep):
    n_long, n_short = 0,0
    data_dict, inds, sizes = [], [], []
    with open(manifest_path) as f:
        for ind, line in enumerate(f):
            items = line.strip().split(":")
            assert len(items) ==5, line

            sz = int(items[5])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                data_dict.append(
                    {
                        "id": items[1].split(" ")[0],
                        "path": items[2].split(" ")[0],
                        "phoneme": items[3].split(" ")[0:-1],
                        "word": items[4].split(" ")[0:-1],
                        "size": sz
                        "style": "paired"
                    }
                )
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"load paired data"
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(data_dict)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return data_dict, inds, sizes

    

def load_text_only_data(text_only_data_set_path, max_text, min_text):
    n_long, n_short = 0.0 
    sizes = []
    ind_offset = inds[-1]+1
    with open(text_only_data_set_path) as f:
        for  ind, line in enumerate(f):
            word = line.strip().split(" ")
            sz = len(word)
            if min_text is not None and sz < min_text:
                n_short+=1
            if max_text is not None and sz > max_text:
                n_long+=1
            inds.append(ind+ind_offset)
            data_dict.append(
                {
                    "word": word,
                    "style": "text",
                    "size": sz
                }
            )
            sizes.append(sz)
    tot += ind + 1
    logger.info(
        (
            f"load text only data"
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(data_dict)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return data_dict, inds, sizes



class AudioDataset(FairseqDataset):
    def __init__(
        self,
        audio_path: str,
        sample_rate: float,
        max_keep_sample_size: int = None,
        min_keep_sample_size: int = None,
        label_processors: Optional[List[Any]] = None,
        pad_list: List[str],
        eos_list: List[str],
        shuffle: bool = True,
        pad_audio: bool = True,
        normalize: bool = False,
        fbank_bins: int = 80,
        max_sample_size: int=None
    ):
        self.audio_data_dict, self.audio_inds, self.sizes = load_paired_data(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.fbank_bins = fbank_bins
        self.max_sample_size = max_sample_size
        self.max_text_size = max_text_size

    def __getitem__(self, index):
        wav = self.get_audio(index)
        phoneme_token,bpe_token = self.get_labels(index)
        return {"id": index, "source": wav, "phoneme": phoneme_token, "bpe":bpe_token}

    def __len__(self):
        return len(self.sizes)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def get_audio(self, index):
        import soundfile as sf

        wav_path = self.audio_data_dict[index]["path"]
        wav = get_fbank(wav_path,self.fbank_bins)
        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def get_label(self, index):
        data = self.audio_data_dict[index]
        phoneme_token = self.label_processors["phoneme"](data["phoneme"])
        bpe_token = self.label_processors["word"](data["word"])
        return phoneme_token, bpe_token

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)

        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        phoneme_target = [s["phoneme"] for s in samples]
        bpe_target = [s["bpe"] for s in samples]

        phoneme_mask = self.phoneme_padding_mask(phoneme_target)
        targets_list, lengths_list, ntokens_list = self.collater_label(
            phoneme_target, bpe_target, audio_size, audio_starts, phoneme_size
        )

        net_input = {
            "source": collated_audios, 
            "padding_mask": padding_mask, 
            "phoneme_source": targets_list[0], 
            "phoneme_mask": phoneme_mask
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        batch["phoneme_length"] = lengths_list[0]
        batch["phoneme_ntoken"] = ntokens_list[0]
        batch["phoneme_target"] = targets_list[0]
        batch["bpe_length"] = lengths_list[0]
        batch["bpe_ntoken"] = ntokens_list[0]
        batch["bpe_target"] = targets_list[0]
        return batch

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        padd_mask = torch.BoolTensor((batch_size, max_size)).fill_(False)
        for  i, phoneme in phoneme_target:
            diff = len(phoneme) - max_size
            if diff == 0:
                continue
            else:
                padd_mask[i,diff:]=True
        return padd_mask

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )

        return collated_audios, padding_mask, audio_starts

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, phoneme_target, bpe_target, audio_size, audio_starts):
        phoneme_targets, phoneme_lengths, phoneme_ntokens = self.collater_seq_label(
            phoneme_targets, self.pad_list[0]
        )

        bpe_targets, bpe_lengths, bpe_ntokens = self.collater_seq_label(
            phoneme_targets, self.pad_list[1]
        )
        
        targets = [phoneme_targets, bpe_targets]
        lengths = [phoneme_lengths, bpe_lengths]
        ntokens = [phoneme_ntokens, bpe_ntokens]

        return targets, lengths, ntokens

class TextDataset(FairseqDataset):
    def __init__(
        self,
        data_file_path: str,
        max_text_num:int = None,
        min_text_num:int = None,
        data_process:Optional[List[Any]] = None,
        shuffle: bool = True
    ):
        self.data_dict, self.inds = load_text_only_data(
            data_file_path, max_text_num, min_text_num
        )
        self.shuffle = shuffle

    def __getitem__(self, index):
        wav = self.get_audio(index)
        phoneme_token,bpe_token = self.get_labels(index)
        return {"id": index, "source": wav, "phoneme": phoneme_token, "bpe":bpe_token}

    def __len__(self):
        return len(self.sizes)

# We do batch sample in __init__
class AudioTextDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        text_only_data_set_path: str,
        sample_rate: float,
        pad_list: List[str],
        eos_list: List[str],
        fbank_bins: int = 80,
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: int = None,
        min_keep_sample_size: int = None,
        max_text_num: int = None,
        min_text_num: int = None,
        batch_max_sample_size: int = 3200000,
        batch_max_text_size: int = 3200000,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
    ):
        
        self.audio_data_dict, self.audio_inds = load_paired_data(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )

        self.text_data_dict, self.text_inds =  load_text_only_data(
            text_only_data_set_path, max_text_num, min_text_num
        ) 

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.fbank_bins = fbank_bins
        self.max_sample_size = max_sample_size
        self.max_text_size = max_text_size

        assert (
            label_processors is None
            or len(label_processors) == 2
        )

        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, batch_max_text_size={self.batch_max_text_size} "
            f"normalize={normalize}, batch_max_sample_size={self.batch_max_sample_size}"
        )
        # order by 

        # create text batch and audio batch
        audio_batch = self.create_batch(self.audio_data_dict, batch_max_sample_size)
        text_batch = self.create_batch(self.text_data_dict, batch_max_text_size)

    def create_batch(self, data_dict, batch_max_size):
        batchs = []
        batch = []
        size_accum = 0
        for data in data_dict:
            size = data['size']
            size_accum += size
            if size > batch_max_size:
                batchs.append(batch)
                batch=[data]
                size_accum = size
            else:
                batch.append(data)
        return batchs
            

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav = get_fbank(wav_path,self.fbank_bins)
        return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    
    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.pad_list)
        for targets, pad in itr:
            
            targets, lengths, ntokens = self.collater_seq_label(
                targets, pad
            )
            
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
