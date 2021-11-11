# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from random import sample
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import get_fbank
from g2p_en import G2p

logger = logging.getLogger(__name__)

def load_paired_data(manifest_path, max_keep, min_keep):
    n_long, n_short = 0,0
    data_dict, inds, sizes = [], [], []
    with open(manifest_path) as f:
        for ind, line in enumerate(f):
            items = line.strip().split(":")
            assert len(items) ==6, line

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
                        "size": sz,
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
    n_long, n_short = 0.0, 0.0  
    data_dict, inds, sizes = [],[],[]
    with open(text_only_data_set_path) as f:
        for  ind, line in enumerate(f):
            word = line.strip().split(" ")
            sz = len(word)
            if min_text is not None and sz < min_text:
                n_short+=1
            if max_text is not None and sz > max_text:
                n_long+=1
            inds.append(ind)
            data_dict.append(
                {
                    "word": word,
                    "style": "text",
                    "size": sz
                }
            )
            sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"load text only data"
            f"max_keep={max_text}, min_keep={min_text}, "
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
        pad_list: List[str] = None,
        eos_list: List[str] = None,
        shuffle: bool = True,
        pad_audio: bool = True,
        normalize: bool = False,
        fbank_bins: int = 80,
        max_sample_size: int=100000000,
    ):
        self.audio_data_dict, self.audio_inds, self.audio_sizes = load_paired_data(
            audio_path, max_keep_sample_size, min_keep_sample_size
        )

        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.fbank_bins = fbank_bins
        self.max_sample_size = max_sample_size
        self.normalize = normalize
        self.dataset = self
        self.pad_audio = pad_audio

    def __getitem__(self, index):
        wav = self.get_audio(index)
        phoneme_token,bpe_token = self.get_label(index)
        '''
            notice!!!
            phoneme > 10 is because of the 0-10 in the dictionary of phoneme is <eps>, SIL, SPN 
        '''
        phoneme_token_no_rep = torch.from_numpy(np.array( [ int(phoneme_token[i]) for i in range(1,len(phoneme_token)) if phoneme_token[i] > 10 and (i==1 or phoneme_token[i]!=phoneme_token[i-1]) ] ))
        return {"id": index, "source": wav, "phoneme": phoneme_token, "bpe":bpe_token, "phoneme_target": phoneme_token_no_rep}

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self.audio_sizes

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
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def get_label(self, index):
        data = self.audio_data_dict[index]
        phoneme_token = self.label_processors["phoneme"](data["phoneme"])
        bpe_token = self.label_processors["word"](data["word"])
        bpe_token = self.label_processors["bpe"](bpe_token)
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

        phoneme_input = [s["phoneme"] for s in samples]
        bpe_target = [s["bpe"] for s in samples]
        phoneme_target = [s["phoneme_target"] for s in samples] 

        phoneme_mask = self.phoneme_padding_mask(phoneme_input)
        data_list, lengths_list, ntokens_list = self.collater_label(
            phoneme_input, bpe_target, phoneme_target
        )
        net_input = {
            "audio_source": collated_audios, 
            "padding_mask": padding_mask, 
            "prev_phoneme": data_list[0], 
            "phoneme_padding_mask": phoneme_mask,
            "mode": "speech",
            "lengths": (torch.from_numpy(np.array(audio_sizes))- (400-320)) / 320
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        batch["input_audio_length"] = (torch.from_numpy(np.array(audio_sizes)) - (400-320)) / 320
        batch["phoneme_length"] = lengths_list[2]
        batch["phoneme_ntoken"] = ntokens_list[2]
        batch["phoneme_target"] = data_list[2]
        batch["bpe_length"] = lengths_list[1]
        batch["bpe_ntoken"] = ntokens_list[1]
        batch["bpe_target"] = data_list[1]
        return batch

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        padd_mask = torch.zeros((batch_size, max_size)).bool()
        for  i, phoneme in enumerate(phoneme_target):
            diff =  max_size - len(phoneme) 
            if diff == 0:
                continue
            elif diff < 0:
                padd_mask[i, diff:] = True
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

    def collater_label(self, phoneme_input, bpe_target, phoneme_target):
        phoneme_inputs, phoneme_lengths, phoneme_ntokens = self.collater_seq_label(
            phoneme_input, self.pad_list[0]
        )
        bpe_targets, bpe_lengths, bpe_ntokens = self.collater_seq_label(
            bpe_target, self.pad_list[1]
        )
        phoneme_targets, t_phoneme_lengths, t_phoneme_ntokens = self.collater_seq_label(
            phoneme_target, self.pad_list[0]
        )
        
        targets = [phoneme_inputs, bpe_targets, phoneme_targets]
        lengths = [phoneme_lengths, bpe_lengths, t_phoneme_lengths]
        ntokens = [phoneme_ntokens, bpe_ntokens, t_phoneme_ntokens]

        return targets, lengths, ntokens

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index: int):
        return self.size(index)

class TextDataset(FairseqDataset):
    def __init__(
        self,
        data_file_path: str,
        lexicon_path: str,
        accume_path: str,
        max_text_num:int = None,
        min_text_num:int = None,
        data_process:Optional[List[Any]] = None,
        shuffle: bool = True,
        pad_list: List[str] = None,
        
    ):
        self.data_dict, self.inds, self.text_sizes = load_text_only_data(
            data_file_path, max_text_num, min_text_num
        )
        self.shuffle = shuffle
        self.pad_list = pad_list
        self.dataset = self
        self.lexicon = self.load_lexicon(lexicon_path)
        self.accum_stat = self.load_accum_stat(accume_path)
        self.data_process = data_process
        self.g2p = G2p()

    def load_lexicon(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f.readlines():
                item = line.strip().split()
                lexicon[item[0]] = item[1:]
        return lexicon

    @property
    def sizes(self):
        return self.text_sizes

    def load_accum_stat(self, accum_path):
        accum_stat = {}
        str_map = {}
        with open(accum_path) as f:
            for  line in f.readlines():
                item = line.strip().split()
                accum_stat[item[0]]=int(item[1])
        for  key in accum_stat.keys():
            phoneme = key.split("_")[0]
            if phoneme not in str_map.keys():
                str_map[phoneme] = ((phoneme+"_B"+" ") * accum_stat[phoneme+"_B"] + \
                                    (phoneme+"_I"+" ") * accum_stat[phoneme+"_I"] + \
                                    (phoneme+"_E"+" ") * accum_stat[phoneme+"_E"] + \
                                    (phoneme+"_S"+" ") * accum_stat[phoneme+"_S"] ).split()
        return str_map
    def __getitem__(self, index):
        phoneme_token,bpe_token, phoneme_target = self.get_labels(index)
        return {"id": index,  "phoneme": phoneme_token, "bpe":bpe_token, "phoneme_target": phoneme_target}

    def get_labels(self, index):
        words = self.data_dict[index]["word"]
        bpe_token = self.data_process["word"](words)
        bpe_token = self.data_process["bpe"](bpe_token)
        phoneme_token = []
        phoneme_norep_token = []
        for word in words:
            if word in self.lexicon.keys():
                build_string = ''
                for s in word:
                    build_string += s+ " "
                phoneme_seq = self.g2p(build_string)
                phoneme_seq = [i for i in phoneme_seq if i != ' ']
                phoneme_norep_token.extend(phoneme)
                for phoneme in phoneme_seq:
                    phoneme_token.extend(self.accum_stat[phoneme])
        phoneme_token = self.data_process["phoneme"](phoneme_token)
        phoneme_norep_token = self.data_process["phoneme"](phoneme_norep_token)
        return phoneme_token, bpe_token, phoneme_norep_token

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index: int):
        return self.size(index)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def collater(self, samples):
        phoneme_input = [s["phoneme"] for s in samples]
        bpe_output = [s["bpe"] for s in samples ]
        phoneme_target = [s["phoneme_target"] for s in samples]
        phoneme_mask = self.phoneme_padding_mask(phoneme_input)
        phoneme_input, phoneme_lengths, phoneme_ntokens = self.collater_seq_label(
            phoneme_input, self.pad_list[0]
        )
        phoneme_target, phoneme_target_lengths, phoneme_target_ntokens = self.collater_seq_label(
            phoneme_target,self.pad_list[0]
        )
        bpe_output, bpe_lengths, bpe_ntokens = self.collater_seq_label(
            bpe_output, self.pad_list[1]
        )

        net_input = {
            "prev_phoneme": phoneme_input, 
            "phoneme_padding_mask": phoneme_mask,
            "mode":"text",
            "lengths":phoneme_lengths
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        batch["input_phoneme_lengths"] = phoneme_target_lengths
        batch["bpe_target"] = bpe_output
        batch["bpe_length"] = bpe_lengths
        batch["phoneme_target"] = phoneme_target
        batch["phoneme_length"] = phoneme_target_lengths
        return batch

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def __len__(self):
        return len(self.sizes)

    def phoneme_padding_mask(self, phoneme_target):
        phoneme_sizes = [ len(s) for s in phoneme_target]
        max_size = max(phoneme_sizes)
        batch_size = len(phoneme_target)
        padd_mask = torch.BoolTensor((batch_size, max_size)).fill_(False)
        for  i, phoneme in enumerate(phoneme_target):
            diff = len(phoneme) - max_size
            if diff == 0:
                continue
            else:
                padd_mask[i,diff:]=True
        return padd_mask

