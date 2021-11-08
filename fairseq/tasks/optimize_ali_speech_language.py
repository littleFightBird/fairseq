# v-zhuoyao@microsoft.com
# 2021 / 10 / 27
# Define the task of optimizing alignment of speech and language latent spaces for e2e speech
# recognition and understanding
# arxiv:
import logging
import os
from random import shuffle
import sys
from typing import Dict, List, Optional, Tuple, Union
from fairseq.dataclass import configs

import numpy as np
import sentencepiece as spm

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.data.audio.audio_text_dataset import AudioDataset, TextDataset
from fairseq.data.audio.multi_modality_dataset import MultiModalityDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )

class SentencepiecesTokenizer(object):
    def __init__(self, model: str):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def __call__(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)


@dataclass
class OptimizingAlignmentConfig(FairseqDataclass):
    speech_data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    text_data: str = field(
        default=MISSING, metadata={"help": "path to text only data directory"}
    )
    label_dir: str = field(
        default=MISSING, metadata={"help": "path to label"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["phoneme", "word"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    text_in_mask_ratio: float = field(
        default=0.2,
        metadata={"help":"mask text ratio"}
    )
    swap_embedding_ratio: float = field(
        default=0.2,
        metadata={"help":"the ratio of swap embedding from speech encoder and text encoder"}
    )
    swap_embedding_phoneme_aware: bool=field(
        default=True,
        metadata={"help":"swap embedding with the phoneme limit"}
    )
    shuffle: bool=field(
        default=True,
        metadata={"help":"shuffle the dataset or not"}
    )
    fbank_bin:bool = field(
        default=80,
        metadata={"help": "fbank bins of the model"}
    )
    audio_max_token: int = field(
        default=350000,
        metadata={"help": "max token in batch for audio"}
    )
    audio_max_sentences: int = field(
        default=350000,
        metadata={"help": "max sentences in batch for audio"}
    )
    text_max_token: int = field(
        default=350000,
        metadata={"help": "max token in batch for text"}
    )
    text_max_sentences: int = field(
        default=350000,
        metadata={"help": "max sentences in batch for text"}
    )


@register_task("optimizing_alignment_speech_language", dataclass=OptimizingAlignmentConfig)
class OptimizingAlignmentTask(FairseqTask):

    cfg: OptimizingAlignmentConfig

    def __init__(
        self,
        cfg: OptimizingAlignmentConfig
    ) -> None:
        super().__init__()
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries['bpe']

    @property
    def phoneme_dictionary(self) -> Optional[Dictionary]:
        return self.state.dictionaries['phoneme']

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries
    
    @classmethod
    def setup_task(
        cls, cfg: OptimizingAlignmentConfig, **kwargs
    ) -> "OptimizingAlignmentTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = {
            "phoneme":Dictionary.load(f"{label_dir}/dict.phoneme.txt"),
            "bpe":Dictionary.load(f"{label_dir}/dict.bpe.txt") 
        }
        self.MASK = dictionaries["phoneme"].add_symbol("<mask>")
        return dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split:str, **kwargs) ->None:
        bpe_model = f"{self.cfg.label_dir}/bpe_model/bpe.model"
        dicts = self.dictionaries()
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]

        procs = [LabelEncoder(dicts["phoneme"]), SentencepiecesTokenizer(bpe_model) ]
        audio_dataset = AudioDataset(
            audio_path=self.cfg.speech_data,
            sample_rate=self.cfg.sample_rate,
            label_processors=procs,
            pad_list=pad_list,
            eos_list=eos_list,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            shuffle=self.cfg.shuffle,
            normalize=self.cfg.normalize,
            pad_audio=self.cfg.pad_audio,
            fbank_bins=self.cfg.fbank_bin,
            max_sample_size=self.cfg.max_keep_size,
            max_tokens=self.cfg.audio_max_token,
            max_sentences=self.cfg.audio_max_sentences
        )

        text_dataset = TextDataset(
            data_file_path=self.cfg.text_data,
            max_text_num=self.cfg.max_sample_size,
            min_text_num=self.cfg.min_sample_size,
            data_process=procs,
            shuffle=self.cfg.shuffle,
            pad_list=pad_list,
            max_tokens=self.cfg.text_max_token,
            max_sentences=self.cfg.text_max_sentences
        )

        self.datasets[split] = MultiModalityDataset(
            datasets=[audio_dataset,text_dataset]
        )
        # hubert v1: pad_audio=True, random_crop=False;
        # self.datasets[split] = AudioTextDataset(
        #     manifest,
        #     text_only_data_set,
        #     sample_rate=self.cfg.sample_rate,
        #     label_paths=paths,
        #     pad_list=pad_list,
        #     eos_list=eos_list,
        #     label_processors=procs,
        #     max_keep_sample_size=self.cfg.max_keep_size,
        #     min_keep_sample_size=self.cfg.min_sample_size,
        #     max_sample_size=self.cfg.max_sample_size,
        #     pad_audio=self.cfg.pad_audio,
        #     normalize=self.cfg.normalize,
        #     store_labels=False,
        #     random_crop=self.cfg.random_crop,
        #     single_target=self.cfg.single_target,
        # )