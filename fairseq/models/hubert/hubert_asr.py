# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from _typeshed import Self
import contextlib
from argparse import Namespace
from logging import setLogRecordFactory
import math
from typing import Any, List
import random
from fairseq.data.dictionary import Dictionary

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.masked_lm import MaskedLMEncoder
from fairseq.tasks.optimize_ali_speech_language import OptimizingAlignmentTask
from fairseq.data.data_utils import compute_mask_indices

@dataclass
class HubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class HubertCtcConfig(HubertAsrConfig):
    pass


@register_model("hubert_ctc", dataclass=HubertCtcConfig)
class HubertCtc(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x



@dataclass
class HubertSeq2SeqConfig(HubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )

class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }



        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class MaskedTextEncoderConfig(HubertCtcConfig):
    text_encoder_layers: int = field(
        default=8, metadata={"help": "num encoder layers in the text encoder"}
    )
    
    # masking
    text_encoder_apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    text_encoder_mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    text_encoder_mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    text_encoder_mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    text_encoder_mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    text_encoder_no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    text_encoder_mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

@dataclass
class HubertTextMTLConfig(MaskedTextEncoderConfig):
    shared_encoder_layer: int = field(
        default=3,
        metadata={"help": "the number of shared encoder layers"},
    )
    embedding_aligner_dim: int = field(
        default=1024,
        metadata={"help": "the dimension of embedding aligner"}
    )
    swap_embedding_ratio: float = field(
        default=0,
        metadata={"help": "the probability of embedding swapping"}
    )
    swap_embedding_phoneme_aware = field(
        default=True,
        metadata={"help": "swap embedding with phoneme aware"}
    )

class MaskedTextEncoder(BaseFairseqModel):
    def __init__(
        self,
        cfg:HubertTextMTLConfig,
        task:OptimizingAlignmentTask,
        dictionaries
    ):
        super().__init__()
        # 1. token embedding
        self.token_embedding = self.build_embedding(cfg,dictionaries["phoneme"],cfg.w2v_args.encoder_embed_dim)
        # 2. text encoder
        self.MASK = task.MASK
        self.encoder_layers = [ self.build_encoder_layer(cfg) for i in range(cfg.text_encoder_layers)]
        self.proj = nn.Linear(cfg.w2v_args.model.encoder_embed_dim, cfg.encoder_output_dim)
        self._dictionaries = dictionaries
        self._mask_prob = cfg.text_encoder_mask_prob
        self._apply_mask = cfg.text_encoder_apply_mask
        self._mask_length = cfg.text_encoder_mask_length
        self._mask_selection = cfg.text_encoder_mask_selection
        self._mask_other = cfg.text_encoder_mask_other
        self._no_mask_overlap = cfg.text_encoder_no_mask_overlap
        self._mask_min_space = cfg.text_encoder_mask_min_space

    def forward(
        self,
        prev_phoneme,
        prev_phoneme_mask,
    ):
        # 1. apply mask
        if self.apply_mask:
            prev_phoneme = self.apply_mask(prev_phoneme, self.dictionaries["phoneme"])
        # 2. embedding
        prev_phoneme = self.token_embedding(prev_phoneme)
        # 3. encoder
        for transformer in self.encoder_layers:
            prev_phoneme = transformer(prev_phoneme, prev_phoneme_mask)
        # 4. project
        prev_phoneme = self.proj(prev_phoneme)
        return {
            "encoder_out": prev_phoneme,
            "padding_mask": prev_phoneme_mask
        }

    def apply_mask(self, x, padding_mask, target_list):
        B, T = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self._mask_prob,
                self._mask_length,
                self._mask_selection,
                self._mask_other,
                min_masks=2,
                no_overlap=self._no_mask_overlap,
                min_space=self._mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.MASK
        else:
            mask_indices = None
        return x, mask_indices
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

@register_model("hubert_text_mtl")
class HubertTextMTL(BaseFairseqModel):
    def __init__(
        self, 
        cfg: HubertTextMTLConfig, 
        w2v_encoder: BaseFairseqModel, 
        text_encoder:BaseFairseqModel, 
        decoder: BaseFairseqModel,
        embedding_aligner,
        ctc_proj
    ):
        super().__init__()
        self.cfg = cfg
        # 1. audio encoder
        self.w2v_encoder = w2v_encoder
        # 2. text encoder
        self.text_encoder = text_encoder
        # 3. decoder
        self.decoder = decoder
        # 4. shared encoder
        self.shared_encoder = [ self.build_encoder_layer(cfg) for i in range(cfg.shared_encoder_layer)]
        # 5. embedding aligner
        self.embedding_aligner = embedding_aligner
        # 6. ctc proj
        self.proj = ctc_proj
        self.swap_embedding_ratio = cfg.swap_embedding_ratio
        self.swap_embedding_phoneme_aware = cfg.swap_embedding_phoneme_aware
        
        
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, cfg: HubertTextMTLConfig, task: FairseqTask):
        """Build a new model instance."""
        # 1. audio encoder
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)

        # 2. text encoder
        text_encoder_embedding = cls.build_embedding(
            cfg, task.state.dictionaries["phoneme"], cfg.text_encoder_embed_dim, None
        )
        text_encoder = MaskedTextEncoder(
            cfg,
            task,
            task.state.dictionaries
        )
        # 3. decoder
        decoder_embedding = cls.build_embedding(
            cfg, task.state.dictionaries["bpe"], cfg.decoder_embed_dim, None
        )
        decoder = TransformerDecoder(
            cfg, 
            task.target_dictionary,
            decoder_embedding
        )
        # embedding_aligner
        embedding_aligner = nn.Parameter(
            torch.FloatTensor(
                (cfg.encoder_ffn_embed_dim, 
                len(task.state.dictionaries["phonome"]))
            )
        )

        # ctc proj
        ctc_proj = nn.Linear(cfg.encoder_output_dim, len(task.state.dictionaries["bpe"]))
        return cls(cfg, w2v_encoder, text_encoder, decoder, embedding_aligner, ctc_proj)

    def forward(
        self, 
        audio_source, 
        padding_mask,
        prev_phoneme,
        phoneme_padding_mask,
        _type: str = "speech",
    ):
        if _type == "speech":
            return self.forward_speech(
                audio_source,
                padding_mask,
                prev_phoneme,
                phoneme_padding_mask
            )
        else:
            return self.forward_text(
                prev_phoneme,
                phoneme_padding_mask
            )

    def swap_embedding(self,audio_embedding, text_embedding, accum_alignment):
        # audio_embedding is B*T*D 
        # text_embedding is B*T*D
        # assert the length of audio embedding is the same as text embedding
        assert(audio_embedding.shape[1] == text_embedding.shape[1])
        # building mask
        bsz = audio_embedding.shape[0]
        for i in range(bsz):
            if self.swap_embedding_phoneme_aware:
                mask = torch.ones((audio_embedding.shape[1]), device=audio_embedding.device)

                indices = random.sample(list(range(len(1,accum_alignment[i]))), 
                    math.ceil(len(accum_alignment)* self.swap_embedding_ratio))
                for index in indices:
                    start,end = accum_alignment[i][index-1], accum_alignment[i][index]
                mask[i][start:end] = 0
            else:
                mask = (
                    torch.randn(
                        text_embedding[i].shape[0], 
                        device=text_embedding.device
                    ).uniform() > self.swap_embedding_ratio
                ).float()
            text_embedding_tmp = text_embedding[i] * mask + audio_embedding[i] * (1 - mask)
            audio_embedding[i] = audio_embedding[i] * mask + text_embedding[i] * (1 - mask)
            text_embedding[i] = text_embedding_tmp
            


    def get_accum_from_phoneme_seq(self, phoneme_seq, phoneme_padding_mask):
        bsz = phoneme_seq.shape[0]
        accum_lists = []
        for i in range(bsz):
            accum = [indice+1 for indice,j in enumerate(range(phoneme_seq[i].shape[0])) 
                if phoneme_padding_mask[i][j] == True and phoneme_seq[i][j]!=phoneme_seq[i][j+1] ]
            accum_lists.append(accum)
        return accum_lists
    
    def forward_speech(
        self,
        audio_source,
        padding_mask,
        prev_phoneme,
        phoneme_padding_mask
    ):
        # 1. audio encoder
        # assert audio input is feature
        assert(len(audio_source)==3)
        encoder_out = self.w2v_encoder(audio_source, padding_mask, False)
        padding_mask = padding_mask[:, :3:, ]
        # 2. text_encoder 
        text_encoder_out = self.text_encoder(prev_phoneme,phoneme_padding_mask)
        # 3. text_encoder -> swap embedding
        self.swap_embedding(
            encoder_out["encoder_out"], 
            text_encoder_out["encoder_out"],
            self.get_accum_from_phoneme_seq(prev_phoneme)
        )
        x = encoder_out["encoder_out"]
        xt = text_encoder_out["encoder_out"]
        # 4. audio encoder -> embedding aligner -> ctc prob
        #    text encoder -> embedding aligner -> mlm prob
        x = nn.functional.softmax(nn.functional.pairwise_distance(x,self.embedding_aligner), -1)
        xt = nn.functional.softmax(nn.functional.pairwise_distance(xt,self.embedding_aligner), -1)
        # 5. audio encoder -> shared encoder
        out = encoder_out["encoder_out"]
        for transformer in self.shared_encoder:
            out = transformer(out, encoder_out["encoder_padding_mask"])
        out = self.proj(out)
        return {
            "ctc_prob": x,
            "mlm_prob": xt,
            "final_ctc_prob": out,
            "phoneme_padding_mask": padding_mask,
            
        }


    def forward_text(
        self,
        prev_phoneme,
        phoneme_padding_mask
    ):
        # 1. text encoder
        encoder_out = self.text_encoder(prev_phoneme, phoneme_padding_mask)
        # 2. audio encoder -> embedding aligner -> MLM prob
        x = encoder_out["encoder_out"]
        x = nn.functional.softmax(
            nn.functional.pairwise_distance(x,self.embedding_aligner)
        )
        # 4. audio encoder -> shared encoder
        out = encoder_out["encoder_out"]
        for transformer in self.shared_encoder:
            out = transformer(out, encoder_out["encoder_padding_mask"])
        out = self.proj(out)
        return {
            "mlm_prob": x,
            "final_ctc_prob": out,
            "phoneme_padding_mask": phoneme_padding_mask
        }

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


if __name__=='__main__':
    from fairseq.tasks.optimize_ali_speech_language import OptimizingAlignmentConfig, OptimizingAlignmentTask
    import numpy as np
    from fairseq.data import data_utils
    def collater_seq_label(targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens
    def get_mask(input, length):
        bsz = len(length)
        max_length  = len(input)
        padding_mask = (
            torch.BoolTensor(bsz, max_length).fill_(False)
        )
        for i in range(bsz):
            padding_mask[i,  length[i]:] = True
        return padding_mask

    config = HubertTextMTLConfig()
    task_config = OptimizingAlignmentConfig(
        speech_data="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/train_960/data_format.train",
        text_data="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/text_only_data/librispeech-lm-norm.txt",
        label_dir="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/train_960/label"
    )
    task = OptimizingAlignmentTask(task_config)
    model = HubertTextMTL.build_model(config,task)
    audio_input = torch.randn((16,500))
    lengths = [np.randint(450,500) for i in range(16)]
    audio_mask = get_mask(audio_input, lengths)
    text_input =  [ [np.randint(364) for  i in range(500)] for i in range(16)]
    text_lengths = [500 for i in range]
    text_input, text_lengths, _ = collater_seq_label(text_input, task.dictionaries["phoneme"].pad())
    text_mask = get_mask(text_input, text_lengths)
    output = model(
        audio_input,
        audio_mask,
        text_input,
        text_mask,
        _type="speech"
    )