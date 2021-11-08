from fairseq import data
from fairseq.tasks.optimize_ali_speech_language import OptimizingAlignmentConfig, OptimizingAlignmentTask
import numpy as np
from fairseq.data import data_utils
from fairseq.models.hubert.hubert_asr import HubertTextMTLConfig, HubertTextMTL
import torch
import random

if __name__=='__main__':
    
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

    config = HubertTextMTLConfig(
        w2v_path="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/exp/hubert_pretrain/hubert_base_ls960.pt",
        normalize=False,
    )
    task_config = OptimizingAlignmentConfig(
        speech_data="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/train_960/data_format.train",
        text_data="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/text_only_data/librispeech-lm-norm.txt",
        label_dir="/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech/train_960/label"
    )
    task = OptimizingAlignmentTask(task_config)
    model = HubertTextMTL.build_model(config,task)
    audio_input = torch.randn((16,500))
    lengths = [random.randint(450,500) for i in range(16)]
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