from functools import reduce
from fairseq import criterions, data
from fairseq.tasks.optimize_ali_speech_language import OptimizingAlignmentConfig, OptimizingAlignmentTask
import numpy as np
from fairseq.data import data_utils
from fairseq.models.hubert.hubert_asr import HubertTextMTLConfig, HubertTextMTL
import torch
import random
from fairseq.criterions.ctc_mlm_mtl import CtcMlmCriterion, CtcMLMCriterionConfig

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
        max_length  = input.shape[1]
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
    audio_input = torch.randn((16,80000))
    lengths = [random.randint(70000,80000) for i in range(16)]
    audio_mask = get_mask(audio_input, lengths)
    text_input =  [ torch.from_numpy(np.array([random.randint(0,364) for  i in range(498)])) for i in range(16)]
    text_lengths = [498 for i in range(16)]
    text_input, text_lengths, _ = collater_seq_label(text_input, task.dictionaries["phoneme"].pad())
    text_mask = get_mask(text_input, text_lengths)
    bpe_input =  [ torch.from_numpy(np.array([random.randint(0,10000) for  i in range(200)])) for i in range(16)]
    bpe_lengths = [200 for i in range(16)]
    bpe_input, bpe_lengths, _ = collater_seq_label(text_input, task.dictionaries["bpe"].pad())
    bpe_mask = get_mask(text_input, text_lengths)


    criterions_conf = CtcMLMCriterionConfig()
    crit = CtcMlmCriterion(criterions_conf, task)
    print(text_input.shape[1]/2)
    sample={
        'net_input':{
            "audio_source":audio_input,
            "padding_mask": audio_mask,
            "prev_phoneme": text_input,
            "phoneme_padding_mask": text_mask,
            "mode": "speech"
        },
        'input_audio_length': (torch.from_numpy(np.array(lengths)) - (400-320)) / 320,
        'phoneme_length': text_lengths/2,
        'phoneme_targets': text_input[:,:int(text_input.shape[1]/2)],
        'bpe_length': bpe_lengths,
        'bpe_target': bpe_input,
        'mode': 'speech'
    }
    out = crit(model, sample, reduce=False)
    # output = model(
    #     audio_input,
    #     audio_mask,
    #     text_input,
    #     text_mask,
    #     _type="speech"
    # )
    # output2 = model(
    #     audio_source= None,
    #     padding_mask=None,
    #     prev_phoneme=text_input,
    #     phoneme_padding_mask=text_mask,
    #     _type="text"
    # )
    # print(output)