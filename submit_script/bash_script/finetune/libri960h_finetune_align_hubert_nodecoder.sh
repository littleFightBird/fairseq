#!/bin/bash
model_path=/home/v-zhuoyao/workspace/fairseq/submit_script/bash_script/finetune
data_path=/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech
train_subset=train_960
valid_subset=dev_other

mkdir -p ${model_path}

cd /home/v-zhuoyao/workspace/fairseq

python train.py \
    --distributed-world-size 1 \
    --distributed-port 0 \
    --nprocs-per-node 1 \
    --save-dir ${model_path} \
    --speech-data ${data_path}/${train_subset}/data_format.train \
    --text-data ${data_path}/text_only_data/librispeech-lm-norm.txt \
    --lexicon-path ${data_path}/text_only_data/librispeech-lexicon.txt \
    --accum-path ${data_path}/${train_subset}/label/dict.accum.txt \
    --valid-subset ${data_path}/${valid_subset}/data_format.dev \
    --label-dir ${data_path}/${train_subset}/label \
    --no-epoch-checkpoints \
    --best-checkpoint-metric wer \
    --num-workers 4 \
    --max-update 80000 \
    --sentence-avg \
    --task optimizing_alignment_speech_language \
    --arch hubert_text_mtl \
    --criterion ctc_mlm \
    --w2v-path /home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/exp/hubert_pretrain/hubert_base_ls960.pt \
    --text-encoder-mask-selection static \
    --text-encoder-mask-other 0 \
    --text-encoder-mask-length 10 \
    --text-encoder-mask-prob 0.65 \
    --zero-infinity \
    --feature-grad-mult 0 \
    --freeze-finetune-updates 0 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-08 \
    --lr 0.00003 \
    --lr-scheduler tri_stage \
    --warmup-steps 8000 \
    --hold-steps 32000 \
    --decay-steps 40000 \
    --final-lr-scale 0.05 \
    --final-dropout 0.1 \
    --dropout 0.1 \
    --activation-dropout 0.1 \
    --attention-dropout 0.1 \
    --dropout-input 0.1 \
    --max-tokens 5000 \
    --seed 2337 \
    --log-format json \
    --log-interval 200 \
    --ddp-backend c10d \
    --update-freq 1 \
    --keep-interval-updates 1 \
    --find-unused-parameters  \
    ${data_path} >> ${model_path}/log
