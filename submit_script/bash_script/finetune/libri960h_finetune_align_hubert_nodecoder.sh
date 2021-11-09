#!/bin/bash
model_path=/home/v-zhuoyao/workspace/fairseq/submit_script/bash_script/finetune
train_subset=train_960
valid_subset=dev_other

mkdir -p ${model_path}


python train.py \
--distributed-world-size 8 \
--distributed-port 0 \
--nprocs-per-node 8 \
/home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/data/librispeech \
--save-dir ${model_path} \
--train-subset ${train_subset} \
--valid-subset ${valid_subset} \
--no-epoch-checkpoints \
--best-checkpoint-metric wer \
--num-workers 4 \
--max-update 80000 \
--sentence-avg \
--task hubert_pretraining \
--fine-tuning \
--single-target \
--arch hubert_ctc \
--w2v-path /home/v-zhuoyao/workspace/fairseq_master/examples/zhuoyao_optimizing_ali/s0/exp/hubert_pretrain/hubert_base_ls960.pt \
--labels '["ltr"]' \
--apply-mask \
--mask-selection static \
--mask-other 0 \
--mask-length 10 \
--mask-prob 0.65 \
--layerdrop 0.1 \
--mask-channel-selection static \
--mask-channel-other 0 \
--mask-channel-length 64 \
--mask-channel-prob 0.5 \
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
--criterion ctc \
--attention-dropout 0.1 \
--dropout-input 0.1 \
--max-tokens 3200000 \
--seed 2337 \
--log-format json \
--log-interval 200 \
--ddp-backend c10d \
--fp16 \
--update-freq 1 \
--keep-interval-updates 1 \
--find-unused-parameters >> ${model_path}/log
