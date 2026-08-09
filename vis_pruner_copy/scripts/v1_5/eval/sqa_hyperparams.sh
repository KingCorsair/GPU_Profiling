#!/bin/bash

CKPT_DIR="/workspace/GPU_Profiling/vis_pruner_copy/checkpoints"
DATA_DIR="/workspace/GPU_Profiling/ScienceQA/data"

CKPT="llava-v1.5-7b"
SPLIT="llava_test_CQM-A"

# RATIO is now the first argument defualts to 0.5
RATIO=${1:-0.5}

# Array of token settings to iterate over
TOKENS=(64 144 288 576)

for TOKEN in "${TOKENS[@]}"; do
    echo " Running Evaluation @ TOKEN: ${TOKEN} and RATIO: ${RATIO}"

    # Create an output dir here if we don't have one
    mkdir -p ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}

    python -m llava.eval.model_vqa_science \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/scienceqa/${SPLIT}.json \
        --image-folder ${DATA_DIR}/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
        --visual_token_num ${TOKEN} \
        --important_ratio ${RATIO} \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    python -m llava.eval.eval_science_qa \
        --base-dir ${DATA_DIR}/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}_output.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}_result.json
done