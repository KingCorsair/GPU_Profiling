#!/bin/bash

# DOWNLOADING DATASET
git clone https://github.com/lupantech/ScienceQA.git
mkdir data/scienceqa/images
bash ScienceQA/tools/download.sh


# DOWNLOADING HF HUB AND MODEL
pip install huggingface_hub
mkdir vis_pruner_copy/checkpoints
hf-hub download liuhaotian/llava-v1.5-7b --local-dir /vis_pruner_copy/checkpoints/llava-v1.5-7b
