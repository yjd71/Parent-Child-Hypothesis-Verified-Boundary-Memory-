#!/bin/bash

devices=$1
config=$2
ckpt=$3
testset="CHAMELEON+TE-COD10K+TE-CAMO+NC4K"

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=$devices python scripts/evaluate.py --config=${config} --model_dir=${ckpt} --testset=${testset}
