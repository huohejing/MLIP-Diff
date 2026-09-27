#!/bin/bash

#python scripts/train.py --config ./configs/train/train.yml --device cuda:0 --logdir ./logs
#python scripts/train_bond.py --config ./configs/train/train_bond.yml --device cuda:0 --logdir ./bond_logs
python scripts/sample.py --outdir ./outputs --config ./configs/sample/sample.yml --device cuda:0
#python scripts/evaluate.py --config ./configs/eval/eval.yml

