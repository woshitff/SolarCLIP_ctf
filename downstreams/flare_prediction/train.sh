#!/bin/bash

# python ./downstreams/flare_prediction/train.py --config_dir no

# pip install tensorboard
pip install pytorch_lightning
pip install omegaconf
export PLATFORM=TQ
python -u -m downstreams.flare_prediction.train --config_dir no 2>&1 | tee /mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/logs/logs/train.log