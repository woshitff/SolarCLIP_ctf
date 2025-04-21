#!/bin/bash
export MASTER_ADDR="localhost"  # 必须用 export 导出为环境变量
export MASTER_PORT="29500"      # 同上
python models/reconmodels/autoencoder/models/vae/JointVAE_train_v2.py