#!/usr/bin/bash
# CUDA_VISIBLE_DEVICES=0,1 python train_scripts/train.py --b 'configs/train_configs/reconmodels/autoencoder/vqvae/vqgan/aia00942aia0094_vqgan_vqvae2.yaml'
#CUDA_VISIBLE_DEVICES=0,1 python train_scripts/train.py --b 'configs/train_configs/reconmodels/autoencoder/vqvae/vqgan/aia00942aia0094_vqgan_vqvae2_nobook.yaml'
#CUDA_VISIBLE_DEVICES=0,1 python train_scripts/train.py --b /mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/zyz_106/configs/train_configs/reconmodels/ldm/mainconfig/zyz/zyz.yaml
# python train_scripts/train.py --b '/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/zyz_106/configs/train_configs/reconmodels/autoencoder/vae/aia0094Taia0094_cnnvae_v2.yaml'
python train_scripts/train.py --b '/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/zyz_106/configs/train_configs/reconmodels/ldm/mainconfig/zyz/zyz_16.yaml'