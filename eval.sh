#!/bin/bash
#srun -p NTU --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=DEEPBLENDER --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45  \
    python main.py -c run_configs/davisGAN.yaml --task infer --wts saved_models/deepblender_gan_inpaint/model_best_train.pth
#python main.py -c run_configs/davisGAN.yaml --task infer --wts saved_models/deepblender_gan/checkpoint_35100.pth
#--pty bash -i
