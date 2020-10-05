#!/bin/bash
srun -p NTU --mpi=pmi2 --gres=gpu:$1 -n1 --ntasks-per-node=1 --job-name=jupyter --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45  \
python main.py -c run_configs/davis.yaml --task infer --wts saved_models/csn/bmvc_final.pth
#--pty bash -i
