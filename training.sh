#!/bin/bash
srun -p NTU --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=DEEPBLENDER --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45  \
python main.py -c run_configs/davis.yaml --num_workers 0 --task train
#--pty bash -i
