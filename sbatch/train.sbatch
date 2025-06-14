#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=attention_fine
#SBATCH --output=e2e6imges_baseline_train_a100.out

module purge

OVERLAYS="/scratch/cc7287/environments/region/habitat6.ext3:rw,/vast/cc7287/Town_Opt_1000_Val.sqf"

for sqf in /vast/cc7287/Town_Opt_1000/*.sqf; do
  OVERLAYS+=",${sqf}:ro"
done 

singularity exec --nv --overlay "$OVERLAYS" /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /scratch/cc7287/env.sh; conda activate E2EParking; cd /vast/cc7287/trajectory/ParkWithUncertainty_spatial_temporal_original; python pl_train.py --model_path_dynamics ckpt/dynamic_model_speed.ckpt"
            
