#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --job-name=My_project_rtx8000
#SBATCH --output=e2e6imges_baseline_train_rtx8000_all.out

module purge

singularity exec --nv --overlay /scratch/cc7287/environments/traj/habitat5.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /scratch/cc7287/env.sh; conda activate E2EParking; cd /vast/cc7287/trajectory/ParkWithUncertainty_develop_shunyu_refine2_all; python pl_train.py"
            
