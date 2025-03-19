#!/bin/bash

#SBATCH --account=pr_117_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=147:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=e2e_image
#SBATCH --output=e2e6imges_img_feature_train_new.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs9193@nyu.edu

module purge

singularity exec --nv --overlay /scratch/cc7287/environments/habitat7.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "source /scratch/cc7287/env.sh && conda activate E2EParking && (cd /home/rs9193/ && python gpu_test.py > /dev/null 2>&1 &) && (cd /scratch/rs9193/ParkWithUncertainty/ && python pl_train.py )"
