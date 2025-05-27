#!/bin/bash

#SBATCH --account=pr_117_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=dynamics
#SBATCH --output=dynamics.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs9193@nyu.edu

module purge

singularity exec --nv --overlay $SCRATCH/ENV/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "source /scratch/cc7287/env.sh && conda activate E2EParking && (cd /home/rs9193/ && python gpu_test.py > /dev/null 2>&1 &) && (cd /scratch/rs9193/park_test/ParkWithUncertainty/ && python pl_dynamics_train.py)"

#model path: --model_path ./ckpt/exp_2025_3_23_8_15_35/last.ckpt
#time
# attention loss weight
#job name
#output file name