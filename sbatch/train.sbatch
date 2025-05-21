#!/bin/bash

#SBATCH --account=pr_117_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=160GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --job-name=al0.005
#SBATCH --output=attention_loss_0.005_2gpu.out
#SBATCH --error=attention_loss_0.005_2gpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs9193@nyu.edu

module purge

singularity exec --nv --overlay $SCRATCH/ENV/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "source /scratch/cc7287/env.sh && conda activate E2EParking && (cd /home/rs9193/ && python gpu_test.py > /dev/null 2>&1 &) && (cd /scratch/rs9193/e2e-attention/Park_attention/ && python pl_train.py --model_path /scratch/rs9193/e2e/ckpt/al0.005-2/exp_2025_5_16_1_30_35/last.ckpt)"

#model path: --model_path ./ckpt/exp_2025_3_23_8_15_35/last.ckpt
#time
# attention loss weight
#job name
#output file name
