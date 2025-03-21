#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=My_project
#SBATCH --output=e2e6imges_baseline_train.out

module purge

singularity exec --nv --overlay /scratch/sy3913/overlay_images/ParkingWithUncertainty3.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; conda activate E2EParking; cd /scratch/sy3913/e2eBaseline/ParkWithUncertainty;python pl_train.py --model_path /scratch/sy3913/e2eBaseline/ParkWithUncertainty/ckpt/exp_2025_1_26_2_9_21/E2EParking-epoch\=11-val_loss\=1.06.ckpt "
            
