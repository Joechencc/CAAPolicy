#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --job-name=continuous_test
#SBATCH --mail-user=yh4007@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=parallel_ACT_8steps.out

module purge

singularity exec --nv --overlay /scratch/yh4007/e2ebaseline/E2EParking.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; conda activate E2EParking; cd /scratch/yh4007/ParkWithUncertainty-continuous;python pl_train.py"