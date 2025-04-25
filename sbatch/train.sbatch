#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=ctrl_group
#SBATCH --mail-type=END
##SBATCH --mail-user=yh4007@nyu.edu
#SBATCH --output=e2e6imges_baseline_train.out

module purge

singularity exec --nv --overlay /scratch/yh4007/e2ebaseline/E2EParking.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
            /bin/bash -c "source /ext3/env.sh; conda activate E2EParking; cd /scratch/yh4007/e2ebaseline/ParkWithUncertainty-e2eBaseline_6_images_waypointHeader_triplet_control;python pl_train.py"
