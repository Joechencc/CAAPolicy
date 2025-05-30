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

OVERLAYS="$SCRATCH/ENV/overlay-50G-10M.ext3:ro,$SCRATCH/squashfs/Town_Opt_1000_Val.sqf:ro"
for sqf in /scratch/rs9193/Town_Opt_1000/*.sqf; do
  OVERLAYS+=",${sqf}:ro"
done

# 2. Launch Singularity
singularity exec --nv \
  --overlay "$OVERLAYS" \
  --bind    /scratch/rs9193/ENV/e2e_carla:/opt/e2e_carla \
  --bind    /scratch/cc7287/env.sh:/opt/env.sh \
  /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
  /bin/bash -lc '
    source /opt/env.sh
    conda activate E2EParking

    # fire off the GPU test in background
    (cd /home/rs9193 && python gpu_test.py > /dev/null 2>&1 &)

    # then launch your training
    cd /scratch/rs9193/e2e-attention/Park_attention/
    python pl_train.py \
      --model_path /scratch/rs9193/e2e/ckpt/al0.005-2/exp_2025_5_18_8_51_47/last.ckpt
  '
#model path: --model_path ./ckpt/exp_2025_3_23_8_15_35/last.ckpt
#time
# attention loss weight
#job name
#output file name
