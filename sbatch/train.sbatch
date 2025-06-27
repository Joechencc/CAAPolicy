#!/bin/bash

#SBATCH --account=pr_110_tandon_priority
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs9193@nyu.edu
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=350GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --job-name=baseline
#SBATCH --output=baseline%j.out
#SBATCH --error=baseline%j.err

module purge
IMG=$SCRATCH/carla-0.9.11.sif

OVERLAYS="$SCRATCH/ENV/overlay-25GB-500K.ext3:ro,/vast/rs9193/Town_Opt_1000_Val/Town_Opt_1000_Val.sqf:ro"
for sqf in /vast/rs9193/Town_Opt_1000/*.sqf; do
  OVERLAYS+=",${sqf}:ro"
done

# singularity exec --nv --overlay "$OVERLAYS"\
#             /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
#             /bin/bash -c "source /scratch/cc7287/env.sh && conda activate E2EParking; cd /scratch/rs9193/park_test/ParkWithUncertainty/ && python pl_train.py"
           
singularity exec --nv \
  --overlay "$OVERLAYS" \
  --bind    /scratch/rs9193/ENV/e2e_carla:/opt/e2e_carla \
  "$IMG" \
  bash -lc "

    # 1) point at your pre-built Conda env
    export PATH=/opt/e2e_carla/bin:\$PATH
    export PYTHONPATH=/opt/e2e_carla/lib/python3.7/site-packages:/home/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:\$PYTHONPATH
    export LD_LIBRARY_PATH=/opt/e2e_carla/lib:\$LD_LIBRARY_PATH

    # 2) run your evaluation script
    cd /scratch/rs9193/park_test/ParkWithUncertainty/ && python3 pl_train.py --model_path ckpt/baseline_and_waypoints/E2EParking-epoch=20-val_loss=1.25.ckpt --exp_name baseline
  "
#--caa --ttm --waypoints