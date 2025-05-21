#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs9193@nyu.edu
#SBATCH --account=pr_117_tandon_advanced
#SBATCH --job-name=carla_eva
#SBATCH --output=carla_eva_%j.out
#SBATCH --error=carla_eva_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

set -euo pipefail

### ———————— Configuration ————————
IMG=$SCRATCH/carla-0.9.11.sif
OVL_SERVER=$SCRATCH/carla_cache_eva.ext3 # server’s shader-cache overlay
OVL_CLIENT=$SCRATCH/ENV/overlay-25GB-500K.ext3 # client’s Conda overlay
CACHE_DIR=/tmp/ue_ddc                          # inside the container

RPC_PORT=3000    # ← change this to your desired RPC port
WORLD_PORT=3000  # ← change this to your desired world port
### ———————————————————————————————————

echo "[$(date)] Launching CARLA server on RPC $RPC_PORT / WORLD $WORLD_PORT …"

singularity exec --nv --overlay "$OVL_SERVER" "$IMG" \
  bash -c "mkdir -p $CACHE_DIR"

singularity exec --nv \
  --overlay "${OVL_SERVER}:ro" \
  "${IMG}" \
  bash -c "
    export SDL_VIDEODRIVER=offscreen
    export UE-LocalDataCachePath=$CACHE_DIR
    /home/carla/CarlaUE4.sh \
      -opengl \
      -carla-server \
      -RenderOffScreen \
      -carla-rpc-port=$RPC_PORT \
      -world-port=$WORLD_PORT \
      -nosound -log
  " &

SERVER_PID=$!
echo "CARLA server PID: $SERVER_PID"
echo "Sleeping 3 min to let the server warm up…"
sleep 200

echo "[$(date)] Running parking evaluation"

singularity exec --nv \
  --overlay "${OVL_CLIENT}":ro \
  --bind    /scratch/rs9193/ENV/e2e_carla:/opt/e2e_carla \
  "$IMG" bash -lc "

    # 1) point at your pre-built Conda env
    export PATH=/opt/e2e_carla/bin:\$PATH
    export PYTHONPATH=/opt/e2e_carla/lib/python3.7/site-packages:/home/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:\$PYTHONPATH
    export LD_LIBRARY_PATH=/opt/e2e_carla/lib:\$LD_LIBRARY_PATH

    # 2) run your evaluation script
    python3 /scratch/rs9193/e2e-attention/Park_attention/carla_parking_eva.py --port $RPC_PORT
  "

# singularity exec --nv --overlay "${OVL_CLIENT}:ro" "$IMG" \
#   bash -lc "
#     source /ext3/env.sh && conda activate /scratch/rs9193/ENV/e2e_carla
#     python /scratch/rs9193/e2e-attention/Park_attention/carla_parking_eva.py \
#       --port $RPC_PORT
#   "

echo "[$(date)] Client finished; stopping CARLA server (PID $SERVER_PID)…"
kill $SERVER_PID || true

echo "[$(date)] All done."
