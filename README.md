# Hybrid A*


## Setup

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```Shell
conda env create -f environment.yml
conda activate E2EParking_9_11
chmod +x setup_carla.sh
./setup_carla.sh
```
Install hybrid A*
```Shell
pip install cython
pip install open3d
pip install heapdict

cd agent
chmod +x quick_setup.sh
./quick_setup
```
CUDA 11.7 is used as default. We also validate the compatibility of CUDA 10.2 and 11.3.

The first step is to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl -RenderOffScreen
```
In a separate terminal, use the script below for generating training data:
```Shell
./data_collect.sh
```

