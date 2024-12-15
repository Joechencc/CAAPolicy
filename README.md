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

Run the following files to visualize new data:
```Shell
python carla_data_gen.py
```
Main changes are within path_collector_TF_Dec13.py, hybrid_A_star_TF_Dec13.pyx, data_generator.py, and carla_data_gen.py. Please make sure these files are updated, as well as the direcotries used in those files.

**Note**: The updated files should be able to generate good movements based on our tests. However, there might be small chances that the code may encounter fine_tunning. If the code enters the fine_tunning conditions, please don't save such specific data. Skip saving and generate the new one. 
