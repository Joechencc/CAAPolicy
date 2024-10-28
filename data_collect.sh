#!/bin/bash


for i in {1..8}
do

  python carla_data_gen.py --random_seed $i
done
