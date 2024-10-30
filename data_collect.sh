#!/bin/bash

for i in {1..8}
do
  timestamp=$(date +%s)  
  python carla_data_gen.py --random_seed $timestamp
done