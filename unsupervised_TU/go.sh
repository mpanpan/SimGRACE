#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  CUDA_VISIBLE_DEVICES=$1 python simgrace.py --DS $2 --lr 0.01 --local --num-gc-layers 5 --eta$3 --seed $seed
done

