#!/bin/bash

# Train on Part 1
python training/train.py -c ../configs/Set1_finetune_part1.yaml
mv sam2_logs sam2_logs_1

# Train on Part 2
python training/train.py -c ../configs/Set1_finetune_part2.yaml
mv sam2_logs sam2_logs_2

# Train on Part 3
python training/train.py -c ../configs/Set1_finetune_part3.yaml
mv sam2_logs sam2_logs_3

# Train on Part 4
python training/train.py -c ../configs/Set1_finetune_part4.yaml
mv sam2_logs sam2_logs_4

# Train on Part 5
python training/train.py -c ../configs/Set1_finetune_part5.yaml
mv sam2_logs sam2_logs_5

# Train on Part 6
python training/train.py -c ../configs/Set1_finetune_part6.yaml
mv sam2_logs sam2_logs_6


