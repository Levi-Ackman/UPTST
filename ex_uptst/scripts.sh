#!/bin/bash

# 'prediction wothout ano'
# Define parameter values
seeds=(2021 2022 2023 2024)
seq_opt=(96)
pre_opt=(96 192 336 720)
loss_fns=("mse")
batch_sizes=(16 32)
alphas=(1.0 0.7 0.6 0.5 0.4 0.3)

# Use nested loops to iterate through all parameter combinations
for seed in "${seeds[@]}"; do
    for pre_len in "${pre_opt[@]}"; do
        for seq_len in "${seq_opt[@]}"; do
            for loss_fn in "${loss_fns[@]}"; do
                for bs in "${batch_sizes[@]}"; do
                    for alpha in "${alphas[@]}"; do
                            # Run the Python script and pass the parameters
                            python run.py \
                                --seed "$seed" \
                                --seq_len "$seq_len" \
                                --pre_len  "$pre_len" \
                                --loss_fn "$loss_fn" \
                                --bs "$bs" \
                                --alpha "$alpha" \
                                
                    done
                done
            done
        done
    done
done
