#!/bin/bash

# 'predict with another'
# Define parameter values
seeds=( 2021 2022 2023 2024)
seq_opt=(48 96 192)
pre_opt=(7 14 28 84  )
loss_fns=("mse" "mae")
batch_sizes=(64 32)

# Use nested loops to iterate through all parameter combinations
for seed in "${seeds[@]}"; do
    for pre_len in "${pre_opt[@]}"; do
        for seq_len in "${seq_opt[@]}"; do
            for loss_fn in "${loss_fns[@]}"; do
                for bs in "${batch_sizes[@]}"; do
                            # Run the Python script and pass the parameters
                            python run.py \
                                --model  Autoformer\
                                --seed "$seed" \
                                --seq_len "$seq_len" \
                                --pred_len  "$pre_len" \
                                --loss "$loss_fn" \
                                --batch_size "$bs" \
                                --tasks S \
                                --enc_in 1 \
                                --dec_in 1 \
                                --c_out 1 \
                                --e_layers 2 \
                                --d_layers 1 \
                                
                    done
                done
            done
        done
    done
done
