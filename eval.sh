#!/bin/sh
export CUDA_VISIBLE_DEVICES=1

checkpoint_dir="./$1"
test_output="$checkpoint_dir/eval_test.txt"
train_output="$checkpoint_dir/eval_train.txt"
num_residual_units=$2
k=$3
batch_size=100
test_iter=100
train_iter=500
gpu_fraction=0.96

python eval.py --ckpt_path $checkpoint_dir \
               --output $test_output \
               --num_residual_units $num_residual_units \
               --k $k \
               --batch_size $batch_size \
               --test_iter $test_iter \
               --gpu_fraction $gpu_fraction

python eval.py --ckpt_path $checkpoint_dir \
               --output $train_output \
               --num_residual_units $num_residual_units \
               --k $k \
               --batch_size $batch_size \
               --test_iter $train_iter \
               --gpu_fraction $gpu_fraction \
               --train_data True
