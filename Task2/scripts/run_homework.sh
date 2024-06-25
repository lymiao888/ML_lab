#!/bin/bash

script_dir=$(dirname "$0")
mkdir -p "$script_dir/../output"
python -u "$script_dir/../src/homework.py" "$script_dir/../dataset/house.csv" \
        --batch_size 64 \
        --epochs 16 \
        > "$script_dir/../output/output_homework.txt" 