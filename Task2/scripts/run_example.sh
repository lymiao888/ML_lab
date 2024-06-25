#!/bin/bash

script_dir=$(dirname "$0")

mkdir -p "$script_dir/../output"
python -u "$script_dir/../src/example.py" > "$script_dir/../output/output_example.txt"

