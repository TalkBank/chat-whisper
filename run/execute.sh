#!/bin/bash

sbatch -t 40:00:00 -p GPU-shared --gpus=v100-32:1 main.job

