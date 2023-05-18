#!/bin/bash

sbatch -p GPU-shared --gpus=v100-32:1 main.job

