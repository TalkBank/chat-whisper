#!/bin/bash

sbatch -p GPU-shared --gpus=v100-16:1 main.job

