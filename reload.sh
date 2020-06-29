#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate openmm

cd devtools/conda-envs/
conda env update -n openmm --file test_env.yaml
cd ../../

python setup.py install