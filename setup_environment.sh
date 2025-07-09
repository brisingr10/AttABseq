#!/bin/bash

# Create a new conda environment with Python 3.7
conda create -n attabseq python=3.7 -y

# Activate the environment
source activate attabseq

# Install dependencies
conda install pytorch=1.7.0 torchvision=0.8.0 -c pytorch -y
conda install numpy=1.21.5 -y
conda install pandas=1.3.5 -y
conda install scikit-learn=1.0.2 -y
conda install scipy=1.7.3 -y
conda install seaborn=0.12.2 -y
conda install matplotlib=3.5.3 -y
conda install networkx=2.6.3 -y
conda install xarray -y

echo "Conda environment 'attabseq' created and packages installed successfully."
echo "To activate the environment, run: conda activate attabseq"
