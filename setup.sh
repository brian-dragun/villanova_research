#!/bin/bash

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install pybind11-dev
sudo apt-get install libopenmpi-dev


# Configure Git settings
echo "Configuring Git..."
git config --global credential.helper store
git config --global user.email "bdragun@villanova.edu"
git config --global user.name "Brian Dragun"

# Install Python dependencies
echo "Installing requirements..."
pip install -r requirements.txt
pip uninstall numpy
pip install numpy<2.0

# Set CPLUS_INCLUDE_PATH for pybind11 (optional)
echo "Setting CPLUS_INCLUDE_PATH for pybind11..."
export CPLUS_INCLUDE_PATH=$(python -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':')

# HuggingFace Login
echo "Logging into Hugging Face..."
# Automatically provide the token and answer "y" for git credential prompt.
# The token is: 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
#echo -e "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\ny" | huggingface-cli login

huggingface-cli login

echo "Setup complete."