#!/bin/bash

# Configure Git settings
echo "Configuring Git..."
git config --global credential.helper store
git config --global user.email "bdragun@villanova.edu"
git config --global user.name "Brian Dragun"

# Install Python dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# HuggingFace Login
echo "Logging into Hugging Face..."
# Automatically provide the token and answer "y" for git credential prompt.
# The token is: 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
#echo -e "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\ny" | huggingface-cli login

huggingface-cli login

echo "Setup complete."