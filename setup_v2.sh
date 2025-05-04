#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color variables
YELLOW='\033[1;33m'
NC='\033[0m' # No Color (resets the color back to normal)

# Hugging Face token
export HF_TOKEN="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"

# Configure Git settings
echo -e "${YELLOW}Configuring Git...${NC}"
git config --global credential.helper store
git config --global user.email "bdragun@villanova.edu"
git config --global user.name "Brian Dragun"

echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

echo -e "${YELLOW}Installing dependencies...${NC}"
sudo apt-get install -y pybind11-dev libopenmpi-dev

# Set CPLUS_INCLUDE_PATH for pybind11 (optional)
echo -e "${YELLOW}Setting CPLUS_INCLUDE_PATH for pybind11...${NC}"
export CPLUS_INCLUDE_PATH=$(python -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':')

# Install Python dependencies
echo -e "${YELLOW}Installing Python requirements...${NC}"
pip install -r requirements.txt
pip uninstall -y numpy
pip install 'numpy<2.0'

# Ensure the Hugging Face Hub library is installed
echo -e "${YELLOW}Ensuring huggingface_hub is installed...${NC}"
pip install huggingface_hub --upgrade

# Hugging Face Login using Python API
echo -e "${YELLOW}Logging into Hugging Face using Python API...${NC}"
python3 - <<EOF
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")  # Retrieve token from environment
if token:
    login(token)
    print("Hugging Face login successful!")
else:
    print("Error: HF_TOKEN is not set.")
EOF

echo -e "${YELLOW}Setup complete.${NC}"


# Install oh-my-posh and add executable permission.
sudo wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64 -O /usr/local/bin/oh-my-posh
sudo chmod +x /usr/local/bin/oh-my-posh

# Setup themes and set permission.
mkdir ~/.poshthemes
wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/themes.zip -O ~/.poshthemes/themes.zip
unzip ~/.poshthemes/themes.zip -d ~/.poshthemes
chmod u+rw ~/.poshthemes/*.json

# Clean up files.
rm ~/.poshthemes/themes.zip

# Setup in profile in when load terminal. (use theme bash as example.)
echo 'eval "$(oh-my-posh init bash)"' >> ~/.profile