import subprocess
import os

# Hugging Face token
HF_TOKEN = "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"

def run_command(command, shell=True):
    """Runs a shell command and prints output in real-time."""
    process = subprocess.run(command, shell=shell, check=True, text=True)

def main():
    # Update and upgrade system
    print("Updating system packages...")
    run_command("sudo apt-get update && sudo apt-get upgrade -y")

    # Install required system packages
    print("Installing dependencies...")
    run_command("sudo apt-get install -y pybind11-dev libopenmpi-dev")

    # Configure Git
    print("Configuring Git settings...")
    run_command('git config --global credential.helper store')
    run_command('git config --global user.email "bdragun@villanova.edu"')
    run_command('git config --global user.name "Brian Dragun"')

    # Install Python dependencies
    print("Installing Python requirements...")
    run_command("pip install -r requirements.txt")
    run_command("pip install huggingface_hub")
    run_command("pip uninstall -y numpy")
    run_command("pip install 'numpy<2.0'")

    # Set CPLUS_INCLUDE_PATH for pybind11
    print("Setting CPLUS_INCLUDE_PATH for pybind11...")
    cplus_include_path = subprocess.run("python -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':'", 
                                        shell=True, check=True, capture_output=True, text=True).stdout.strip()
    os.environ["CPLUS_INCLUDE_PATH"] = cplus_include_path

    # Use Python API to log in instead of CLI
    print("Logging into Hugging Face using API...")
    from huggingface_hub import login
    login(HF_TOKEN)  

    print("Setup complete.")

if __name__ == "__main__":
    main()
