#!/usr/bin/env python3
"""
Setup script for OTFS Communication System with GPT-based Decision Making

This script helps set up the environment and dependencies for the OTFS project.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ Success: {description or command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description or command}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_conda():
    """Check if conda is available."""
    result = run_command("conda --version", "Checking conda installation")
    return result is not None

def setup_conda_environment():
    """Set up conda environment."""
    env_name = "otfs-env"
    
    # Check if environment already exists
    result = run_command(f"conda env list | grep {env_name}", "Checking existing environment")
    
    if result and env_name in result:
        print(f"✓ Conda environment '{env_name}' already exists")
        response = input(f"Do you want to recreate the environment? (y/N): ")
        if response.lower() == 'y':
            run_command(f"conda env remove -n {env_name}", f"Removing existing environment")
        else:
            return env_name
    
    # Create new environment
    run_command(f"conda create -n {env_name} python=3.8 -y", 
                f"Creating conda environment '{env_name}'")
    
    return env_name

def install_dependencies(use_conda=True, env_name=None):
    """Install project dependencies."""
    if use_conda and env_name:
        # Install with conda environment
        if platform.system() == "Windows":
            activate_cmd = f"conda activate {env_name} && "
        else:
            activate_cmd = f"source activate {env_name} && "
        
        pip_cmd = f"{activate_cmd}pip install -r requirements.txt"
    else:
        # Install with system pip
        pip_cmd = "pip install -r requirements.txt"
    
    run_command(pip_cmd, "Installing Python dependencies")

def setup_mingpt_package():
    """Set up the mingpt package structure."""
    mingpt_dir = "mingpt"
    
    if not os.path.exists(mingpt_dir):
        os.makedirs(mingpt_dir)
        print(f"✓ Created {mingpt_dir} directory")
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(mingpt_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# MinGPT package for OTFS communication system\n")
        print(f"✓ Created {init_file}")
    
    # Copy files from mingpt_repository_from_github if they don't exist
    source_dir = "mingpt_repository_from_github"
    if os.path.exists(source_dir):
        for filename in ["model_atari.py", "trainer_atari.py", "utils.py"]:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(mingpt_dir, filename)
            
            if os.path.exists(source_file) and not os.path.exists(target_file):
                import shutil
                shutil.copy2(source_file, target_file)
                print(f"✓ Copied {filename} to mingpt package")

def print_usage_instructions(use_conda=True, env_name=None):
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    
    if use_conda and env_name:
        print(f"\nTo use the project:")
        print(f"1. Activate the conda environment:")
        print(f"   conda activate {env_name}")
        print(f"2. Run the main script:")
        print(f"   python run_me.py")
        
        if platform.system() == "Windows":
            print(f"\nOn Windows, if you have conda issues, you can also run:")
            print(f'   & "C:\\Users\\$env:USERNAME\\anaconda3\\envs\\{env_name}\\python.exe" run_me.py')
    else:
        print(f"\nTo use the project:")
        print(f"   python run_me.py")
    
    print(f"\nFor more information, see README.md")

def main():
    """Main setup function."""
    print("OTFS Communication System Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we should use conda
    use_conda = check_conda()
    env_name = None
    
    if use_conda:
        response = input("Use conda environment? (Y/n): ")
        if response.lower() != 'n':
            env_name = setup_conda_environment()
        else:
            use_conda = False
    else:
        print("Conda not found, using system Python")
        use_conda = False
    
    # Install dependencies
    install_dependencies(use_conda, env_name)
    
    # Set up mingpt package
    setup_mingpt_package()
    
    # Print usage instructions
    print_usage_instructions(use_conda, env_name)

if __name__ == "__main__":
    main()