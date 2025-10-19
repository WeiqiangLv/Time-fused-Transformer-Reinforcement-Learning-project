# Installation Guide

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the setup script:
```bash
python setup.py
```

This will:
- Check your Python version
- Set up a conda environment (if conda is available)
- Install all dependencies
- Configure the mingpt package
- Provide usage instructions

### Option 2: Manual Setup

1. **Create conda environment** (recommended):
   ```bash
   conda create -n otfs-env python=3.8
   conda activate otfs-env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; import numpy; import scipy; print('All dependencies installed successfully!')"
   ```

## Platform-Specific Instructions

### Windows

If you encounter conda activation issues:

1. **Initialize conda for PowerShell**:
   ```powershell
   conda init powershell
   ```

2. **Restart your terminal** and then:
   ```powershell
   conda activate otfs-env
   python run_me.py
   ```

3. **Alternative method** (if conda activation fails):
   ```powershell
   & "C:\Users\$env:USERNAME\anaconda3\envs\otfs-env\python.exe" run_me.py
   ```

### Linux/macOS

Standard conda commands should work:
```bash
conda activate otfs-env
python run_me.py
```

## Troubleshooting

### Common Issues

1. **"conda: command not found"**
   - Install Anaconda or Miniconda from https://docs.conda.io/en/latest/miniconda.html
   - Or use system Python with `pip install -r requirements.txt`

2. **"No module named 'mingpt'"**
   - Run the setup script: `python setup.py`
   - Or manually create the mingpt package structure

3. **CUDA warnings**
   - These are normal if you don't have a CUDA-capable GPU
   - The system will automatically use CPU

4. **Memory errors**
   - Reduce batch_size in run_me.py (line with `batch_size=128`)
   - Close other applications to free up RAM

### Dependency Conflicts

If you encounter dependency conflicts:

1. **Create a fresh environment**:
   ```bash
   conda create -n otfs-env-fresh python=3.8
   conda activate otfs-env-fresh
   pip install -r requirements.txt
   ```

2. **Use pip only** (if conda causes issues):
   ```bash
   python -m venv otfs-venv
   source otfs-venv/bin/activate  # On Windows: otfs-venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Verification

After installation, verify everything works:

```bash
# Activate your environment
conda activate otfs-env  # or your chosen environment name

# Test imports
python -c "
import torch
import numpy as np
import scipy
from mingpt.utils import set_seed
from environment import ENV
print('✓ All core modules imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Run a quick test
python run_me.py
```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Any modern multi-core processor

### Recommended for Better Performance
- **RAM**: 16GB or more
- **GPU**: CUDA-capable GPU with 4GB+ VRAM
- **CPU**: Intel i7/AMD Ryzen 7 or better

## Next Steps

After successful installation:

1. **Read the README.md** for detailed usage instructions
2. **Customize parameters** in `run_me.py` for your use case
3. **Prepare your data** following the data format guidelines
4. **Run the system** and monitor the training progress

## Getting Help

If you encounter issues not covered here:

1. Check the main README.md for detailed documentation
2. Verify your Python and dependency versions
3. Try the automated setup script: `python setup.py`
4. Create an issue in the repository with:
   - Your operating system
   - Python version (`python --version`)
   - Error messages (full traceback)
   - Steps you've already tried