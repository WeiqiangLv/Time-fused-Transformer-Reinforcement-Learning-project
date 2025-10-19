# OTFS Communication System with GPT-based Decision Making

This repository implements an Orthogonal Time Frequency Space (OTFS) communication system integrated with a GPT-based decision transformer for intelligent modulation scheme selection in drone communication scenarios.

## Overview

The project combines:
- **OTFS Modulation**: Advanced modulation technique for high-mobility wireless communications
- **Decision Transformer**: GPT-based model for learning optimal communication strategies
- **Environment Simulation**: Drone communication environment with varying SNR and height conditions

## Features

- OTFS vs OFDM modulation comparison
- Reinforcement learning-based modulation selection
- Support for different SNR levels (20, 24, 28, 32, 36, 40 dB)
- Variable height scenarios (100m, 1000m, 10000m)
- Real-time BER (Bit Error Rate) monitoring
- GPU acceleration support

## Project Structure

```
OTFS_git/
├── run_me.py                    # Main execution script
├── environment.py               # Communication environment simulation
├── OTFS_modulator.py           # OTFS modulation implementation
├── mingpt/                     # GPT model implementation
│   ├── __init__.py
│   ├── model_atari.py          # GPT model architecture
│   ├── trainer_atari.py        # Training logic
│   └── utils.py                # Utility functions
├── mingpt_repository_from_github/  # Original MinGPT source
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for acceleration)
- Anaconda or Miniconda (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd OTFS_git
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n otfs-env python=3.8
   conda activate otfs-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Windows-specific Setup

If you encounter conda activation issues on Windows:

1. Initialize conda for PowerShell:
   ```powershell
   conda init powershell
   ```

2. Restart your terminal and activate the environment:
   ```powershell
   conda activate otfs-env
   ```

3. Run with full Python path if needed:
   ```powershell
   & "C:\Users\$env:USERNAME\anaconda3\envs\otfs-env\python.exe" run_me.py
   ```

## Usage

### Basic Usage

Run the main script with default parameters:
```bash
python run_me.py
```

### Configuration

The system can be configured by modifying parameters in `run_me.py`:

- **SNR**: Signal-to-Noise Ratio (default: 20 dB)
- **Height**: Communication height (default: 10000m)
- **Training epochs**: Number of training iterations (default: 20)
- **Batch size**: Training batch size (default: 128)

### Custom Data Input

To use your own communication data:

1. **Prepare your data**: Ensure your data is in the correct format (time series of communication states)

2. **Update environment.py**: Replace the sample data generation in `run_me.py`:
   ```python
   # Replace this section with your data loading
   sample_map = your_state_data  # Shape: (1700, 4, 84, 84) or similar
   sample_speed = your_speed_data  # Shape: (1700,)
   e.over(sample_map, sample_speed)
   ```

3. **Adjust parameters**: Update `done_idxs`, `timesteps`, and other parameters according to your data structure.

## Model Architecture

### GPT Configuration
- **Layers**: 6 transformer layers
- **Heads**: 8 attention heads  
- **Embedding dimension**: 128
- **Model type**: Reward-conditioned decision transformer
- **Context length**: 90 timesteps

### Training Configuration
- **Learning rate**: 6e-4 with decay
- **Warmup tokens**: 10,240
- **Final tokens**: Based on dataset size
- **Optimizer**: AdamW

## Environment Details

### Communication Actions
- **Action 0**: Keep current modulation
- **Action 1**: Switch to OFDM (32 subcarriers)
- **Action 2**: Keep current modulation  
- **Action 3**: Switch to OTFS (32 subcarriers)
- **Action 4**: Switch to OTFS (64 subcarriers)

### Reward Structure
The system provides rewards based on:
- Bit Error Rate (BER) performance
- Modulation efficiency
- Communication reliability

### State Representation
States include:
- Channel conditions
- SNR measurements
- Mobility parameters
- Historical performance data

## Performance Monitoring

The system tracks:
- **Training loss**: Model learning progress
- **BER values**: Communication quality metrics
- **Reward signals**: Decision quality indicators
- **Action distributions**: Strategy analysis

### Output Files

The system automatically creates an `output/` directory and saves results as Excel files:
- `BER_all_SNR_HEIGHT.xlsx`: Bit Error Rate results
- `R_all_SNR_HEIGHT.xlsx`: Reward signals over time
- `L_all_SNR_HEIGHT.xlsx`: Training loss progression

Where SNR and HEIGHT are the configured values (e.g., `BER_all_20_10000.xlsx`).

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: 'mingpt'**
   - Ensure the mingpt package is properly installed
   - Check that `__init__.py` exists in the mingpt directory

2. **CUDA warnings**
   - These are normal if you don't have CUDA installed
   - The system will automatically fall back to CPU computation

3. **Memory issues**
   - Reduce batch size in training configuration
   - Use smaller datasets for initial testing

4. **Conda activation problems**
   - Run `conda init` for your shell
   - Restart terminal after initialization

5. **File path errors (OSError: Cannot save file)**
   - ✅ **FIXED**: The system now uses local `output/` directory
   - Ensure write permissions in the current directory
   - Run `python test_output.py` to verify output functionality
   - See `FIXES_APPLIED.md` for detailed information

6. **Excel file issues**
   - Ensure `openpyxl` is installed: `pip install openpyxl`
   - Check pandas version: `pip install pandas>=1.4.0`

### Performance Tips

- Use GPU acceleration when available
- Adjust batch size based on available memory
- Monitor training progress through loss curves
- Use tensorboard for detailed training visualization

## Dependencies

Core dependencies include:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- SciPy (scientific computing)
- scikit-commpy (communication algorithms)
- matplotlib (visualization)
- tqdm (progress bars)

See `requirements.txt` for complete dependency list with versions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={OTFS Communication with GPT-based Decision Making},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Contact

[Add your contact information here]

## Acknowledgments

- MinGPT implementation by Andrej Karpathy
- OTFS modulation research community
- PyTorch development team