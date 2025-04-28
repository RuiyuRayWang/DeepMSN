# DeepMSN

This project implements the DeepMSN model using PyTorch.

## Directory Structure

```
DeepMSN/
├── datasets/               # Dataset storage
├── models/             # Model definitions
├── notebooks/          # Jupyter notebooks for experiments
├── scripts/            # Training and evaluation scripts
├── utils/              # Utility functions
├── tests/              # Unit tests
└── README.md           # Project description
```

## Getting Started

1. Install dependencies:
Refer to https://pytorch.org/get-started/ for instructions to install pytorch.

2. Run tests:
   ```bash
   conda activate deepmsn
   python -m unittest test.test_data_loader
   
   # Run single test case
   python -m unittest tests.test_data_loader.TestDummyDataLoader.test_load_dummy_data

   (deepmsn) luolab@luolab-X11DAi-N:~/GITHUB_REPOS/DeepMSN$ python -m unittest test.test_deepmsn_model
   ```

3. Run training:
   ```bash
   python scripts/train.py
   ```

4. Evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`
