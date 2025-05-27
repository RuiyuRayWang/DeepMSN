# DeepFlyBrain

This project implements the DeepFlyBrain model using PyTorch.

## Directory Structure

```
DeepFlyBrain/
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
   ```bash
   $ conda activate deepmsn
   (deepmsn) $ pip install -r requirements.txt 
   (deepmsn) $ conda install bedtools
   ```

2. Preprocess data:
   Edit `configs/config.yaml`
   
   Do preprocessing in command line:
   ```bash
   (deepmsn)$ python -m scripts.preprocess -c configs/config.yaml --sort
   ```

3. # TODO: Run training:
   ```bash
   python train.py
   ```

4. # TODO: Evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`
