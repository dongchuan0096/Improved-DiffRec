# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a recommendation system based on PyTorch, which utilizes a Diffusion Model and an Autoencoder. The main logic for training and evaluation is contained within the `LT-DiffRec/` directory.

## How to Run

The main script to run experiments is `LT-DiffRec/run.sh`. It takes numerous command-line arguments to configure the dataset, model hyperparameters, and training settings.

Example usage from `run.sh`:
```bash
sh LT-DiffRec/run.sh amazon-book_clean 5e-4 1e-4 0 0 400 2 [300] [] 0.03 [300] 10 x0 5 0.7 0.001 0.005 0 1 0.1 1.0 log 1 0
```

The script invokes `LT-DiffRec/main.py` with the provided arguments.

## Code Structure

- **`LT-DiffRec/main.py`**: The main entry point for training and evaluating the model. It handles argument parsing, data loading, model initialization, and the training loop.
- **`LT-DiffRec/models/`**: This directory contains the core model definitions:
    - **`Autoencoder.py`**: Defines the Autoencoder model used for representation learning.
    - **`DNN.py`**: Defines the MLP/DNN model.
    - **`gaussian_diffusion.py`**: Implements the Gaussian Diffusion process.
- **`LT-DiffRec/data_utils.py`**: Contains utility functions for loading and preprocessing the data.
- **`LT-DiffRec/evaluate_utils.py`**: Contains functions for computing evaluation metrics like Top-N accuracy.
- **`datasets/`**: This directory is intended to store the datasets used for training and evaluation.
