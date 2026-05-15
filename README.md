# Project_PAW

A small Python project for experimenting with **Poisson Regression** on synthetic datasets.  
The repository generates sample data, trains a custom Poisson regression model with several optimizers, and saves training/validation results and loss curves.

## Overview

`Project_PAW` compares multiple optimization strategies on two generated datasets:

- Gradient Descent
- Regularization-based update
- Adam
- AdamW
- AdamW with adaptive decay

For each dataset, the project:
1. Loads the CSV file
2. Splits it into training, validation, and test sets
3. Trains a Poisson regression model
4. Evaluates the final model on the test set
5. Writes metrics to the `results/` folder
6. Plots loss curves as PDF files

## Repository structure

```
Project_PAW/
├── src/
│   ├── dataset.py          # Dataset generation and loading
│   ├── model.py            # PoissonRegression model (compile, fit, evaluate)
│   └── utils.py            # Optimizer implementations
├── dataset/
│   ├── ds1.csv             # Small dataset (10 features, 1000 samples)
│   └── ds2.csv             # Large dataset (100 features, 10000 samples)
├── results/
│   ├── dataset1_results.txt
│   └── dataset2_results.txt
├── generate_dataset.py     # Script to generate synthetic datasets
├── generate_graphs.py      # Script to plot training/validation loss curves
├── main.py                 # Main training and evaluation script
├── requirement.txt
└── LICENSE
```

## Requirements

The code uses:

- `pandas`
- `torch`
- `matplotlib`


## How to run

### Installation

```bash
git clone https://github.com/DarkerNemesis14/Project_PAW.git
cd Project_PAW
pip install -r requirement.txt
```

### Generate the datasets

This creates:

- `dataset/ds1.csv`
- `dataset/ds2.csv`

```bash
python generate_dataset.py
```

### Train and evaluate the models

This runs Poisson regression on both datasets and saves test loss plus per-epoch training and validation losses.

```bash
python main.py
```

### Generate the loss graphs

This reads the saved result files and exports PDF plots in `results/`.

```bash
python generate_graphs.py
```

## Output files

After running the scripts, you should see files like:

- `results/dataset1_results.txt`
- `results/dataset2_results.txt`
- `results/dataset1_loss.pdf`
- `results/dataset2_loss.pdf`

## Dataset details

The synthetic datasets are generated with random features and Poisson-distributed targets:

- features are sampled from `[-1, 1]`
- target counts are generated from a Poisson distribution


## License

This project is released under the GPL-3.0 license.
