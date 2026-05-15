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

```text
.
├── dataset/              # Generated CSV datasets
├── results/              # Saved metrics and plots
├── src/
│   ├── dataset.py        # Synthetic dataset generation and loading
│   ├── model.py          # Poisson regression model
│   └── utils.py          # Optimizers / update rules
├── generate_dataset.py   # Creates the datasets
├── generate_graphs.py    # Builds loss graphs from saved results
├── main.py               # Trains and evaluates all models
├── requirement.txt       # Python dependencies
└── LICENSE
```

## Requirements

The code uses:

- `pandas`
- `torch`
- `matplotlib`

Install them with:

```bash
pip install pandas torch matplotlib
```

## How to run

### 1) Generate the datasets

This creates:

- `dataset/ds1.csv`
- `dataset/ds2.csv`

```bash
python generate_dataset.py
```

### 2) Train and evaluate the models

This runs Poisson regression on both datasets and saves test loss plus per-epoch training and validation losses.

```bash
python main.py
```

### 3) Generate the loss graphs

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
- target counts are generated from a Poisson process
- each CSV contains feature columns (`x1`, `x2`, ...) and a target column `y`

## Model details

The model in `src/model.py` implements Poisson regression with:

- manual weight initialization
- Poisson log-likelihood style loss
- prediction via `exp(X @ w)`
- a training loop that updates weights using the selected optimizer

## Notes

- The repository currently uses a custom training loop rather than a high-level ML framework.
- The `requirement.txt` file only lists `pandas`, so you may need to install the other dependencies manually.
- The project is licensed under GPL-3.0.

## License

This project is released under the GPL-3.0 license.
