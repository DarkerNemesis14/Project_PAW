# include src to search path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# import necessary modules
from dataset import Dataset
from utils import GradientDescent, Regularizer, Adam, Adamw, AdamwDecay
from model import PoissonRegression

# define variables
DS_PATH1 = "dataset/ds1.csv"
DS_PATH2 = "dataset/ds2.csv"
RESULT_PATH = "results/poi_grad.txt"

# load dataset 1
X, y = Dataset().load_dataset(DS_PATH1)

# compile model
pr = PoissonRegression()
pr.compile(AdamwDecay())

# train model
grad_train_loss = pr.fit(X, y, 50)

#evaluate model
grad_test_loss = pr.evaluate(X, y)

# save results
with open(RESULT_PATH, 'w') as file:
    file.write(f"grad_testloss: {grad_test_loss:.4f}\n")
    file.write(f"grad_TrainingLoss: {' '.join(map(str, grad_train_loss.numpy()))}\n")