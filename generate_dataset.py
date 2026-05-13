# include src to search path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# import necessary modules
from dataset import Dataset

# define variables
DS_PATH1 = "dataset/ds1.csv"
DS_PATH2 = "dataset/ds2.csv"

# generate dataset
dt = Dataset()
dt.generate_dataset(10, 1000, DS_PATH1)
dt.generate_dataset(100, 10000, DS_PATH2)