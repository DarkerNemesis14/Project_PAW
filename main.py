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

# define hyperparameters
DS1_EPOCHS = 100
DS1_LR_GRAD = 1e-3
DS1_LR_REGU = 1e-3
DS1_LR_ADAM = 1e-1
DS1_LR_ADAMW = 1e-1
DS1_LR_ADAMWD = 1e-1
DS2_EPOCHS = 100
DS2_LR_GRAD = 1e-7
DS2_LR_REGU = 1e-7
DS2_LR_ADAM = 1e-2
DS2_LR_ADAMW = 1e-2
DS2_LR_ADAMWD = 1e-2

# define paths
DS1_PATH = "dataset/ds1.csv"
DS2_PATH = "dataset/ds2.csv"
DS1_RESULT_PATH = "results/dataset1_results.txt"
DS2_RESULT_PATH = "results/dataset2_results.txt"

# load dataset 1
X, y = Dataset().load_dataset(DS1_PATH)

# define split percentage
TRAIN_PERCENTAGE = 0.6
TEST_PERCENTAGE = 0.2

# split dataset
X_train, y_train = X[ : int(len(X) * TRAIN_PERCENTAGE)], y[ : int(len(y) * TRAIN_PERCENTAGE)]
X_val, y_val = X[int(len(X) * TRAIN_PERCENTAGE) : int(len(X) * (1 - TEST_PERCENTAGE))], y[int(len(X) * TRAIN_PERCENTAGE) : int(len(y) * (1 - TEST_PERCENTAGE))]
X_test, y_test = X[int(len(X) * (1 - TEST_PERCENTAGE)) : ], y[int(len(y) * (1 - TEST_PERCENTAGE)) : ]


# compile model with gradient descent
print("Training with Gradient Descent")
pr_grad = PoissonRegression()
pr_grad.compile(GradientDescent(learning_rate=DS1_LR_GRAD))

# train model
grad_train_loss, grad_val_loss = pr_grad.fit(X_train, y_train, X_val, y_val, DS1_EPOCHS)

#evaluate model
grad_test_loss = pr_grad.evaluate(X_test, y_test)

# save results
with open(DS1_RESULT_PATH, 'w') as file:
    file.write(f"grad_testloss: {grad_test_loss:.4f}\n")
    file.write(f"grad_TrainingLoss: {' '.join(map(str, grad_train_loss.numpy()))}\n")
    file.write(f"grad_ValidationLoss: {' '.join(map(str, grad_val_loss.numpy()))}\n")


# compile model with regularization
print("Training with Regularization")
pr_regu = PoissonRegression()
pr_regu.compile(Regularizer(learning_rate=DS1_LR_REGU))

# train model
regu_train_loss, regu_val_loss = pr_regu.fit(X_train, y_train, X_val, y_val, DS1_EPOCHS)

#evaluate model
regu_test_loss = pr_regu.evaluate(X_test, y_test)

# save results
with open(DS1_RESULT_PATH, 'a') as file:
    file.write(f"regu_testloss: {regu_test_loss:.4f}\n")
    file.write(f"regu_TrainingLoss: {' '.join(map(str, regu_train_loss.numpy()))}\n")
    file.write(f"regu_ValidationLoss: {' '.join(map(str, regu_val_loss.numpy()))}\n")


# compile model with adam
print("Training with Adam")
pr_adam = PoissonRegression()
pr_adam.compile(Adam(learning_rate=DS1_LR_ADAM))

# train model
adam_train_loss, adam_val_loss = pr_adam.fit(X_train, y_train, X_val, y_val, DS1_EPOCHS)

#evaluate model
adam_test_loss = pr_adam.evaluate(X_test, y_test)

# save results
with open(DS1_RESULT_PATH, 'a') as file:
    file.write(f"adam_testloss: {adam_test_loss:.4f}\n")
    file.write(f"adam_TrainingLoss: {' '.join(map(str, adam_train_loss.numpy()))}\n")
    file.write(f"adam_ValidationLoss: {' '.join(map(str, adam_val_loss.numpy()))}\n")


# compile model with adamw
print("Training with AdamW")
pr_adamw = PoissonRegression()
pr_adamw.compile(Adamw(learning_rate=DS1_LR_ADAMW))

# train model
adamw_train_loss, adamw_val_loss = pr_adamw.fit(X_train, y_train, X_val, y_val, DS1_EPOCHS)

#evaluate model
adamw_test_loss = pr_adamw.evaluate(X_test, y_test)

# save results
with open(DS1_RESULT_PATH, 'a') as file:
    file.write(f"adamw_testloss: {adamw_test_loss:.4f}\n")
    file.write(f"adamw_TrainingLoss: {' '.join(map(str, adamw_train_loss.numpy()))}\n")
    file.write(f"adamw_ValidationLoss: {' '.join(map(str, adamw_val_loss.numpy()))}\n")


# compile model with adamw with decay
print("Training with AdamW with Decay")
pr_adamwd = PoissonRegression()
pr_adamwd.compile(AdamwDecay(learning_rate=DS1_LR_ADAMWD))

# train model
adamwd_train_loss, adamwd_val_loss = pr_adamwd.fit(X_train, y_train, X_val, y_val, DS1_EPOCHS)

#evaluate model
adamwd_test_loss = pr_adamwd.evaluate(X_test, y_test)

# save results
with open(DS1_RESULT_PATH, 'a') as file:
    file.write(f"adamwd_testloss: {adamwd_test_loss:.4f}\n")
    file.write(f"adamwd_TrainingLoss: {' '.join(map(str, adamwd_train_loss.numpy()))}\n")
    file.write(f"adamwd_ValidationLoss: {' '.join(map(str, adamwd_val_loss.numpy()))}\n")


# load dataset 2
X, y = Dataset().load_dataset(DS2_PATH)

# define split percentage
TRAIN_PERCENTAGE = 0.6
TEST_PERCENTAGE = 0.2

# split dataset
X_train, y_train = X[ : int(len(X) * TRAIN_PERCENTAGE)], y[ : int(len(y) * TRAIN_PERCENTAGE)]
X_val, y_val = X[int(len(X) * TRAIN_PERCENTAGE) : int(len(X) * (1 - TEST_PERCENTAGE))], y[int(len(X) * TRAIN_PERCENTAGE) : int(len(y) * (1 - TEST_PERCENTAGE))]
X_test, y_test = X[int(len(X) * (1 - TEST_PERCENTAGE)) : ], y[int(len(y) * (1 - TEST_PERCENTAGE)) : ]


# compile model with gradient descent
print("Training with Gradient Descent")
pr_grad = PoissonRegression()
pr_grad.compile(GradientDescent(learning_rate=DS2_LR_GRAD))

# train model
grad_train_loss, grad_val_loss = pr_grad.fit(X_train, y_train, X_val, y_val, DS2_EPOCHS)

#evaluate model
grad_test_loss = pr_grad.evaluate(X_test, y_test)

# save results
with open(DS2_RESULT_PATH, 'w') as file:
    file.write(f"grad_testloss: {grad_test_loss:.4f}\n")
    file.write(f"grad_TrainingLoss: {' '.join(map(str, grad_train_loss.numpy()))}\n")
    file.write(f"grad_ValidationLoss: {' '.join(map(str, grad_val_loss.numpy()))}\n")


# compile model with regularization
print("Training with Regularization")
pr_regu = PoissonRegression()
pr_regu.compile(Regularizer(learning_rate=DS2_LR_REGU))

# train model
regu_train_loss, regu_val_loss = pr_regu.fit(X_train, y_train, X_val, y_val, DS2_EPOCHS)

#evaluate model
regu_test_loss = pr_regu.evaluate(X_test, y_test)

# save results
with open(DS2_RESULT_PATH, 'a') as file:
    file.write(f"regu_testloss: {regu_test_loss:.4f}\n")
    file.write(f"regu_TrainingLoss: {' '.join(map(str, regu_train_loss.numpy()))}\n")
    file.write(f"regu_ValidationLoss: {' '.join(map(str, regu_val_loss.numpy()))}\n")


# compile model with adam
print("Training with Adam")
pr_adam = PoissonRegression()
pr_adam.compile(Adam(learning_rate=DS2_LR_ADAM))

# train model
adam_train_loss, adam_val_loss = pr_adam.fit(X_train, y_train, X_val, y_val, DS2_EPOCHS)

#evaluate model
adam_test_loss = pr_adam.evaluate(X_test, y_test)

# save results
with open(DS2_RESULT_PATH, 'a') as file:
    file.write(f"adam_testloss: {adam_test_loss:.4f}\n")
    file.write(f"adam_TrainingLoss: {' '.join(map(str, adam_train_loss.numpy()))}\n")
    file.write(f"adam_ValidationLoss: {' '.join(map(str, adam_val_loss.numpy()))}\n")


# compile model with adamw
print("Training with AdamW")
pr_adamw = PoissonRegression()
pr_adamw.compile(Adamw(learning_rate=DS2_LR_ADAMW))

# train model
adamw_train_loss, adamw_val_loss = pr_adamw.fit(X_train, y_train, X_val, y_val, DS2_EPOCHS)

#evaluate model
adamw_test_loss = pr_adamw.evaluate(X_test, y_test)

# save results
with open(DS2_RESULT_PATH, 'a') as file:
    file.write(f"adamw_testloss: {adamw_test_loss:.4f}\n")
    file.write(f"adamw_TrainingLoss: {' '.join(map(str, adamw_train_loss.numpy()))}\n")
    file.write(f"adamw_ValidationLoss: {' '.join(map(str, adamw_val_loss.numpy()))}\n")


# compile model with adamw with decay
print("Training with AdamW with Decay")
pr_adamwd = PoissonRegression()
pr_adamwd.compile(AdamwDecay(learning_rate=DS2_LR_ADAMWD))

# train model
adamwd_train_loss, adamwd_val_loss = pr_adamwd.fit(X_train, y_train, X_val, y_val, DS2_EPOCHS)

#evaluate model
adamwd_test_loss = pr_adamwd.evaluate(X_test, y_test)

# save results
with open(DS2_RESULT_PATH, 'a') as file:
    file.write(f"adamwd_testloss: {adamwd_test_loss:.4f}\n")
    file.write(f"adamwd_TrainingLoss: {' '.join(map(str, adamwd_train_loss.numpy()))}\n")
    file.write(f"adamwd_ValidationLoss: {' '.join(map(str, adamwd_val_loss.numpy()))}\n")