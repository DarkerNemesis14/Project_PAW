import torch
from tqdm import tqdm

class PoissonRegression:
    def __init__(self):
        # set seed
        torch.manual_seed(42)

    def compile(self, optimizer: object):
        # set optimizer
        self.optimizer = optimizer

    def evaluate(self, X: torch.Tensor, y: torch.Tensor):
        # calculate eta
        eta = X @ self.w

        # calculate loss
        total_loss = torch.sum(torch.exp(eta) - y * eta)
        loss = total_loss / X.shape[0]

        return loss

    def predict(self, X: torch.Tensor):
        # calculate y
        y = torch.exp(X @ self.w)
        
        return y

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int):
        # intiate weights
        self.w = torch.rand(X_train.shape[1]) * 2 - 1

        # initiate other variables
        train_loss = torch.zeros(epochs+1)
        train_loss[0] = self.evaluate(X_train, y_train)

        with tqdm(total=epochs, desc="Training Model:") as pbar:
            # training loop
            for epoch in range(epochs):
                # update parameters
                gradient = X_train.T @ (self.predict(X_train) - y_train)
                self.w = self.optimizer.update(self.w, gradient)

                # calculate loss
                train_loss[epoch+1] = self.evaluate(X_train, y_train)
                
                print(f"Epoch: {epoch+1}; Loss: {train_loss[epoch+1]}")
                # update bar
                pbar.update(1)
        
        return train_loss
