import torch
import pandas as pd

class Dataset:
    def __init__(self):
        # set seed
        torch.manual_seed(42)

    def generate_dataset(self, n_features: int, n_samples: int, path: str):
        # generate features
        X = torch.rand(n_samples, n_features) * 2 - 1

        # generate parameters
        w = torch.rand(n_features) * 2 - 1

        # calculate mu
        mu = torch.exp(X @ w)

        # generate y
        y = torch.poisson(mu)

        # create dataframe
        columns = [f'x{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X.numpy(), columns= columns)
        df['y'] = y.numpy().astype(int)

        # generate csv file
        df.to_csv(path, index=False)


    def load_dataset(self, path: str):
        # load csv
        df = pd.read_csv(path)

        # extract values
        X = df.drop(columns="y").values
        y = df["y"].values

        # convert to tensor
        X = torch.tensor(X, dtype= torch.float32)
        y = torch.tensor(y, dtype= torch.float32)

        return X, y
