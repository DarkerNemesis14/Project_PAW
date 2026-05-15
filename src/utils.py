import torch

class GradientDescent:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate

    def update(self, w: torch.Tensor, gradient: torch.Tensor):
        # update parameters
        w -= self.learning_rate * gradient
        
        return w


class Regularizer:
    def __init__(self, learning_rate=1e-2, decay=0.01):
        self.learning_rate = learning_rate
        self.decay = decay

    def update(self, w: torch.Tensor, gradient: torch.Tensor):
        # update parameters
        w -= self.learning_rate * (gradient + self.decay * w)
        
        return w


class Adam:
    def __init__(self, learning_rate=1e-2, beta1=0.9, beta2=0.999):
        # initiate variables
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 1

    def update(self, w: torch.Tensor, gradient: torch.Tensor):
        # calculate moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / 1 - self.beta1**self.t
        v_hat = self.v / 1 - self.beta2**self.t
        
        # update parameters
        w -= self.learning_rate * (m_hat / (torch.sqrt(v_hat) + 1e-5))

        # update time
        self.t += 1

        return w


class Adamw:
    def __init__(self, learning_rate=1e-2, beta1=0.9, beta2=0.999, decay = 0.01):
        # initiate variables
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.m = 0
        self.v = 0
        self.t = 1

    def update(self, w: torch.Tensor, gradient: torch.Tensor):
        # calculate moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / 1 - self.beta1**self.t
        v_hat = self.v / 1 - self.beta2**self.t

        # update parameters
        w -= self.learning_rate * (m_hat / (torch.sqrt(v_hat) + 1e-5) + self.decay * w)
        
        # update time
        self.t += 1

        return w
    

class AdamwDecay:
    def __init__(self, learning_rate=1e-2, beta1=0.9, beta2=0.999):
        # initiate variables
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 1

    def update(self, w: torch.Tensor, gradient: torch.Tensor):
        # calculate moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / 1 - self.beta1**self.t
        v_hat = self.v / 1 - self.beta2**self.t

        # calculate decay
        gt_hat = (torch.abs(gradient) - gradient.mean()) / (gradient.std() + 1e-5)
        decay = 2 * torch.sigmoid(4*gt_hat)

        # update parameters
        w -= self.learning_rate * (m_hat / (torch.sqrt(v_hat) + 1e-5) + decay * w)

        # update time
        self.t += 1

        return w
