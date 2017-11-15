# From http://pyro.ai/examples/bayesian_regression.html
#
#

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam

N = 100     # size of data
p = 1       # num of features
LR = 0.01
steps = 500


def build_linear_dataset(N, noise_std=0.1):  # fixed observation noise of 0.1
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X, y = X.reshape((N, 1)), y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)


regression_model = RegressionModel(p)  # Network model - linear regression
loss_fn = nn.MSELoss(size_average=False)  # Mean squared error loss function
optim = Adam(regression_model.parameters(), LR)  # Adam optimizer


def main():
    data = build_linear_dataset(N, p)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    for j in range(num_iterations):
        # run model forward
        y_pred = regression_model(x_data)
        loss = loss_fn(y_pred, y_data)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (j + 1) % 50 == 0:
            print("[Iteration {:04}] Loss: {:.4}".format((j + 1), loss.data[0]))
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print("{}: {:.3}".format(name, param.data.numpy()))

if __name__ == '__main__':
    main()
