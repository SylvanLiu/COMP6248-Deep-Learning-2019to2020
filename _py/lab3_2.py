from statistics import mean
from torch import optim
from typing import Tuple
import torch.nn.functional as F
from torch import nn
import torch
from sklearn.utils.extmath import randomized_svd
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' +
                 '/iris/iris.data', header=None)

df = df.sample(frac=1, random_state=0)  # shuffle
df = df[df[4].isin(['Iris-virginica', 'Iris-versicolor'])]  # filter
# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = (2*df[4].map(mapping)) - 1  # labels in {−1,1}
# normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)
data_tr = alldata[:75]
data_va = alldata[75:]



class SVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 1) / math.sqrt(4))
        self.b = nn.Parameter(torch.zeros(1, 1))

    def forward(self, xb):
        return xb@self.w + self.b


def simple_loss(pred, yb):
    return torch.abs(yb-pred).mean()


def get_model(optimiser, momentum, lr, weight_decay):
    if momentum != 0:
        model = SVM()
        return model, optimiser(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        model = SVM()
        return model, optimiser(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(training_data, training_labels, optimiser, batch_size=25, momentum=0, epochs=100, lr=0.01, weight_decay=0.0001):
    n = (training_data.shape)[0]
    loss_list = []
    loss_func = simple_loss
    model, opt = get_model(optimiser, momentum, lr, weight_decay)
    for _ in range(epochs):
        for i in range((n - 1) // batch_size + 1):
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = training_data[start_i:end_i]
            yb = training_labels[start_i:end_i]
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss_list.append(loss_func(model(xb), yb))
    return loss_list, model


def validation(validation_data, validation_labels, model):
    w = model.w
    b = model.b
    pred_raw = validation_data@w + b
    pred_labels = list(
        map(lambda x: 1 if x > 0 else -1, pred_raw.data.numpy()))
    acc_valid = sum(pred_labels == validation_labels.numpy()) / \
        len(validation_labels)
    return acc_valid


valid_acc_1 = []
valid_acc_2 = []
a = 0
while a < 1024:
    loss_list_1, model_1 = train(data_tr, targets_tr, optimiser=optim.SGD)
    valid_acc_1.append(float(validation(data_va, targets_va, model_1)))

    loss_list_2, model_2 = train(data_tr, targets_tr, optimiser=optim.Adam)
    valid_acc_2.append(float(validation(data_va, targets_va, model_2)))

    # ax1.plot(loss_list_1, color='red', alpha=0.2)
    # ax1.plot(loss_list_2, color='blue',alpha=0.2)

    a += 1

fig1 = plt.figure('1')
ax1 = fig1.gca()

fig2 = plt.figure('2')
ax2 = fig2.gca()

ax1.hist(valid_acc_1, density=True, bins=32, color='blue',alpha = 0.7)
ax2.hist(valid_acc_2, density=True, bins=32, color='red',alpha = 0.7)

plt.show()
