import math
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' +
                 '/iris/iris.data', header=None)
df = df.sample(frac=1)  # shuffle

# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)

# normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

# create datasets
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]


def train(training_data, training_labels, epochs=100, lr=0.01, batch_size=16):
    loss_func = F.cross_entropy
    W1 = torch.randn(4, 12) / math.sqrt(4)
    W2 = torch.randn(12, 3) / math.sqrt(12)
    W1.requires_grad_()
    W2.requires_grad_()
    b1 = torch.zeros(12, requires_grad=True)
    b2 = torch.zeros(3, requires_grad=True)
    n = (training_data.shape)[0]
    loss_list = []
    for _ in range(epochs):
        for i in range((n - 1) // batch_size + 1):
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = training_data[start_i:end_i]
            yb = training_labels[start_i:end_i]
            pred = torch.relu(xb@W1 + b1)@W2 + b2
            loss = loss_func(pred, yb)
            loss.backward()
            with torch.no_grad():
                W1 -= W1.grad * lr
                W2 -= W2.grad * lr
                b1 -= b1.grad * lr
                b2 -= b2.grad * lr
                W1.grad.zero_()
                W2.grad.zero_()
                b1.grad.zero_()
                b2.grad.zero_()
        loss_list.append(loss_func(pred, yb))
    plt.plot(loss_list)
    return W1, W2, b1, b2

def validate(model, validation_data, validation_labels):
    W1 = model[0]
    W2 = model[1]
    b1 = model[2]
    b2 = model[3]
    pred_raw = torch.relu(data_va @ W1 + b1) @ W2 +b2
    pred_labels = list(map(lambda x:np.argmax(x), pred_raw.data.numpy()))
    acc_valid = sum(pred_labels == validation_labels.numpy())/len(validation_labels)
    print(acc_valid)
    
validate(train(data_tr, targets_tr),data_va,targets_va)
plt.show()