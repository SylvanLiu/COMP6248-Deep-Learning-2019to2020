from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch import nn
from IPython.core.debugger import set_trace
import math
import torch
import numpy as np
import gzip
import pickle
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid),
     _) = pickle.load(f, encoding="latin-1")


# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)
# pyplot.show()


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


train_dl, valid_dl = get_data(train_ds, valid_ds, bs=5000)


class MLP(nn.Module):
    def __init__(self, hln_num_list):
        super().__init__()
        self.hln_num_list = hln_num_list
        self.relu = nn.ReLU()
        if len(hln_num_list) != 0:
            self.input_layer = nn.Linear(784, hln_num_list[0])
            self.hidden_layer = self._make_hiden_layer(hln_num_list)
            self.output_layer = nn.Linear(hln_num_list[-1], 10)
        else:
            self.input_layer = nn.Linear(784, 784)
            self.output_layer = nn.Linear(784, 10)

    def forward(self, x):
        x = self.input_layer(x)
        if len(self.hln_num_list) != 0:
            x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

    def _make_hiden_layer(self, hln_num_list):
        layers = []
        l = len(hln_num_list)
        for i in range(l):
            if i+1 == l:
                layers.append(nn.Linear(hln_num_list[i], hln_num_list[i]))
            else:
                layers.append(nn.Linear(hln_num_list[i], hln_num_list[i+1]))
        return nn.Sequential(*layers)


def get_model(hln_num_list, optimiser, momentum, lr, weight_decay):
    if momentum != 0:
        model = MLP(hln_num_list=hln_num_list)
        return model, optimiser(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        model = MLP(hln_num_list=hln_num_list)
        return model, optimiser(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(training_data, hln_num_list, optimiser, loss_func, epochs, lr, momentum=0, weight_decay=0.0001):
    model, opt = get_model(hln_num_list, optimiser, momentum, lr, weight_decay)
    loss_list = []
    for epoch in range(epochs):
        for xb, yb in training_data:
            model.train()
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_list.append(loss)
        print(str(epoch+1) + '/' + str(epochs))
    return loss_list, model


def valid(validation_data, model):
    model.eval()
    for xb, yb in validation_data:
        pred = model(xb)
        _labels = []
        for result in pred.tolist():
            label = result.index(max(result))
            _labels.append(label)
        _labels = np.asarray(_labels)
        yb = yb.numpy()
        acc = sum(_labels == yb)/yb.size
        return acc


x = [10, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 350, 400,
     500, 600, 700, 800, 1000, 1200, 1500, 3000, 5000, 10000]
y = []
i = 0
len_x = len(x)
fig2 = plt.figure('2')
ax2 = fig2.gca()
for a in x:
    loss_list, model = train(train_dl, hln_num_list=[
        448], optimiser=optim.SGD, loss_func=F.cross_entropy, epochs=50, lr=0.05, momentum=0.9)
    y.append(valid(valid_dl, model))
    ax2.plot(loss_list, label=str(a))
    ax2.legend()
    i += 1
    print('------------ '+str(i)+'/' + str(len_x) + ' ------------')

print(y)
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.grid(True)
ax1.plot(x, y)
plt.xticks(x)
plt.show()

''' y = []
a = 0
while a < 10:
    hln_num_list = []
    for _ in range(a):
        hln_num_list.append(10)
    print(hln_num_list)
    loss_list, model = train(train_dl, hln_num_list=hln_num_list, optimiser=optim.SGD,
                             loss_func=F.cross_entropy, epochs=50, lr=0.05, momentum=0.9)
    y.append(valid(valid_dl, model))
    a += 1
    print('------------ '+str(a)+'/10 ------------')

print(y)
aa = 0
x= []
for i in y:
    x.append(aa)
    aa += 1
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.grid(True)
ax1.plot(x, y)
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.show() '''
