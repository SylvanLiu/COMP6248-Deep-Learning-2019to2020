from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
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
import torchbearer as tb
from torchbearer import Trial
import seaborn as sns
sns.set()


class MyDataset(Dataset):
    def __init__(self, size=5000, dim=40, random_offset=0):
        super(MyDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.random_offset = random_offset

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("{} index out of range".format(
                self.__class__.__name__))

        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)

        while True:
            img = torch.zeros(self.dim, self.dim)
            dx = torch.randint(-10, 10, (1,), dtype=torch.float)
            dy = torch.randint(-10, 10, (1,), dtype=torch.float)
            c = torch.randint(-20, 20, (1,), dtype=torch.float)

            params = torch.cat((dy/dx, c))
            xy = torch.randint(0, img.shape[1], (20, 2), dtype=torch.float)
            xy[:, 1] = xy[:, 0] * params[0] + params[1]

            xy.round_()
            xy = xy[xy[:, 1] > 0]
            xy = xy[xy[:, 1] < self.dim]
            xy = xy[xy[:, 0] < self.dim]

            for i in range(xy.shape[0]):
                x, y = xy[i][0], self.dim - xy[i][1]
                img[int(y), int(x)] = 1
            if img.sum() > 2:
                break

        torch.set_rng_state(rng_state)
        return img.unsqueeze(0), params

    def __len__(self):
        return self.size


train_data = MyDataset()
val_data = MyDataset(size=500, random_offset=33333)
test_data = MyDataset(size=500, random_offset=99999)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
valid_dl = DataLoader(val_data, batch_size=32, shuffle=True)
test_dl = DataLoader(test_data, batch_size=32, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, (3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(48 * 40**2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class MidiumCNN(nn.Module):
    def __init__(self):
        super(MidiumCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, (3, 3), stride=1, padding=1)
        self.gmp1 = nn.AdaptiveMaxPool2d((2, 2))
        self.fc1 = nn.Linear(48 * 2**2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.shape)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        print(out.shape)
        out = self.gmp1(out)
        print(out.shape)
        out = out.view(out.shape[0], -1)
        print(out.shape)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

# Model Definition


class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, (3, 3), stride=1, padding=1)
        self.gmp1 = nn.AdaptiveMaxPool2d((2, 2))
        self.fc1 = nn.Linear(48 * 2**2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        plt.imshow(x.numpy()[0, 0, :, :])
        plt.show()
        idxx = torch.repeat_interleave(
            torch.arange(-20, 20, dtype=torch.float).unsqueeze(0) / 40.0, repeats=40, dim=0).to(x.device)
        plt.imshow(idxx.numpy())
        plt.show()
        idxy = idxx.clone().T
        plt.imshow(idxy.numpy())
        plt.show()
        idx = torch.stack([idxx, idxy]).unsqueeze(0)
        idx = torch.repeat_interleave(idx, repeats=x.shape[0], dim=0)
        x = torch.cat([x, idx], dim=1)
        out = self.conv1(x)
        plt.imshow(out.detach().numpy()[0, 0, :, :])
        plt.show()
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.gmp1(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


model = BetterCNN()
loss_function = nn.L1Loss()
optimiser = optim.Adam(model.parameters())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function,
              metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_dl, valid_dl, test_generator=test_dl)
history = trial.run(epochs=1)
results = trial.evaluate(data_key=tb.TEST_DATA)
print(results)
loss_list = []
for item in history:
    loss_list.append(item['loss'])
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.plot(loss_list)
plt.show()
