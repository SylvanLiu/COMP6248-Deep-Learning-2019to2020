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

pi = torch.acos(torch.zeros(1)).item() * 2
coe = 1.0

class Rastrigin(nn.Module):
    def __init__(self):
        super().__init__()
        self.X = nn.Parameter(torch.tensor([5.]))
        self.Y = nn.Parameter(torch.tensor([5.]))

    def forward(self):
        return (self.X**2 - coe * torch.cos(2 * pi * self.X)) + (self.Y**2 - coe * torch.cos(2 * pi * self.Y)) + 2*coe


def get_model(optimiser, momentum, lr):
    if momentum!=0:
        model = Rastrigin()
        return model, optimiser(model.parameters(), lr=lr, momentum=momentum)
    else:
        model = Rastrigin()
        return model, optimiser(model.parameters(), lr=lr)


def search(optimiser, momentum=0, epochs=100, lr=0.01):
    model, opt = get_model(optimiser, momentum, lr)
    path = []
    z_list = []
    for _ in range(epochs):
        Z = model()
        Z.backward()
        opt.step()
        opt.zero_grad()
        z_list.append(float(Z))
        path.append([float(model.X), float(model.Y), float(Z)])
    print(z_list[-1])
    print(path[-1])
    return z_list, path


def plot_3d_line(ax,path,color):
    for i in range(len(path)-1):
        ax.plot([path[i][0], path[i+1][0]], [path[i][1],path[i+1][1]],[path[i][2],path[i+1][2]],color = color,  linewidth=3)





X_bg = np.linspace(-5.12, 5.12, 100)
Y_bg = np.linspace(-5.12, 5.12, 100)
X_bg, Y_bg = np.meshgrid(X_bg, Y_bg)

Z_bg = (X_bg**2 - coe * np.cos(2 * float(pi) * X_bg)) + \
    (Y_bg**2 - coe * np.cos(2 * float(pi) * Y_bg)) + 2*coe

fig1 = plt.figure('3d')
ax1 = fig1.gca(projection='3d')
ax1.plot_surface(X_bg, Y_bg, Z_bg, cmap=plt.cm.viridis)

z_list_1, path_1 = search(optimiser=optim.SGD)
z_list_2, path_2 = search(optimiser=optim.SGD, momentum=0.9)
z_list_3, path_3 = search(optimiser=optim.Adagrad)
z_list_4, path_4 = search(optimiser=optim.Adam)
plot_3d_line(ax1,path_1,'red')
plot_3d_line(ax1,path_2,'blue')
plot_3d_line(ax1,path_3,'black')
plot_3d_line(ax1,path_4,'green')

# ax.w_xaxis.set_pane_color ((1, 1, 1, 1))
# ax.w_yaxis.set_pane_color ((1, 1, 1, 1))
# ax.w_zaxis.set_pane_color ((1, 1, 1, 1))
ax1.set_facecolor('white')
ax1.grid(True)

fig2 = plt.figure('2d')
ax2 = fig2.gca()

ax2.plot(z_list_1,color = 'red')
ax2.plot(z_list_2,color = 'blue')
ax2.plot(z_list_3,color = 'black')
ax2.plot(z_list_4,color = 'green')

plt.show()
