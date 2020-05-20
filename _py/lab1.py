from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'+'/iris/iris.data', header=None)
data = torch.tensor(df.iloc[:, [0,1,2,3]].values)
data = data - data.mean(dim=0)

A_original = torch.tensor([[0.3374, 0.6005, 0.1735], [3.3359, 0.0492, 1.8374], [
    2.9407, 0.5301, 2.2620]], requires_grad=False)


def gd_factorise_ad(A: torch.Tensor, rank: int, epochs=65536, lr=0.0001) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_list = []
    U_estimate = torch.tensor(torch.rand(
        [*A.size()][0], rank), requires_grad=True)
    V_estimate = torch.tensor(torch.rand(
        [*A.size()][1], rank), requires_grad=True)
    for _ in range(epochs):
        loss = torch.norm(A - U_estimate@(V_estimate.T))
        loss.backward()
        loss_list.append(float(loss))
        with torch.no_grad():
            U_estimate -= lr*U_estimate.grad
            V_estimate -= lr*V_estimate.grad
            V_estimate.grad.zero_()
            U_estimate.grad.zero_()
    # print(U_estimate@(V_estimate.T))
    print(torch.norm(A - U_estimate@(V_estimate.T)))
    plt.plot(loss_list)
    return U_estimate, V_estimate

U_estimate, V_estimate = gd_factorise_ad(data, 2)

U_svd, Sigma_svd, VT_svd = randomized_svd(
    np.asarray(data), n_components=2, transpose=True)

# print(U_svd@np.diag(Sigma_svd)@VT_svd)
print(torch.norm(data - torch.tensor(U_svd@np.diag(Sigma_svd)@VT_svd)))

plt.show()

# U_estimate, V_estimate = gd_factorise_ad(A_original, 2)

# U_svd, Sigma_svd, VT_svd = randomized_svd(
#     np.asarray(A_original), n_components=2, transpose=True)


# print(U_svd@np.diag(Sigma_svd)@VT_svd)
# print(torch.norm(A_original - torch.tensor(U_svd@np.diag(Sigma_svd)@VT_svd)))


''' def sgd_factorise_sub(A: torch.Tensor, rank: int, epoch=1024, lr=0.005) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    e_list = []
    U_estimate = torch.rand(m, rank)
    V_estimate = torch.rand(n, rank)
    for _ in range(epoch):
        for r in range(m):
            for c in range(n):
                e = A[r, c] - U_estimate[r] @ (V_estimate[c].T)
                U_estimate[r] = U_estimate[r] + lr*e*V_estimate[c]
                V_estimate[c] = V_estimate[c] + lr*e*U_estimate[r]
        e_list.append(float(torch.norm(A - torch.tensor(U_estimate@(V_estimate.T)))))
    print(U_estimate@(V_estimate.T))
    print(float(torch.norm(A - torch.tensor(U_estimate@(V_estimate.T)))))
    plt.plot(e_list)
    return U_estimate, V_estimate


U_estimate_sub, V_estimate_sub = sgd_factorise_sub(A_original, 2)

plt.show() '''


''' def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank: int, epochs=1024, lr=0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    e_list = []
    m, n = A.shape
    U_estimate = torch.tensor(torch.rand(m, rank), requires_grad=True)
    V_estimate = torch.tensor(torch.rand(n, rank), requires_grad=True)
    for _ in range(epochs):
        for r in range(m):
            for c in range(n):
                if M[r, c] == 1:
                    e = A[r, c] - U_estimate[r] @ (V_estimate[c].T)
                    U_estimate[r] = U_estimate[r] + lr*e*V_estimate[c]
                    V_estimate[c] = V_estimate[c] + lr*e*U_estimate[r]
        e_list.append(
            float(torch.norm(A_original - torch.tensor(U_estimate @ (V_estimate.T)))))
    plt.plot(e_list)
    print(U_estimate @ (V_estimate.T))
    print(torch.norm(A_original - torch.tensor(U_estimate @ (V_estimate.T))))
    return U_estimate, V_estimate


A_masked = torch.Tensor([[0.3374, 0.6005, 0.1735],
                         [0, 0.0492, 1.8374],
                         [2.9407, 0, 2.2620]])
M = torch.Tensor([[1, 1, 1],
                  [0, 1, 1],
                  [1, 0, 1]])


U_estimate_masked, V_estimate_masked = sgd_factorise_masked(A_masked, M, 2)
plt.show()
 '''