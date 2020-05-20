import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import FashionMNIST
import torchbearer
import torchbearer.callbacks as callbacks
from torchbearer import Trial, state_key
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
MU = state_key('mu')
LOGVAR = state_key('logvar')


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),   # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_size)
        self.logvar = nn.Linear(64 * 7 * 7, latent_size)

        self.upsample = nn.Linear(latent_size, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2)   # B, 1, 28, 28
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, state):
        image = x
        x = self.encoder(x).relu().view(x.size(0), -1)

        mu = self.mu(x)
        logvar = self.logvar(x)
        sample = self.reparameterize(mu, logvar)

        result = self.decoder(self.upsample(sample).relu().view(-1, 64, 7, 7))

        if state is not None:
            state[torchbearer.Y_TRUE] = image
            state[MU] = mu
            state[LOGVAR] = logvar

        return result


transform = transforms.Compose([transforms.ToTensor()])  # No augmentation
trainset = FashionMNIST(root='../data', train=True,
                        transform=transform, download=True)
testset = FashionMNIST(root='../data', train=False,
                       transform=transform, download=True)
traingen = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)
testgen = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=8)


def beta_kl(mu_key, logvar_key, beta=5):
    @callbacks.add_to_loss
    def callback(state):
        mu = state[mu_key]
        logvar = state[logvar_key]
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    return callback


def plot_progress(key=torchbearer.Y_PRED, num_images=128, nrow=16):
    @callbacks.on_step_validation
    @callbacks.once_per_epoch
    def callback(state):
        images = state[key]
        image = make_grid(images[:num_images],
                          nrow=nrow, normalize=True)[0, :, :]
        plt.imshow(image.detach().cpu().numpy(), cmap="gray")
        plt.show()
    return callback


model = torch.load('fashionMNIST_VAE_20epochs.pytorchweights')


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(sum([np.prod(p.size()) for p in model_parameters]))

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), metrics=['acc', 'loss'], callbacks=[
    beta_kl(MU, LOGVAR),
    callbacks.ConsolePrinter(),
    plot_progress()
], verbose=1).with_generators(train_generator=traingen, test_generator=testgen)

history = trial.run(10)
trial.evaluate(verbose=0, data_key=torchbearer.TEST_DATA)

torch.save(model, 'fashionMNIST_VAE_30epochs.pytorchweights')

# torch.save(model, 'fashionMNIST_VAE_10epochs.pytorchweights')
print(history)
train_loss_list = []
val_loss_list = []
for item in history:
    train_loss_list.append(item['running_loss'])
    val_loss_list.append(item['test_loss'])
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.plot(train_loss_list, color='blue', alpha=0.7)
ax1.plot(val_loss_list, color='red', alpha=0.7)
plt.legend()
plt.show()
