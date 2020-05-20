import numpy as np
from sklearn import svm

import torchbearer
import torch
from torch import optim
from torchbearer import Trial
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from torchvision.models import resnet50
from urllib.request import urlopen

path = "data/boat/train"

# the number of images that will be processed in a single step
batch_size = 128
# the size of the images that we'll learn on - we'll shrink them from the original size for speed
image_size = (30, 100)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # convert to tensor
])

train_dataset = ImageFolder(path, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageFolder(path, transform)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = ImageFolder(path, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet50(pretrained=True)

preprocess_input = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function,
              metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader,
                      test_generator=test_loader)
history = trial.run(epochs=1)
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
print(results)
torch.save(model, 'third_approach_resnet50_100epochs.pytorchweights')
train_loss_list = []
val_loss_list = []
for item in history:
    train_loss_list.append(item['loss'])
    val_loss_list.append(item['val_loss'])
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.plot(train_loss_list,color = 'blue',alpha = 0.7)
ax1.plot(val_loss_list,color = 'red',alpha = 0.7)
plt.legend()
plt.show()