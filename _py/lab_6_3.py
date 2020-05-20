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

from torchvision.models import resnet50
from torchvision.models import resnet152
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




training_features = np.load('Resnet50Features/training_features.npy')
training_labels = np.load('Resnet50Features/training_labels.npy')

valid_features = np.load('Resnet50Features/valid_features.npy')
valid_labels = np.load('Resnet50Features/valid_labels.npy')

testing_features = np.load('Resnet50Features/testing_features.npy')
testing_labels = np.load('Resnet50Features/testing_labels.npy')
print(training_labels)

clf = svm.SVC()
clf.fit(training_features, training_labels)
prelabels = clf.predict(testing_features)

count = 0
for i in range(testing_labels.shape[0]):
  if prelabels[i] ==testing_labels[i]:
    count +=1
acc_svm = count/testing_labels.shape[0]
print('The acc using SVM classifier is: ', acc_svm)

from sklearn import metrics
print(metrics.classification_report(testing_labels, prelabels, target_names=train_dataset.classes))