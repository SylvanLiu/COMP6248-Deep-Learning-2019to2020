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



""" transform = transforms.Compose([
    transforms.Resize((240, 800)),
    transforms.ToTensor()  # convert to tensor
])

train_dataset = ImageFolder(path, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# generate the first batch
(batch_images, batch_labels) = train_loader.__iter__().__next__()


# plot 4 images
plt.subplot(221).set_title(train_dataset.classes[batch_labels[0]])
plt.imshow(batch_images[0].permute(1, 2, 0), aspect='equal')
plt.subplot(222).set_title(train_dataset.classes[batch_labels[1]])
plt.imshow(batch_images[1].permute(1, 2, 0), aspect='equal')
plt.subplot(223).set_title(train_dataset.classes[batch_labels[2]])
plt.imshow(batch_images[2].permute(1, 2, 0), aspect='equal')
plt.subplot(224).set_title(train_dataset.classes[batch_labels[3]])
plt.imshow(batch_images[3].permute(1, 2, 0), aspect='equal')

# show the plot
plt.show() """

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


# Model Definition


class BetterCNN(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels_in, 30, (5, 5), padding=0)
        self.conv2 = nn.Conv2d(30, 15, (3, 3), padding=0)
        self.fc1 = nn.Linear(1725, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

model = torch.load(path+'lab_6_boat_cnn_100epochs.pytorchweights')
# model = BetterCNN(3, len(train_dataset.classes))

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function,
              metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader,
                      test_generator=test_loader)
# history = trial.run(epochs=100)
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)
# torch.save(model, path+'lab_6_boat_cnn_100epochs.pytorchweights')
""" loss_list = []
for item in history:
    loss_list.append(item['loss'])
fig1 = plt.figure('1')
ax1 = fig1.gca()
ax1.plot(loss_list)
plt.show()
 """


predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()
true_classes = list(x for (_,x) in test_dataset.samples)

from sklearn import metrics
print(metrics.classification_report(true_classes, predicted_classes, target_names=train_dataset.classes))
