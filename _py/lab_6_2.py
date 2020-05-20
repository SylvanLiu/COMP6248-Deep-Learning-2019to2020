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

imagenet_labels = urlopen("https://raw.githubusercontent.com/kundan2510/resnet152-pre-trained-imagenet/master/imagenet_classes.txt").read().decode('utf-8').split("\n")

""" model = resnet152(pretrained=True)
model.eval()
print(model)
preprocess_input = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
 """
""" from PIL import Image as PImage
img_path = 'data/boat/IMG_1494.jpeg'
img = PImage.open(img_path)
# plt.imshow(preprocess_input(img).permute(1, 2, 0))
# plt.show()
preds = model(preprocess_input(img).unsqueeze(0))

_, indexes = preds.topk(5)
for i in indexes[0]:
    print('Predicted:', imagenet_labels[i])
 """



# model = resnet50(pretrained=True)
# model.avgpool = nn.AdaptiveAvgPool2d((1,1))
# model.fc = nn.Linear(2048, len(train_dataset.classes))
# model.train()
# print(model)
 # Freeze layers by not tracking gradients
# for param in model.parameters():
#     param.requires_grad = False
# model.fc.weight.requires_grad = True #unfreeze last layer weights
# model.fc.bias.requires_grad = True #unfreeze last layer biases
# define the loss function and the optimiser
""" loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) #only optimse non-frozen layers
device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
history = trial.run(epochs=90)
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
torch.save(model, path+'lab_6_boat_resnet50_100epochs.pytorchweights')
loss_list = []
for item in history:
    loss_list.append(item['loss'])
# fig1 = plt.figure('1')
# ax1 = fig1.gca()
# ax1.plot(loss_list)
# plt.show()
print(loss_list)
print(results) """

model = torch.load('data/boat/trainlab_6_boat_resnet50_100epochs.pytorchweights')
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
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
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