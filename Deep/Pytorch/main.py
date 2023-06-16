import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


# dowload và chia dữ liệu thành tệp train, val,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.modules import transformer

dataset_train_val = torchvision.datasets.FashionMNIST("./", download=True, train = True, transform = torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.FashionMNIST("./", download=True, train = False, transform = torchvision.transforms.ToTensor())
#split train and val
from sklearn.model_selection import train_test_split
dataset_train, dataset_val = train_test_split(dataset_train_val, test_size=0.2, random_state=4)

train_loader = DataLoader(dataset= dataset_train, batch_size = 32)
val_loader = DataLoader(dataset= dataset_val, batch_size = 32)
test_loader = DataLoader(dataset = dataset_test, batch_size = 32)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

#tạo model
class Model(nn.Module):
    def __init__(self, nc):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, nc)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model = Model(10)

#chọn hàm mất mát và hàm tối ưu hóa có sẵn
import torch.optim as opt

optimizer = opt.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

#hàm train
from tqdm import tqdm

def train(train_loader, val_loader):
    LOSSES_train = []
    ACCS = []
    LOSSES_val = []
    for epoch in range(10):
        LOSS_train = []
        acc = 0
        for x, y in tqdm(train_loader):
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS_train.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            acc += (torch.argmax(torch.softmax(yhat, dim=1), dim = 1) == y).float().mean()

        LOSSES_train.append(sum(LOSS_train)/len(LOSS_train))
        ACCS.append(acc)

        with torch.no_grad():
            acc = 0
            LOSS_val = []
            for x, y in tqdm(val_loader):
                yhat = model(x)
                loss = criterion(yhat, y)
                LOSS_val.append(loss.item())

                acc += (torch.argmax(torch.softmax(yhat, dim = 1), dim = 1) == y).float().mean()

            LOSSES_val.append(sum(LOSS_val))
            print(acc/len(val_loader))

    return (LOSSES_train , ACCS, LOSSES_val)


#hàm để test lại model
def test(test_loader):
    LOSSES_test = []
    ACCS = []
    for epoch in range(10):
        print(epoch)
        LOSS_train = []
        acc = 0
        for x, y in tqdm(test_loader):
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS_train.append(loss.data)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            acc += (torch.argmax(torch.softmax(yhat, dim=1), dim = 1) == y).float().mean()

        LOSSES_test.append(sum(LOSS_train)/len(LOSS_train))
        ACCS.append(acc)
        print(acc/len(test_loader))
    return (LOSSES_test , ACCS)

#trainning
losses_train , accs_train, losses_val = train(train_loader, val_loader)


#trực quan hóa loss train, loss val, accuracy
import matplotlib.pyplot as plt

fg, ax = plt.subplots(1, 3, figsize=(14, 10))

ax[0].set_title('LOSS TRAIN')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].plot(losses_train, 'ro-')

ax[1].set_title('LOSS VAL')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].plot(losses_val, 'bo-')

ax[2].set_title('Accuracy')
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('ACC_mean')
ax[2].plot(accs_train, 'yo-')

plt.show()

#testing
losses_test , accs_test = test(test_loader)

# trực quan hóa loss test, acc test
fg, ax1 = plt.subplots(1, 2)

ax1[0].set_title('LOSS TRAIN')
ax1[0].set_xlabel('Epoch')
ax1[0].set_ylabel('Loss')
ax1[0].plot(losses_test, 'ro-')

ax1[1].set_title('Accuracy')
ax1[1].set_xlabel('Epoch')
ax1[1].set_ylabel('ACC_mean')
ax1[1].plot(accs_test, 'bo-')

plt.show()

# trực quan hóa 10 dữ liệu test đầu tiên
import numpy as np

dataiter = iter(test_loader)
images, labels = next(dataiter)
desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
for i in range(10):
  plt.subplot(2, 5, i+1)
  print(desc[labels[i]])
  plt.imshow(np.squeeze(images[i]), cmap="gray")
plt.show()


#trực quan hóa predict
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
fg, ax = plt.subplots(10, 2, figsize=(13, 60))
for i in range(10):
  img, label = images[i], labels[i]
  preds = model(images)
  desc = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
  ax[i, 0].axis('off')
  ax[i, 0].imshow(images[i].detach().numpy().squeeze())
  ax[i, 0].set_title(desc[label.item()])
  preds = preds.detach().numpy()[i]
  preds = np.exp(preds)
  ax[i, 1].bar(range(10), preds)
  ax[i, 1].set_xticks(range(10))
  ax[i, 1].set_xticklabels(desc)
  ax[i, 1].set_title('Predicted Probabilities')
plt.tight_layout()