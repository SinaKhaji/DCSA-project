import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(in_features=64*7*7, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def download_dataset(data_dir="../data"):
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

def get_dataloaders(batch_size=128, data_dir="../data"):
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_set = torchvision.datasets.MNIST(data_dir, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)
    print("Dataset loaded.")
    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()                            # In DDP the gradients from different processes gets aggregated

        optimizer.step()

        batch_size = len(images)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)         # predicted class = the class with the highest score
        good_answers = (predictions == labels)      # True where the prediction is correct
        correct += good_answers.sum().item()

        count += batch_size

    average_loss = total_loss / count
    accuracy = correct / count
    return average_loss, accuracy

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            batch_size = len(images)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            good_answers = (predictions == labels)
            correct += good_answers.sum().item()

            count += batch_size

    average_loss = total_loss / count
    accuracy = correct / count
    return average_loss, accuracy

def save(directory, filename, report):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{filename}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {path}")
