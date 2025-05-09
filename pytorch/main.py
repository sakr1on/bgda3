import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score

# 1. Определяем класс модели
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# 2. Главная точка входа
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # Настройка параметров
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4

    # Загрузка данных
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    classes = trainset.classes

    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(conv.parameters(), lr=learning_rate)

    print("\nОбучение началось...\n")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = conv(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            if (i + 1) % 500 == 0:
                print(f"[Эпоха {epoch+1}, итерация {i+1}] средняя потеря: {running_loss / 500:.4f}")
                running_loss = 0.0

    print("\nОбучение завершено.")

    # Оценка
    conv.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = conv(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy модели на тестовой выборке: {acc * 100:.2f}%")

    # Визуализация
    indices = random.sample(range(len(testset)), 100)
    subset = torch.utils.data.Subset(testset, indices)
    sample_loader = torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False)
    images, labels = next(iter(sample_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = conv(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    fig = plt.figure(figsize=(20, 10))
    for idx in range(100):
        ax = fig.add_subplot(10, 10, idx + 1, xticks=[], yticks=[])
        img = images[idx] / 2 + 0.5
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f"{classes[predicted[idx]]}", color=("green" if predicted[idx] == labels[idx] else "red"), fontsize=6)
    plt.suptitle("100 изображений: зелёный - верно, красный - ошибка", fontsize=14)
    plt.tight_layout()
    plt.show()
