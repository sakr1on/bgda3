import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score

# 1. Загрузка CIFAR-10 и вывод первых 10 изображений
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(os.getcwd(), download=True, transform=transform)
classes = dataset.classes

# Отображение первых 10 изображений
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

first_images = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
images, labels = next(iter(first_images))
imshow(torchvision.utils.make_grid(images))
print("Классы:", [classes[label] for label in labels])

# 2. Реализация MLP (перцептрона Розенблатта)
class RosenblattMLP(nn.Module):
    def __init__(self):
        super(RosenblattMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3 * 32 * 32, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# Гиперпараметры
cur_batch_size = 10
num_epochs = 5
learning_rate = 1e-4

# Загрузчик данных
trainloader = torch.utils.data.DataLoader(dataset, batch_size=cur_batch_size, shuffle=True, num_workers=1)

# Инициализация модели, функции потерь и оптимизатора
mlp = RosenblattMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

# 3. Обучение модели
print("\nОбучение началось...\n")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if (i + 1) % cur_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        if (i + 1) % 500 == 0:
            print(f"[Эпоха {epoch+1}, итерация {i+1}] средняя потеря: {running_loss / 500:.4f}")
            running_loss = 0.0

print("\nОбучение завершено.")

# 4. Подсчет accuracy на всем датасете
testloader = torch.utils.data.DataLoader(dataset, batch_size=cur_batch_size, shuffle=False)
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy модели на всём CIFAR-10: {accuracy * 100:.2f}%")

# 5. Визуализация 100 случайных изображений с реальными и предсказанными метками
import random

indices = random.sample(range(len(dataset)), 100)
samples = torch.utils.data.Subset(dataset, indices)
sample_loader = torch.utils.data.DataLoader(samples, batch_size=100, shuffle=False)
images, labels = next(iter(sample_loader))

with torch.no_grad():
    outputs = mlp(images)
    _, predicted = torch.max(outputs, 1)

# Визуализация
fig = plt.figure(figsize=(20, 10))
for idx in range(100):
    ax = fig.add_subplot(10, 10, idx + 1, xticks=[], yticks=[])
    img = images[idx] / 2 + 0.5  # де-нормализация, если была
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(f"{classes[predicted[idx]]}", color=("green" if predicted[idx] == labels[idx] else "red"), fontsize=6)
plt.suptitle("100 изображений: зелёный - верно, красный - ошибка", fontsize=14)
plt.tight_layout()
plt.show()
