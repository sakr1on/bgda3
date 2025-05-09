import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загружаем датасет
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Делим данные на обучающую (70%) и тестовую (30%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Перебор параметров m и n в hidden_layer_sizes=(m,n)
min_error = float('inf')
best_mn = (0, 0)
best_clf = None

for m in range(1, 11):
    for n in range(1, 11):
        clf = MLPClassifier(hidden_layer_sizes=(m, n), max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        error = np.mean(y_pred != y_test)  # Средняя ошибка
        if error < min_error:
            min_error = error
            best_mn = (m, n)
            best_clf = clf

print(f"Лучшая конфигурация hidden_layer_sizes: {best_mn}")
print(f"Средняя ошибка на тестовой выборке: {min_error:.4f}")

# Предсказания для лучшей модели
y_pred_best = best_clf.predict(X_test)

# Визуализация (выберем 2 признака: 0 - длина чашелистика, 2 - длина лепестка)
feature_x = 0
feature_y = 2

plt.figure(figsize=(12, 5))

# График 1: Истинные метки
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, feature_x], X_test[:, feature_y], c=y_test, cmap='viridis', edgecolor='k', s=80)
plt.title('Реальные сорта ирисов')
plt.xlabel(feature_names[feature_x])
plt.ylabel(feature_names[feature_y])

# График 2: Предсказанные метки
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, feature_x], X_test[:, feature_y], c=y_pred_best, cmap='viridis', edgecolor='k', s=80)
plt.title('Предсказанные сорта ирисов')
plt.xlabel(feature_names[feature_x])
plt.ylabel(feature_names[feature_y])

plt.tight_layout()
plt.show()
