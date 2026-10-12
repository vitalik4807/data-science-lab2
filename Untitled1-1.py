import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

#фіксація генератора випадкових чисел для відтворюваності результатів
np.random.seed(42)

# Крок 1. Генерація випадкового набору даних (1000 значень)
# Генеруємо ознаку X (від 0 до 10) та залежну змінну Y з додаванням шуму
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = 2 * np.sin(X).ravel() + np.random.normal(0, 0.5, 1000)

print(f"Згенеровано {len(X)} записів даних.")

# Крок 2: Нормалізація значень
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Крок 3: Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(
    f"Розмір навчальної вибірки: {len(X_train)}, розмір тестової вибірки: {len(X_test)}"
)

# Крок 4 & 5: Навчання KNN-регресора з різними K та вибір найкращого K
k_values = range(1, 31)
mse_list = []
r2_list = []
best_k = None
best_mse = float("inf")
best_model = None

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_list.append(mse)
    r2_list.append(r2)

    if mse < best_mse:
        best_mse = mse
        best_k = k
        best_model = knn

print(f"\nНайкраще значення K: {best_k}")
print(f"Найменша середньоквадратична помилка (MSE): {best_mse:.4f}")
print(
    f"Коефіцієнт детермінації (R2) для K={best_k}: {r2_list[best_k - 1]:.4f}"
)

# Крок 6: Візуалізація отриманих рішень
plt.figure(figsize=(14, 5))
# Графік 1: Залежність MSE від значення K
plt.subplot(1, 2, 1)
plt.plot(k_values, mse_list, marker="o", color="b", linestyle="--")
plt.axvline(
    x=best_k,
    color="r",
    linestyle=":",
    label=f"Оптимальне K={best_k} (MSE={best_mse:.4f})",
)
plt.title("Залежність помилки MSE від значення K")
plt.xlabel("Кількість сусідів (K)")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.legend()

# Графік 2: Апроксимація регресором для оптимального K
# Денормалізація даних для наочності відображення
X_test_orig = scaler_X.inverse_transform(X_test)
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

X_grid = np.linspace(0, 10, 1000).reshape(-1, 1)
X_grid_scaled = scaler_X.transform(X_grid)
y_grid_pred_scaled = best_model.predict(X_grid_scaled)
y_grid_pred = scaler_y.inverse_transform(
    y_grid_pred_scaled.reshape(-1, 1)
).ravel()

plt.subplot(1, 2, 2)
plt.scatter(
    X_test_orig, y_test_orig, color="gray", alpha=0.6, label="Тестові дані"
)
plt.plot(
    X_grid,
    y_grid_pred,
    color="red",
    linewidth=2,
    label=f"KNN регресія (K={best_k})",
)
plt.title(f"Результат KNN регресії для K={best_k}")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()