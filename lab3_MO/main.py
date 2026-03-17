import csv
import numpy as np
import matplotlib.pyplot as plt

# --- 1. ЗЧИТУВАННЯ ВХІДНИХ ДАНИХ (Пункт 1) ---
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return np.array(x), np.array(y)

x_data, y_data = read_data("data.csv")
n_points = len(x_data)

# --- 2. ФУНКЦІЇ МЕТОДУ НАЙМЕНШИХ КВАДРАТІВ (Пункт 2) ---

# Формуємо матрицю A (ліва частина системи) [cite: 567, 675-680]
def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x**(i + j))
    return A

# Формуємо вектор b (права частина системи) [cite: 568, 681-685]
def form_vector(x, y, m):
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x**i))
    return b

# Метод Гауса з вибором головного елемента по стовпцю [cite: 576-611, 686-695]
def gauss_solve(A, b):
    n = len(b)
    A = A.copy()
    b = b.copy()
    
    # Прямий хід (робимо матрицю трикутною)
    for k in range(n - 1):
        # Шукаємо найбільший елемент у стовпці (головний елемент)
        max_row = k + np.argmax(np.abs(A[k:, k]))
        # Міняємо рядки місцями
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            
        # Обнуляємо елементи під головним
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            
    # Зворотний хід (знаходимо самі коефіцієнти)
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i+1:] * x_sol[i+1:])) / A[i, i]
    return x_sol

# Обчислення значення многочлена для будь-якого x [cite: 548, 696-700]
def polynomial(x_vals, coef):
    y_poly = np.zeros(len(x_vals))
    for i in range(len(coef)):
        y_poly += coef[i] * (x_vals**i)
    return y_poly

# Обчислення дисперсії (похибки моделі) [cite: 571, 701-703]
def variance(y_true, y_approx):
    # Формула дисперсії: корінь з (сума квадратів відхилень / кількість точок)
    return np.sqrt(np.sum((y_approx - y_true)**2) / len(y_true))

# --- 3. ВИБІР ОПТИМАЛЬНОГО СТУПЕНЯ ПОЛІНОМА (Пункт 3) ---
variances = []
# Перевіряємо степені від 1 до 10 
degrees = range(1, 11) 

for m in degrees:
    A = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x_data, coef)
    var = variance(y_data, y_approx)
    variances.append(var)
    print(f"Степінь m={m:2d} | Дисперсія: {var:.4f}")

# Шукаємо мінімальну дисперсію [cite: 624-626, 711]
optimal_m = degrees[np.argmin(variances)]
print(f"\n---> Оптимальний степінь многочлена: m = {optimal_m}")

# --- 4. ПОБУДОВА АПРОКСИМАЦІЇ ТА ПРОГНОЗУ (Пункт 4-6) ---
# Рахуємо фінальні коефіцієнти для найкращого m
A_opt = form_matrix(x_data, optimal_m)
b_opt = form_vector(x_data, y_data, optimal_m)
coef_opt = gauss_solve(A_opt, b_opt)

# Прогноз на наступні 3 місяці (25, 26, 27) [cite: 639, 722]
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef_opt)
print(f"Прогноз температур на 25, 26, 27 місяці: {y_future.round(2)}")

# --- 5. ВІЗУАЛІЗАЦІЯ (Пункт 5) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Графік 1: Залежність дисперсії від степені 
axes[0].plot(degrees, variances, marker='o', color='purple')
axes[0].set_title("Дисперсія від степені (m)")
axes[0].set_xlabel("Степінь многочлена m")
axes[0].set_ylabel("Дисперсія $\delta$")
axes[0].grid(True)

# Графік 2: Фактичні дані, апроксимація та прогноз [cite: 637]
x_fine = np.linspace(min(x_data), max(x_future), 200)
y_fine = polynomial(x_fine, coef_opt)

axes[1].plot(x_data, y_data, 'ro', label='Фактичні дані')
axes[1].plot(x_fine, y_fine, 'b-', label=f'МНК (m={optimal_m})')
axes[1].plot(x_future, y_future, 'g*', markersize=10, label='Прогноз')
axes[1].set_title("Апроксимація та екстраполяція")
axes[1].set_xlabel("Місяць")
axes[1].set_ylabel("Температура")
axes[1].legend()
axes[1].grid(True)

# Графік 3: Похибка на дрібній сітці [cite: 627-629, 638]
h1 = (max(x_data) - min(x_data)) / (20 * n_points)
x_err = np.arange(min(x_data), max(x_data), h1)
# Лінійна інтерполяція для отримання f(x) між вузлами
y_interp = np.interp(x_err, x_data, y_data) 
y_approx_err = polynomial(x_err, coef_opt)
error = np.abs(y_interp - y_approx_err) # Похибка epsilon = |f(x) - phi(x)|

axes[2].plot(x_err, error, 'orange')
axes[2].fill_between(x_err, error, color='orange', alpha=0.3)
axes[2].set_title(f"Похибка $\epsilon(x)$ для m={optimal_m}")
axes[2].set_xlabel("Місяць")
axes[2].set_ylabel("Похибка")
axes[2].grid(True)

plt.tight_layout()
plt.show()