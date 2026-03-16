import csv
import numpy as np
import matplotlib.pyplot as plt

# --- 1. ЗЧИТУВАННЯ ДАНИХ (Пункт 1) ---
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Objects']))
            y.append(float(row['FPS']))
    return np.array(x), np.array(y)

# Зчитуємо наші дані [cite: 479]
x_data, y_data = read_data("data.csv")

# --- 2. МАТЕМАТИКА: ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (Пункт 2) ---
# Розділені різниці потрібні для побудови полінома Ньютона [cite: 194-200, 480]
def divided_differences(x, y):
    n = len(y)
    # Створюємо порожню таблицю (матрицю) з нулями
    table = np.zeros([n, n])
    table[:, 0] = y # Перший стовпець - це просто значення Y
    
    # Заповнюємо таблицю за формулою [cite: 200]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
            
    # Нам потрібен тільки верхній рядок таблиці (це і є коефіцієнти)
    return table[0, :]

# Отримуємо коефіцієнти для нашого полінома
coefs = divided_differences(x_data, y_data)

# --- 3. ПОБУДОВА ПОЛІНОМА НЬЮТОНА (Пункт 3) ---
# Функція, яка рахує значення FPS для будь-якої кількості об'єктів 
def newton_polynomial(coefs, x_data, x_val):
    n = len(x_data)
    result = coefs[0]
    
    # Рахуємо w_k(x) = (x - x0)*(x - x1)... [cite: 344]
    for i in range(1, n):
        term = coefs[i]
        for j in range(i):
            term *= (x_val - x_data[j])
        result += term
    return result

# Прогноз FPS для 1000 об'єктів [cite: 476, 481]
fps_1000 = newton_polynomial(coefs, x_data, 1000)
print(f"Прогнозований FPS для 1000 об'єктів: {fps_1000:.2f}")

# Шукаємо мінімальну кількість об'єктів для FPS >= 60 [cite: 477]
# Проходимо циклом від 100 до 1600 з кроком 1
for obj_count in range(100, 1600):
    current_fps = newton_polynomial(coefs, x_data, obj_count)
    if current_fps < 60:
        print(f"FPS падає нижче 60 при кількості об'єктів: {obj_count}")
        break

# --- 4. ВІЗУАЛІЗАЦІЯ (Пункт 4) ---
# Створюємо масив точок для плавного графіка [cite: 348, 482]
x_smooth = np.linspace(min(x_data), max(x_data), 200)
y_smooth = [newton_polynomial(coefs, x_data, x) for x in x_smooth]

plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, 'b-', label='Інтерполяційна крива Ньютона')
plt.plot(x_data, y_data, 'ro', label='Експериментальні дані (з CSV)')
plt.plot(1000, fps_1000, 'g*', markersize=12, label=f'Прогноз для 1000 (FPS={fps_1000:.1f})')

plt.title("Залежність FPS від кількості об'єктів (Варіант 5)")
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.axhline(y=60, color='r', linestyle='--', label='Межа комфортної гри (60 FPS)')
plt.grid(True)
plt.legend()


# --- 5. ДОСЛІДЖЕННЯ ПОХИБОК ДЛЯ 5, 10, 20 ВУЗЛІВ (Пункт 5) ---
# Оскільки у нас всього 5 точок, ми генеруємо нові "штучні" точки 
# на основі нашого полінома, щоб виконати вимогу методички [cite: 483-484, 490].
plt.figure(figsize=(10, 6))

for n_nodes in [5, 10, 20]:
    # Генеруємо n вузлів [cite: 350]
    x_nodes = np.linspace(min(x_data), max(x_data), n_nodes)
    y_nodes = [newton_polynomial(coefs, x_data, x) for x in x_nodes] # Беремо значення з "еталону"
    
    # Рахуємо нові коефіцієнти для цієї кількості вузлів
    new_coefs = divided_differences(x_nodes, y_nodes)
    y_approx = [newton_polynomial(new_coefs, x_nodes, x) for x in x_smooth]
    
    # Рахуємо похибку [cite: 347]
    error = np.abs(np.array(y_smooth) - np.array(y_approx))
    plt.plot(x_smooth, error, label=f'Похибка ({n_nodes} вузлів)')

plt.title("Дослідження ефекту Рунге та похибок [cite: 491]")
plt.xlabel("Кількість об'єктів")
plt.ylabel("Похибка ε(x)")
plt.grid(True)
plt.legend()

plt.show()