import numpy as np
import random

# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ У ФАЙЛИ
# =====================================================================
def save_vector(filename, vector, header=""):
    # Зберігає вектор у файл (кожен елемент з нового рядка)
    with open(filename, "w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n")
        for val in vector:
            f.write(f"{val:.15f}\n")

def save_iteration_log(filename, log_data):
    # Зберігає історію ітерацій у вигляді зручної таблички [cite: 246-247]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Ітераційне уточнення розв'язку:\n")
        f.write("№ ітерації | Похибка (eps)\n")
        f.write("-" * 30 + "\n")
        for it, err in log_data:
            f.write(f"Ітерація {it:2}: {err:.2e}\n")

def save_lu(L, U):
    # Записує дві матриці (L та U) поруч у один файл [cite: 242]
    np.savetxt("matrix_LU.txt", np.hstack((L, U)), fmt='%.6f')

# =====================================================================
# ПУНКТ 1: ГЕНЕРАЦІЯ ТА ЗБЕРЕЖЕННЯ ПОЧАТКОВИХ ДАНИХ
# =====================================================================
def generate_and_save_data(n=100, x_val=2.5):
    print("Пункт 1: Генеруємо матрицю А та вектор B...")
    A = np.zeros((n, n))
    
    # Заповнюємо матрицю випадковими числами від -10 до 10 [cite: 237]
    for i in range(n):
        for j in range(n):
            A[i][j] = random.uniform(-10, 10)
    
    # Задаємо ідеальний розв'язок (всі x = 2.5) [cite: 238]
    X_exact = np.full(n, x_val)
    
    # Обчислюємо вектор вільних членів B = A * X [cite: 239-240]
    B = np.dot(A, X_exact)
    
    # Зберігаємо згенеровані дані у текстові файли [cite: 237, 240-241]
    np.savetxt("matrix_A.txt", A, fmt='%.6f')
    np.savetxt("vector_B.txt", B, fmt='%.6f')
    print("Дані успішно збережено у файли.\n")

# =====================================================================
# ПУНКТ 3: LU-РОЗКЛАД МАТРИЦІ
# =====================================================================
def lu_decomposition(A):
    n = len(A)
    # Створюємо порожні матриці L та U [cite: 208]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Діагональні елементи матриці U рівні 1 [cite: 209]
    for i in range(n):
        U[i][i] = 1.0 
        
    for k in range(n):
        # Шукаємо елементи k-го стовпця матриці L [cite: 210-212]
        for i in range(k, n):
            sum_L = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_L
            
        # Шукаємо елементи k-го рядка матриці U [cite: 210-212]
        for i in range(k + 1, n):
            sum_U = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_U) / L[k][k]
            
    return L, U

# =====================================================================
# ПУНКТ 4: РОЗВ'ЯЗОК СИСТЕМИ РІВНЯНЬ ЧЕРЕЗ L ТА U
# =====================================================================
def solve_lu(L, U, B):
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)
    
    # ЕТАП 1: Прямий хід (розв'язуємо LZ = B) [cite: 215-216]
    Z[0] = B[0] / L[0][0]
    for k in range(1, n):
        sum_Z = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_Z) / L[k][k]
        
    # ЕТАП 2: Зворотній хід (розв'язуємо UX = Z) [cite: 215, 217-218]
    X[n-1] = Z[n-1]
    for k in range(n-2, -1, -1):
        sum_X = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_X
        
    return X

# Функція для оцінки похибки
def calc_error(A, X, B):
    # Знаходимо нев'язку: R = A*X - B і беремо максимальне відхилення [cite: 222, 245]
    return np.max(np.abs(np.dot(A, X) - B))

# =====================================================================
# ГОЛОВНА ПРОГРАМА 
# =====================================================================

# 1. Готуємо початкові дані
generate_and_save_data(n=100)
A = np.loadtxt("matrix_A.txt")
B = np.loadtxt("vector_B.txt")

# 2. Робимо LU-розклад та зберігаємо його
print("Пункт 3: Виконуємо LU-розклад матриці А...")
L, U = lu_decomposition(A)
save_lu(L, U)

# 3. Знаходимо "чорновий" (початковий) розв'язок X0
X0 = solve_lu(L, U, B)
save_vector("solution_X0.txt", X0, "Початковий розв'язок (X0)")

# 4. Перевіряємо початкову похибку [cite: 244-245]
eps_initial = calc_error(A, X0, B)
print(f"Пункт 4: Похибка початкового розв'язку (eps) = {eps_initial:.2e}")

# =====================================================================
# ПУНКТ 5: ІТЕРАЦІЙНЕ УТОЧНЕННЯ РОЗВ'ЯЗКУ
# =====================================================================
print("\nПункт 5: Починаємо ітераційне уточнення розв'язку...")
eps_target = 1e-14 # Наша мета - точність 10^-14 [cite: 247]
X_current = np.copy(X0)
iteration_log = [] # Список для збереження історії ітерацій

# Цикл ітерацій (максимум 15, щоб не зависло) [cite: 230-233]
for iteration in range(1, 16):
    # КРОК 1: Знаходимо вектор нев'язки R = B - A * X_current [cite: 222]
    R = B - np.dot(A, X_current)
    
    # КРОК 2: Розв'язуємо систему A * Delta_X = R через готові L та U [cite: 228-229, 234-235]
    delta_X = solve_lu(L, U, R)
    
    # КРОК 3: Уточнюємо розв'язок [cite: 229]
    X_current = X_current + delta_X
    
    # КРОК 4: Оцінюємо нову похибку та записуємо в лог [cite: 230-231]
    current_error = calc_error(A, X_current, B)
    iteration_log.append((iteration, current_error))
    print(f"Ітерація {iteration}: похибка = {current_error:.2e}")
    
    # КРОК 5: Перевіряємо, чи досягли потрібної точності [cite: 232-233]
    if current_error <= eps_target:
        print(f"-> УСПІХ: Досягнуто задану точність eps_0 = 10^-14 за {iteration} ітерацій!")
        break
else:
    print("-> СТОП: Досягнуто межі ітерацій (машинна точність).")

# Зберігаємо фінальні результати уточнення у файли
save_vector("solution_final.txt", X_current, "Уточнений розв'язок (X_final)")
save_iteration_log("refinement_log.txt", iteration_log)

print("\nПрограму завершено. Всі файли досліджень створено успішно!")