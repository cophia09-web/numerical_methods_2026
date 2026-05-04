import numpy as np
import random

# =====================================================================
# ПУНКТ 2: ДОПОМІЖНІ ФУНКЦІЇ ЗЧИТУВАННЯ ТА ОБЧИСЛЕНЬ [cite: 318]
# =====================================================================

def read_matrix(filename):
    """Зчитування матриці А з текстового файлу [cite: 318]"""
    return np.loadtxt(filename)

def read_vector(filename):
    """Зчитування вектора В з текстового файлу [cite: 318]"""
    return np.loadtxt(filename)

def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор [cite: 318]"""
    return np.dot(A, X)

def vec_norm(V):
    """Обчислення норми вектора (максимальний за модулем елемент) [cite: 318]"""
    return np.max(np.abs(V))

def mat_norm(A):
    """Обчислення норми матриці (максимальна сума модулів по рядках) [cite: 318]"""
    return np.max(np.sum(np.abs(A), axis=1))

# =====================================================================
# ПУНКТ 1: ГЕНЕРАЦІЯ ТА ЗБЕРЕЖЕННЯ ДАНИХ
# =====================================================================

def generate_and_save_data(n=100, x_val=2.5):
    print("Пункт 1: Генеруємо матрицю А та вектор B...")
    A = np.zeros((n, n))
    
    # Генеруємо матрицю з ДІАГОНАЛЬНИМ ПЕРЕВАЖАННЯМ [cite: 314]
    # Це обов'язкова умова для збіжності методів Якобі та Зейделя[cite: 290].
    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                A[i][j] = random.uniform(-10, 10)
                row_sum += abs(A[i][j])
        # Робимо діагональний елемент більшим за суму інших елементів у рядку
        A[i][i] = row_sum + random.uniform(1, 10) 
    
    # Зберігаємо отриману матрицю А [cite: 315]
    np.savetxt("matrix_A.txt", A, fmt='%.6f')
    
    # Задаємо точний розв'язок (всі x_i = 2.5) та обчислюємо вектор B [cite: 315]
    X_exact = np.full(n, x_val)
    B = mat_vec_mult(A, X_exact)
    
    # Зберігаємо отриманий вектор В [cite: 316]
    np.savetxt("vector_B.txt", B, fmt='%.6f')
    print("Дані успішно збережено у файли 'matrix_A.txt' та 'vector_B.txt'.\n")

# =====================================================================
# ПУНКТ 2: ІТЕРАЦІЙНІ МЕТОДИ РОЗВ'ЯЗКУ [cite: 317, 318]
# =====================================================================

def solve_simple_iteration(A, B, X0, eps):
    """Розв'язок методом простої ітерації [cite: 318]"""
    n = len(B)
    X_curr = np.copy(X0)
    X_next = np.zeros(n)
    iterations = 0
    
    # Вибираємо параметр tau для збіжності: 0 < tau < 2/||A|| [cite: 281]
    tau = 1.0 / mat_norm(A) 
    
    while True:
        iterations += 1
        # Формула: X^(k+1) = X^(k) - tau * (A * X^(k) - B) [cite: 262]
        AX = mat_vec_mult(A, X_curr)
        for i in range(n):
            X_next[i] = X_curr[i] - tau * (AX[i] - B[i])
            
        # Перевірка умови закінчення: ||X^(k+1) - X^(k)|| <= eps [cite: 308-309]
        if vec_norm(X_next - X_curr) <= eps:
            break
        X_curr = np.copy(X_next)
        
    return X_next, iterations

def solve_jacobi(A, B, X0, eps):
    """Розв'язок методом Якобі [cite: 318]"""
    n = len(B)
    X_curr = np.copy(X0)
    X_next = np.zeros(n)
    iterations = 0
    
    while True:
        iterations += 1
        # Формула Якобі у розгорнутому вигляді 
        for i in range(n):
            s = sum(A[i][j] * X_curr[j] for j in range(n) if j != i)
            X_next[i] = (B[i] - s) / A[i][i]
            
        # Перевірка умови закінчення [cite: 308-309]
        if vec_norm(X_next - X_curr) <= eps:
            break
        X_curr = np.copy(X_next)
        
    return X_next, iterations

def solve_seidel(A, B, X0, eps):
    """Розв'язок методом Зейделя [cite: 318]"""
    n = len(B)
    X_curr = np.copy(X0)
    X_next = np.copy(X0) # Важливо: Зейдель використовує оновлені значення одразу
    iterations = 0
    
    while True:
        iterations += 1
        # Формула Зейделя у розгорнутій формі 
        for i in range(n):
            # Сума з вже оновленими значеннями (j < i)
            s1 = sum(A[i][j] * X_next[j] for j in range(i))
            # Сума зі старими значеннями (j > i)
            s2 = sum(A[i][j] * X_curr[j] for j in range(i + 1, n))
            
            X_next[i] = (B[i] - s1 - s2) / A[i][i]
            
        # Перевірка умови закінчення [cite: 308-309]
        if vec_norm(X_next - X_curr) <= eps:
            break
        X_curr = np.copy(X_next)
        
    return X_next, iterations

# =====================================================================
# ГОЛОВНА ПРОГРАМА
# =====================================================================

# 1. Генеруємо та зчитуємо дані
generate_and_save_data(n=100, x_val=2.5)
A = read_matrix("matrix_A.txt")
B = read_vector("vector_B.txt")

# 3. Задаємо початкове наближення X0 = 1.0 та цільову точність [cite: 319, 321]
n = len(B)
X0 = np.full(n, 1.0)
eps_0 = 1e-13

print("Пункт 3-4: Знаходимо розв'язки ітераційними методами...")
print(f"Початкове наближення: всі x_i = 1.0, Точність: {eps_0}\n")

# 4. Знаходимо розв'язок кожним із методів та рахуємо ітерації [cite: 320, 321]

# Метод простої ітерації
X_simp, it_simp = solve_simple_iteration(A, B, X0, eps_0)
err_simp = vec_norm(mat_vec_mult(A, X_simp) - B)
print(f"1. Метод простої ітерації:")
print(f"   Кількість ітерацій: {it_simp}")
print(f"   Фактична нев'язка: {err_simp:.2e}\n")

# Метод Якобі
X_jac, it_jac = solve_jacobi(A, B, X0, eps_0)
err_jac = vec_norm(mat_vec_mult(A, X_jac) - B)
print(f"2. Метод Якобі:")
print(f"   Кількість ітерацій: {it_jac}")
print(f"   Фактична нев'язка: {err_jac:.2e}\n")

# Метод Зейделя
X_seid, it_seid = solve_seidel(A, B, X0, eps_0)
err_seid = vec_norm(mat_vec_mult(A, X_seid) - B)
print(f"3. Метод Зейделя:")
print(f"   Кількість ітерацій: {it_seid}")
print(f"   Фактична нев'язка: {err_seid:.2e}\n")

print("Розрахунки завершено успішно!")