import math

def save_coefficients(filename):
    # Рівняння: x^3 - x^2 + 2x + 4 = 0
    coeffs = [4.0, 2.0, -1.0, 1.0] 
    with open(filename, "w") as f:
        for c in coeffs: f.write(f"{c}\n")

def read_coefficients(filename):
    coeffs = []
    with open(filename, "r") as f:
        for line in f: coeffs.append(float(line.strip()))
    return coeffs

def newton_horner(coeffs, x0, eps, max_iter=100):
    print(f"\n--- Пункт 8: Дійсний корінь (Ньютон-Горнер) ---")
    m = len(coeffs) - 1
    x_curr = x0
    for iteration in range(1, max_iter + 1):
        b = [0] * (m + 1)
        b[m] = coeffs[m]
        for i in range(m - 1, -1, -1): b[i] = coeffs[i] + x_curr * b[i + 1]
        c = [0] * (m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1): c[i] = b[i] + x_curr * c[i + 1]
        
        if c[1] == 0: break
        x_next = x_curr - b[0] / c[1]
        if abs(x_next - x_curr) < eps:
            print(f"Корінь: x = {x_next:.6f}, Ітерацій: {iteration}")
            return x_next
        x_curr = x_next
    return None

def lin_method(coeffs, alpha0, beta0, eps, max_iter=100):
    print(f"\n--- Пункт 9: Комплексні корені (Метод Ліна) ---")
    a = coeffs
    alpha_c, beta_c = alpha0, beta0
    
    for i in range(1, max_iter + 1):
        p = -2 * alpha_c
        q = alpha_c**2 + beta_c**2
        b3 = a[3]
        b2 = a[2] - p * b3
        
        if abs(b2) < 1e-10: b2 = 1e-10 # Захист від ділення на нуль
        
        q_n = a[0] / b2
        p_n = (a[1] * b2 - a[0] * b3) / (b2**2)
        alpha_n = -p_n / 2.0
        
        # Обчислюємо нову бета, захищаючи від мінуса під коренем
        val = q_n - alpha_n**2
        beta_n = math.sqrt(abs(val)) 
        
        print(f"Ітерація {i}: alpha = {alpha_n:.4f}, beta = {beta_n:.4f}")
        
        if abs(alpha_n - alpha_c) < eps and abs(beta_n - beta_c) < eps:
            print(f"\nКомплексні корені знайдено: x = {alpha_n:.6f} ± i*{beta_n:.6f}")
            print(f"Кількість ітерацій: {i}")
            return alpha_n, beta_n
        alpha_c, beta_c = alpha_n, beta_n
    print("Метод не досяг точності.")

if __name__ == "__main__":
    eps = 1e-10
    save_coefficients("poly_coeffs.txt")
    c = read_coefficients("poly_coeffs.txt")
    newton_horner(c, x0=-2.0, eps=eps)
    # Змінюємо наближення на 1.2 та 1.5 — так воно точно збіжиться до 1 ± i*1.732
    lin_method(c, alpha0=1.2, beta0=1.5, eps=eps)