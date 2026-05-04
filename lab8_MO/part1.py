import math
import cmath

# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ФУНКЦІЇ F(x) ТА ЇЇ ПОХІДНИХ
# =====================================================================
def f(x):
    # Наша нелінійна функція: F(x) = sin(x) - x/3
    return math.sin(x) - x / 3.0

def df(x):
    # Перша похідна F'(x) = cos(x) - 1/3
    return math.cos(x) - 1.0 / 3.0

def ddf(x):
    # Друга похідна F''(x) = -sin(x)
    return -math.sin(x)

def f_interp(x0, x1):
    # Розділена різниця першого порядку (для методу хорд)
    return (f(x1) - f(x0)) / (x1 - x0)

def f_interp2(x0, x1, x2):
    # Розділена різниця другого порядку (для методу парабол)
    return (f_interp(x1, x2) - f_interp(x0, x1)) / (x2 - x0)

# =====================================================================
# ПУНКТ 1: ТАБУЛЯЦІЯ ТА ВИДІЛЕННЯ КОРЕНІВ
# =====================================================================
def tabulate_and_find_roots(a, b, h):
    print("Пункт 1: Табуляція функції та пошук наближень коренів...")
    roots_approx = []
    
    with open("tabulation.txt", "w") as file:
        file.write(" x | F(x)\n")
        file.write("-" * 20 + "\n")
        
        x_prev = a
        f_prev = f(x_prev)
        file.write(f"{x_prev:5.2f} | {f_prev:.6f}\n")
        
        # Рухаємось з кроком h
        x_curr = a + h
        while x_curr <= b + h/2: # h/2 для компенсації похибки float
            f_curr = f(x_curr)
            file.write(f"{x_curr:5.2f} | {f_curr:.6f}\n")
            
            # Якщо функція змінила знак, значить корінь десь між x_prev та x_curr
            if f_prev * f_curr < 0:
                roots_approx.append((x_prev + x_curr) / 2.0)
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr += h
            
    print(f"Знайдено наближені корені: {roots_approx}")
    print("Результати табуляції збережено в 'tabulation.txt'.\n")
    return roots_approx

# =====================================================================
# ПУНКТ 2-3: ІТЕРАЦІЙНІ МЕТОДИ (Умова зупинки з пункту 3)
# =====================================================================

def is_converged(x_new, x_old, eps):
    # Критерій зупинки: |F(x)| < eps ТА |x_new - x_old| < eps
    return abs(f(x_new)) < eps and abs(x_new - x_old) < eps

def method_simple_iteration(x0, eps, max_iter=1000):
    # Метод простої ітерації (релаксації)
    print(f"--- Метод простої ітерації (x0={x0:.4f}) ---")
    x_curr = x0
    # Вибираємо tau так, щоб |1 + tau * F'(x)| < 1
    tau = -0.5 / df(x0) if df(x0) != 0 else -0.1 
    
    for i in range(1, max_iter + 1):
        x_next = x_curr + tau * f(x_curr) #
        if is_converged(x_next, x_curr, eps):
            print(f"Корінь: {x_next:.10f}, Ітерацій: {i}\n")
            return x_next, i
        x_curr = x_next
    print("Метод не збігся.\n")
    return None, max_iter

def method_newton(x0, eps, max_iter=100):
    # Метод Ньютона
    print(f"--- Метод Ньютона (x0={x0:.4f}) ---")
    x_curr = x0
    
    for i in range(1, max_iter + 1):
        if df(x_curr) == 0:
            break
        x_next = x_curr - f(x_curr) / df(x_curr) #
        if is_converged(x_next, x_curr, eps):
            print(f"Корінь: {x_next:.10f}, Ітерацій: {i}\n")
            return x_next, i
        x_curr = x_next
    print("Метод не збігся.\n")
    return None, max_iter

def method_chebyshev(x0, eps, max_iter=100):
    # Метод Чебишева
    print(f"--- Метод Чебишева (x0={x0:.4f}) ---")
    x_curr = x0
    
    for i in range(1, max_iter + 1):
        fx = f(x_curr)
        dfx = df(x_curr)
        ddfx = ddf(x_curr)
        
        if dfx == 0:
            break
        # Формула Чебишева
        x_next = x_curr - fx/dfx - (0.5 * (fx**2) * ddfx) / (dfx**3)
        if is_converged(x_next, x_curr, eps):
            print(f"Корінь: {x_next:.10f}, Ітерацій: {i}\n")
            return x_next, i
        x_curr = x_next
    print("Метод не збігся.\n")
    return None, max_iter

def method_chord(x0, x1, eps, max_iter=100):
    # Метод хорд
    print(f"--- Метод хорд (x0={x0:.4f}, x1={x1:.4f}) ---")
    xn_minus_1 = x0
    xn = x1
    
    for i in range(1, max_iter + 1):
        div_diff = f_interp(xn, xn_minus_1)
        if div_diff == 0:
            break
        # Формула хорд
        x_next = xn - f(xn) / div_diff
        if is_converged(x_next, xn, eps):
            print(f"Корінь: {x_next:.10f}, Ітерацій: {i}\n")
            return x_next, i
        xn_minus_1 = xn
        xn = x_next
    print("Метод не збігся.\n")
    return None, max_iter

# =====================================================================
# ГОЛОВНИЙ БЛОК ПРОГРАМИ (Пункт 4)
# =====================================================================
if __name__ == "__main__":
    # 1. Табуляція від 1.0 до 4.0 з кроком 0.1
    # Там наша функція sin(x) - x/3 має корінь (графік спадає)
    roots = tabulate_and_find_roots(1.0, 4.0, 0.1)
    
    eps_target = 1e-10 #
    
    if len(roots) >= 1:
        root1_approx = roots[0]
        print(f"Пункт 4: Досліджуємо перший корінь (наближення {root1_approx:.2f})...")
        
        # Запускаємо всі методи
        method_simple_iteration(root1_approx, eps_target)
        method_newton(root1_approx, eps_target)
        method_chebyshev(root1_approx, eps_target)
        # Для методу хорд потрібні дві стартові точки
        method_chord(root1_approx - 0.1, root1_approx, eps_target)
        
    print("Частину 1 завершено.")