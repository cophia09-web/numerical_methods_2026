import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ПУНКТ 1: Задаємо систему нелінійних рівнянь та цільову функцію
# Рівняння 1: x1^2 + x2^2 - 4 = 0 (Коло радіусом 2)
# Рівняння 2: x2 - x1^2 = 0       (Парабола)
# =====================================================================

def f1(x):
    return x[0]**2 + x[1]**2 - 4

def f2(x):
    return x[1] - x[0]**2

# Цільова функція (сума квадратів рівнянь)
def objective_function(x):
    return f1(x)**2 + f2(x)**2

# Функція для побудови графіків рівнянь
def plot_equations(start_point, final_point, trajectory):
    x_vals = np.linspace(-3, 3, 400)
    y_vals = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Рівняння для контурів
    Z1 = X**2 + Y**2 - 4
    Z2 = Y - X**2
    
    plt.figure(figsize=(8, 8))
    # Малюємо коло (f1 = 0) та параболу (f2 = 0)
    plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
    
    # Додаємо легенду вручну, бо contour не підтримує label напряму
    plt.plot([], [], color='blue', label='$x_1^2 + x_2^2 - 4 = 0$')
    plt.plot([], [], color='red', label='$x_2 - x_1^2 = 0$')
    
    # Малюємо початкову точку, кінцеву і траєкторію
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    plt.plot(traj_x, traj_y, color='green', linestyle='--', alpha=0.6, label='Траєкторія пошуку')
    plt.scatter(start_point[0], start_point[1], color='orange', s=100, label='Початкова точка', zorder=5)
    plt.scatter(final_point[0], final_point[1], color='black', s=100, marker='*', label='Знайдений розв\'язок', zorder=5)
    
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.legend()
    plt.title("Графіки системи нелінійних рівнянь (Пункт 1)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

# =====================================================================
# ПУНКТ 2: Досліджуючий пошук (алгоритм Хука-Дживса)
# =====================================================================
def exploratory_search(x_start, delta):
    x = np.copy(x_start)
    n = len(x)
    
    for i in range(n):
        f_old = objective_function(x)
        
        # Крок вперед
        x[i] += delta[i]
        if objective_function(x) < f_old:
            continue
            
        # Крок назад (віднімаємо 2*дельта, бо ми вже зробили крок вперед)
        x[i] -= 2 * delta[i]
        if objective_function(x) < f_old:
            continue
            
        # Повертаємо координату назад, якщо не знайшли кращої точки
        x[i] += delta[i]
            
    return x

# =====================================================================
# Основна функція методу Хука-Дживса з детальним виводом
# =====================================================================
def hooke_jeeves(x0, delta0, eps1, eps2, q=2, p=2):
    x_base = np.copy(x0)
    delta = np.copy(delta0)
    trajectory = [x_base.copy()]
    
    step_count = 0
    print("\n--- ПОЧАТОК ПОШУКУ МЕТОДОМ ХУКА-ДЖИВСА ---")
    print(f"Початкова точка: X(0) = {x0}, Ф(X) = {objective_function(x0):.6f}")
    
    while True:
        step_count += 1
        print(f"\n[Ітерація {step_count}]")
        print(f"Поточний розмір кроку: delta = {delta}")
        
        # 1. Досліджуючий пошук
        x_new = exploratory_search(x_base, delta)
        print(f"  Після досліджуючого пошуку: X = {x_new}, Ф(X) = {objective_function(x_new):.6f}")
        
        if objective_function(x_new) < objective_function(x_base):
            # 2. Пошук по зразку
            print("  Функція зменшилась. Робимо пошук по зразку!")
            while True:
                x_pattern = x_new + p * (x_new - x_base)
                x_base = x_new.copy()
                trajectory.append(x_base.copy())
                
                # Досліджуючий пошук навколо зразка
                x_new = exploratory_search(x_pattern, delta)
                print(f"    Точка зразка: X = {x_new}, Ф(X) = {objective_function(x_new):.6f}")
                
                # Перевірка умов зупинки (Пункт 4 та 5)
                if (np.linalg.norm(x_new - x_base) < eps1 or 
                    abs(objective_function(x_new) - objective_function(x_base)) < eps2):
                    print(f"\n--- ДОСЯГНУТО КРИТЕРІЙ ЗУПИНКИ ---")
                    return x_new, trajectory, step_count
                
                if objective_function(x_new) >= objective_function(x_base):
                    print("    Пошук по зразку не дав результату, повертаємось до базисної точки.")
                    break
        else:
            # Зменшення кроку
            if np.max(delta) < eps1:
                print(f"\n--- ДОСЯГНУТО КРИТЕРІЙ ЗУПИНКИ (крок став меншим за eps1) ---")
                return x_base, trajectory, step_count
            print("  Функція не зменшилась. Зменшуємо крок!")
            delta /= q

# =====================================================================
# ПУНКТ 4: Виконання програми
# =====================================================================
if __name__ == "__main__":
    # Задаємо параметри за методичкою
    x_start = np.array([1.0, 1.0])       # Початкове наближення X(0)
    delta_start = np.array([0.5, 0.5])   # Величина початкового кроку
    epsilon1 = 0.001                     # Критерій зупинки по координатах
    epsilon2 = 0.001                     # Критерій зупинки по функції
    
    # Запускаємо метод
    final_solution, path, steps = hooke_jeeves(x_start, delta_start, epsilon1, epsilon2)
    
    # =====================================================================
    # ПУНКТ 5: Вивід у файл та підсумки
    # =====================================================================
    with open("trajectory.txt", "w", encoding="utf-8") as f:
        f.write("Координати точок траєкторії спуску (Пункт 5):\n")
        f.write("-" * 40 + "\n")
        for i, point in enumerate(path):
            f.write(f"Крок {i}: X1 = {point[0]:.6f}, X2 = {point[1]:.6f}, Ф(X) = {objective_function(point):.8f}\n")
            
    print("\n" + "="*50)
    print("ФІНАЛЬНІ РЕЗУЛЬТАТИ:")
    print(f"1. Знайдений розв'язок: X1 = {final_solution[0]:.6f}, X2 = {final_solution[1]:.6f}")
    print(f"2. Значення цільової функції Ф(X): {objective_function(final_solution):.8f}")
    print(f"3. Кількість кроків (ітерацій): {steps}")
    print("4. Координати траєкторії збережено у файл 'trajectory.txt'.")
    print("="*50)
    
    # Будуємо графік наостанок
    plot_equations(x_start, final_solution, path)