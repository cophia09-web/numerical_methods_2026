import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 1. ПІДГОТОВКА ДАНИХ (Пункт 1) ---
# f(x) описує навантаження на сервер протягом 24 годин [cite: 543, 587-589]
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24 # Інтервал часу від 0 до 24 годин [cite: 554, 591]

# --- 2. ТОЧНЕ ЗНАЧЕННЯ (Пункт 2) ---
# Рахуємо "ідеальний" інтеграл для порівняння похибок [cite: 561-563]
I0, _ = quad(f, a, b)
print(f"Пункт 2: Точне значення інтегралу I0 = {I0:.6f}")

# --- 3. ФУНКЦІЯ СІМПСОНА (Пункт 3) ---
# Основний метод розрахунку площі під графіком [cite: 429, 497-498, 565]
def simpson(f, a, b, N):
    if N % 2 != 0: N += 1 # N має бути парним [cite: 492]
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # Формула: h/3 * (перший + останній + 4*непарні + 2*парні вузли) [cite: 498, 565]
    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return integral

# --- 4. ДОСЛІДЖЕННЯ ТОЧНОСТІ (Пункт 4) ---
# Шукаємо, при якому N похибка стане меншою за 1e-12 [cite: 567-568]
N_values = np.arange(10, 1002, 2)
errors = []
N_opt = 0

for N in N_values:
    res = simpson(f, a, b, N)
    err = abs(res - I0)
    errors.append(err)
    if err < 1e-12 and N_opt == 0:
        N_opt = N

eps_opt = abs(simpson(f, a, b, N_opt) - I0)
print(f"Пункт 4: Оптимальне N_opt = {N_opt}, Точність = {eps_opt:.2e}")

# --- 5. РОЗРАХУНОК ПРИ N0 (Пункт 5) ---
# Беремо невелике N0 для перевірки методів уточнення [cite: 569-570]
N0 = 8 
I_N0 = simpson(f, a, b, N0)
print(f"Пункт 5: При N0 = {N0}, Похибка = {abs(I_N0 - I0):.2e}")

# --- 6. МЕТОД РУНГЕ-РОМБЕРГА (Пункт 6) ---
# Уточнюємо результат, використовуючи N0 та N0/2 [cite: 505-513, 571-573]
I_N0_2 = simpson(f, a, b, N0 // 2)
# Для методу Сімпсона знаменник дорівнює 15 [cite: 573]
I_R = I_N0 + (I_N0 - I_N0_2) / 15
print(f"Пункт 6: Похибка Рунге-Ромберга = {abs(I_R - I0):.2e}")

# --- 7. МЕТОД ЕЙТКЕНА (Пункт 7) ---
# Уточнюємо через три кроки: N0, N0/2, N0/4 [cite: 514-524, 576-578]
I_N0_4 = simpson(f, a, b, N0 // 4)
# Формула Ейткена та оцінка порядку точності p [cite: 522, 524, 578]
I_E = (I_N0_2**2 - I_N0 * I_N0_4) / (2 * I_N0_2 - (I_N0 + I_N0_4))
p_val = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))
print(f"Пункт 7: Похибка Ейткена = {abs(I_E - I0):.2e}, Порядок p = {p_val:.2f}")

# --- 8. АДАПТИВНИЙ АЛГОРИТМ (Пункт 9) ---
# Розумний алгоритм, що сам ділить відрізки там, де функція "стрибає" [cite: 525-540, 581]
def adaptive_simpson(f, a, b, tol):
    c = (a + b) / 2
    h = b - a
    I1 = (h / 6) * (f(a) + 4 * f(c) + f(b))
    d, e = (a + c) / 2, (c + b) / 2
    I2 = (h / 12) * (f(a) + 4 * f(d) + f(c)) + (h / 12) * (f(c) + 4 * f(e) + f(b))
    if abs(I1 - I2) <= tol: return I2
    return adaptive_simpson(f, a, c, tol / 2) + adaptive_simpson(f, c, b, tol / 2)

I_adapt = adaptive_simpson(f, a, b, 1e-6)
print(f"Пункт 9: Похибка адаптивного методу = {abs(I_adapt - I0):.2e}")

# --- 9. ВІЗУАЛІЗАЦІЯ (Два графіки в одному вікні) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Лівий графік: Навантаження f(x) [cite: 583-601]
x_f = np.linspace(a, b, 400)
axes[0].plot(x_f, f(x_f), color='teal')
axes[0].set_title('Графік навантаження на сервер')
axes[0].set_xlabel('Час x (год)'); axes[0].set_ylabel('Навантаження f(x)'); axes[0].grid()

# Правий графік: Похибка [cite: 567]
axes[1].plot(N_values, errors, color='blue')
axes[1].axvline(N_opt, color='red', linestyle='--', label=f'N_opt = {N_opt}')
axes[1].set_yscale('log')
axes[1].set_title('Залежність похибки від кількості розбиттів N')
axes[1].set_xlabel('Кількість розбиттів (N)'); axes[1].set_ylabel(r'Похибка $\epsilon(N)$'); axes[1].grid(); axes[1].legend()

plt.tight_layout()
plt.show()