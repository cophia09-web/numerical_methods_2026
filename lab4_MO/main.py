import numpy as np
import matplotlib.pyplot as plt

# --- 1. ПІДГОТОВКА (Пункт 1) ---
# Функція вологості та її аналітична похідна [cite: 870, 886, 919]
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_prime_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0 # Точка дослідження
exact_val = M_prime_exact(t0)

# --- 2. ДОСЛІДЖЕННЯ ОПТИМАЛЬНОГО КРОКУ (Пункт 2) ---
h_range = np.logspace(-20, 3, 100)
errors_h = []

for h in h_range:
    # Центральна різниця: (f(t+h) - f(t-h)) / 2h [cite: 899, 924]
    approx = (M(t0 + h) - M(t0 - h)) / (2 * h)
    errors_h.append(abs(approx - exact_val))

h0 = h_range[np.argmin(errors_h)] # Найкращий крок

# --- 3. ПОРІВНЯННЯ МЕТОДІВ (Пункт 3-7) ---
h_fixed = 0.001 # Фіксований крок 10^-3 [cite: 926]

# Обчислюємо похідну з різними кроками для уточнення [cite: 928, 935-937]
d_h = (M(t0 + h_fixed) - M(t0 - h_fixed)) / (2 * h_fixed)
d_2h = (M(t0 + 2*h_fixed) - M(t0 - 2*h_fixed)) / (4 * h_fixed)
d_4h = (M(t0 + 4*h_fixed) - M(t0 - 4*h_fixed)) / (8 * h_fixed)

# Метод Рунге-Ромберга (уточнення за h та 2h) [cite: 905, 930]
d_RR = d_h + (d_h - d_2h) / 3
# Метод Ейткена (уточнення за h, 2h, 4h) [cite: 910, 941]
d_E = (d_2h**2 - d_4h * d_h) / (2 * d_2h - (d_4h + d_h))

# Масив похибок для порівняння
r1 = abs(d_h - exact_val)   # Звичайна похибка
r2 = abs(d_RR - exact_val)  # Похибка Рунге-Ромберга
r3 = abs(d_E - exact_val)   # Похибка Ейткена

# --- 4. ВІЗУАЛІЗАЦІЯ ВСІХ ДАНИХ ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ГРАФІК 1: Модель вологості M(t) (як на стор. 8 методички) [cite: 872]
t_plot = np.linspace(0, 20, 200)
axes[0].plot(t_plot, M(t_plot), color='teal', label='M(t)')
axes[0].scatter([t0], [M(t0)], color='red', zorder=5, label=f'Точка t0={t0}')
axes[0].set_title("Модель вологості ґрунту")
axes[0].set_xlabel("Час t"); axes[0].set_ylabel("M(t)"); axes[0].grid(); axes[0].legend()

# ГРАФІК 2: Залежність похибки від кроку h (Пункт 2) [cite: 924]
axes[1].loglog(h_range, errors_h, color='blue')
axes[1].axvline(h0, color='red', linestyle='--', label=f'h0={h0:.1e}')
axes[1].set_title("Вибір оптимального кроку h")
axes[1].set_xlabel("Крок h"); axes[1].set_ylabel("Похибка R"); axes[1].grid(); axes[1].legend()

# ГРАФІК 3: Порівняння точності методів (Характер зміни похибки) 
methods = ['Центральна', 'Рунге-Ромберг', 'Ейткен']
error_values = [r1, r2, r3]
axes[2].bar(methods, error_values, color=['gray', 'blue', 'green'])
axes[2].set_yscale('log') # Логарифмічна шкала, щоб було видно малі різниці
axes[2].set_title("Порівняння похибок методів (h=0.001)")
axes[2].set_ylabel("Похибка (log scale)"); axes[2].grid(axis='y')

plt.tight_layout()
plt.show()

# Вивід результатів для звіту
print(f"Точне значення M'(1): {exact_val:.6f}")
print(f"Похибка R1 (h=0.001): {r1:.2e}")
print(f"Похибка R2 (Рунге-Ромберг): {r2:.2e}")
print(f"Похибка R3 (Ейткен): {r3:.2e}")