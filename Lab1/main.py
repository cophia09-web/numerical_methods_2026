import requests
import numpy as np
import matplotlib.pyplot as plt

# --- 1. МАТЕМАТИЧНА МОДЕЛЬ (Клас сплайну) ---
class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x) - 1
        self.h = np.diff(self.x)
        
        # Метод прогонки для пошуку коефіцієнтів c
        self.c = self._solve_c()
        
        # Розрахунок a, b, d за формулами з методички
        self.a = self.y[:-1]
        self.b = (np.diff(self.y) / self.h) - (self.h / 3.0) * (self.c[1:] + 2 * self.c[:-1])
        self.d = (self.c[1:] - self.c[:-1]) / (3.0 * self.h)

    def _solve_c(self):
        n = self.n
        alpha = np.zeros(n); beta = np.zeros(n); gamma = np.zeros(n); delta = np.zeros(n)
        for i in range(1, n):
            alpha[i] = self.h[i-1]
            beta[i] = 2 * (self.h[i-1] + self.h[i])
            gamma[i] = self.h[i]
            delta[i] = 3 * ((self.y[i+1]-self.y[i])/self.h[i] - (self.y[i]-self.y[i-1])/self.h[i-1])

        A = np.zeros(n); B = np.zeros(n)
        for i in range(1, n):
            denom = beta[i] - alpha[i] * A[i-1]
            A[i] = gamma[i] / denom
            B[i] = (delta[i] - alpha[i] * B[i-1]) / denom
            
        c = np.zeros(n + 1)
        for i in range(n-1, -1, -1):
            c[i] = B[i] - A[i] * c[i+1]
        return c

    def evaluate(self, x_new):
        i = np.searchsorted(self.x, x_new) - 1
        i = max(0, min(i, self.n - 1))
        dx = x_new - self.x[i]
        return self.a[i] + self.b[i]*dx + self.c[i]*(dx**2) + self.d[i]*(dx**3)

# --- 2. ОТРИМАННЯ ДАНИХ (Hoverla Route) ---
print("Отримання даних...")
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
res = requests.get(url).json()["results"]

# Відстані (Haversine)
def haversine(p1, p2):
    R = 6371000
    lat1, lon1, lat2, lon2 = np.radians([p1['latitude'], p1['longitude'], p2['latitude'], p2['longitude']])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

elevations = [p['elevation'] for p in res]
distances = [0]
for i in range(1, len(res)):
    distances.append(distances[-1] + haversine(res[i-1], res[i]))

# --- 3. ОБЧИСЛЕННЯ ХАРАКТЕРИСТИК) ---
spline = CubicSpline(distances, elevations)
xx = np.linspace(distances[0], distances[-1], 500)
yy = [spline.evaluate(x) for x in xx]

print(f"Загальна довжина: {distances[-1]:.2f} м")
total_ascent = sum(max(elevations[i]-elevations[i-1], 0) for i in range(1, len(elevations)))
print(f"Сумарний підйом: {total_ascent:.2f} м")

# Механічна енергія для маси 80 кг
mass, g = 80, 9.81
energy = mass * g * total_ascent
print(f"Механічна робота: {energy/1000:.2f} кДж ({energy/4184:.2f} ккал)")

# --- 4. ГРАФІК ---
plt.figure(figsize=(10, 5))
plt.scatter(distances, elevations, color='red', label='GPS Точки')
plt.plot(xx, yy, label='Кубічний сплайн', color='blue')
plt.title("Профіль висоти маршруту на Говерлу")
plt.xlabel("Відстань (м)"); plt.ylabel("Висота (м)")
plt.legend(); plt.grid(True)
plt.show()