import numpy as np

# Целевая функция
F = lambda x: ((x[0] - 2 * x[1]) ** 2 + (x[1] - 3) ** 2)

# Градиент функции
Gradient = lambda x: np.array([2 * x[0] - 4 * x[1], -4 * x[0] + 10 * x[1] - 6])

# Матрица Гессе
HessianMatrix = lambda x: np.array([(2, -4), (-4, 10)])


# Метод градиентного спуска с дробным шагом
def StepSplittingGradientDescentMethod(x_0, M=1000, e=0.01, t=1):
    k = 0
    x_cur = x_0
    while True:
        x_prev = x_cur
        x_cur = x_cur - t * Gradient(x_cur)
        if F(x_cur) - F(x_prev) < 0:
            if (k + 1) > M or np.linalg.norm(Gradient(x_cur)) < e:
                break
        else:
            t /= 2
        k += 1
    print(f"##### Количество итераций: {k}")
    return x_cur


# Метод Марквардта
def MarquardtMethod(x_prev, M=1000, e=0.01, lmbd=10000):
    k = 0
    while True:
        if np.linalg.norm(Gradient(x_prev)) < e or (k + 1) > M:
            break
        else:
            d = -np.linalg.inv(HessianMatrix(x_prev) + lmbd * np.eye(2, dtype=float)) @ Gradient(x_prev)
            x_next = x_prev + d
            if F(x_next) < F(x_prev):
                lmbd = 0.5 * lmbd
                k += 1
                x_prev = x_next
            else:
                lmbd = 2 * lmbd
    print(f"##### Количество итераций: {k}")
    return x_next


if __name__ == "__main__":
    initial_point = np.array([100, 100])
    print("Метод градиентного спуска с дробным шагом:")
    print(f"##### Точка минимума: {StepSplittingGradientDescentMethod(initial_point)}")
    print()
    print("Метод Марквардта:")
    print(f"##### Точка минимума: {MarquardtMethod(initial_point)}")
