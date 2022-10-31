import numpy as np
import numdifftools as nd

# Целевая функция
F = lambda x: ((x[0] - 2 * x[1]) ** 2 + (x[1] - 3) ** 2)


# Метод градиентного спуска с дробным шагом
def StepSplittingGradientDescentMethod(x_0, M=1000, eps=0.01, t=1):
    '''
    :param x_0: начальное приближение
    :param M: допустимое число итераций
    :param eps: >0 - требуемая точность решения
    :param t: длина шага
    :return: точка минимума
    '''
    k = 0
    x_cur = x_0
    while True:
        x_prev = x_cur
        gradient = nd.Gradient(F)(x_cur)  # градиент целевой функции
        x_cur = x_cur - t * gradient
        if F(x_cur) - F(x_prev) < 0:
            if (k + 1) > M or np.linalg.norm(gradient) < eps:
                break
        else:
            t /= 2
        k += 1
    print(f"##### Количество итераций: {k}")
    return x_cur


# Метод Марквардта
def MarquardtMethod(x_0, M=1000, eps=0.01, lmbd=10000):
    '''
    :param x_0: начальное приближение
    :param M: допустимое число итераций
    :param eps: >0 - требуемая точность решения
    :param lmbd: некоторое большое значение, например, 10^4
    :return: точка минимума
    '''
    k = 0
    x_cur = x_0
    while True:
        gradient = nd.Gradient(F)(x_cur)  # градиент целевой функции
        if np.linalg.norm(gradient) < eps or (k + 1) > M:
            break
        else:
            hessian = nd.Hessian(F)(x_cur)  # матрица Гессе целевой функции
            d = -np.linalg.inv(hessian + lmbd * np.eye(2, dtype=float)) @ gradient
            x_next = x_cur + d
            if F(x_next) < F(x_cur):
                lmbd = 0.5 * lmbd
                k += 1
                x_cur = x_next
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
