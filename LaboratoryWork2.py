import numpy as np
import numdifftools as nd

# Целевая функция
F = lambda x: x[0] * x[1] * x[2]

# Ограничения -> l = 2
# Ограничение типа равенства -> m = 1
h = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1  # =0
# Ограничение типа неравенства
g = lambda x: x[0] + x[1] + x[2]  # <=0


# Метод Марквардта
def MarquardtMethod(x_0, func, r, eps=0.01, lmbd=10000):
    '''
    :param x_0: начальное приближение
    :param func: целевая функция
    :param r: дополнительный параметр целевой функции
    :param eps: >0 - требуемая точность решения
    :param lmbd: некоторое большое значение, например, 10^4
    :return: точка минимума
    '''
    k = 0
    x_cur = x_0
    x_next = x_cur
    while True:
        gradient = nd.Gradient(func)(x_cur, r)  # градиент целевой функции
        if np.linalg.norm(gradient) < eps:
            break
        else:
            hessian = nd.Hessian(func)(x_cur, r)  # матрица Гессе целевой функции
            d = -np.linalg.inv(hessian + lmbd * np.eye(3, dtype=float)) @ gradient
            x_next = x_cur + d
            if func(x_next, r) < func(x_cur, r):
                lmbd = 0.5 * lmbd
                k += 1
                x_cur = x_next
            else:
                lmbd = 2 * lmbd
    # print(f"### Количество итераций: {k}")
    return x_next


# Метод внешних штрафов
def ExteriorPenaltyMethod(x_0, n=3, m=1, l=2, eps_1=0.01, eps_2=0.01, r_0=0.1, C=5):
    '''
    :param x_0: начальное приближение, задается вне множества допустимых решений D
    :param n: размерность вектора x
    :param m: число ограничений равенств
    :param l: число всех ограничений
    :param eps_1: точность решения задачи
    :param eps_2: точность решения задачи безусловной минимизации
    :param r_0: >0 - начальное значение параметра штрафа, обычно выбирают r_0 = 0.01;0.1;1
    :param C: >0 - число для увеличения параметра штрафа, обычно выбирают C[4,10]
    :return: точка условного экстремума
    '''
    assert (g(x_0) > 0 and h(x_0) != 0), "Неверное начальное приближение!"

    k = 0
    r = r_0
    x_cur = x_0
    # Вспомогательная функция
    P = lambda x, r: F(x) + Phi(x, r, h, g)

    # Штрафная функция
    def Phi(x, r, h, g):

        # Срезка функции g(x)
        def g_slice(x):
            return g(x) if g(x) > 0 else 0

        phi = r / 2. * ((h(x) ** 2) + (g_slice(x) ** 2))
        return phi

    # print("Итераций метода Марквардта: ")
    while True:
        x_min = MarquardtMethod(x_cur, P, r, eps=eps_2)
        phi = Phi(x_min, r, h, g)
        if np.fabs(phi) <= eps_1:
            break
        else:
            r *= C
            x_cur = x_min
            k = k + 1
    print(f"### Количество итераций метода внешних штрафов: {k}")
    return x_min


# Комбинированный метод штрафных функций
def CombinedPenaltyMethod(x_0, n=3, m=1, l=2, eps_1=0.01, eps_2=0.01, r_0=10, C=5):
    '''
    :param x_0: начальное приближение, задается так, чтобы строго выполнялись ограничения типа неравенств g(x)<0
    :param n: размерность вектора x
    :param m: число ограничений-равенств
    :param l: число всех ограничений
    :param eps_1: точность решения задачи
    :param eps_2: точность решения задачи безусловной оптимизации
    :param r_0: >0 - начальное значение параметра штрафа, обычно выбирают r_0 = 1;10;100
    :param C: >1 - число для увеличения параметра штрафа
    :return: точка условного экстремума
    '''
    assert (g(x_0) <= 0), "Неверное начальное приближение!"

    k = 0
    r = r_0
    x_cur = x_0
    # Вспомогательная функция
    P = lambda x, r: F(x) + Phi(x, r, h, g)

    # Штрафная функция
    def Phi(x, r, h, g):
        phi = 1 / (2. * r) * (h(x) ** 2) - r * 1. / g(x)
        return phi

    # print("Итераций метода Марквардта: ")
    while True:
        x_min = MarquardtMethod(x_cur, P, r, eps=eps_2)
        phi = Phi(x_min, r, h, g)
        if np.fabs(phi) <= eps_1:
            break
        else:
            r /= C
            x_cur = x_min
            k = k + 1
    print(f"### Количество итераций комбинированного метода штрафных функций: {k}")
    return x_min


if __name__ == "__main__":
    eps_1 = 0.01
    eps_2 = 0.01

    print("Метод внешних штрафов:")
    initial_point1 = np.array([10., 10., 10.])
    print(f"### Начальное приближение: {initial_point1}")
    answ1 = ExteriorPenaltyMethod(initial_point1, eps_1=eps_1, eps_2=eps_2)
    print(f"### Точка локального экстремума: {answ1}\n")

    print("Комбинированный метод штрафных функций:")
    initial_point2 = np.array([-10., -10., -10.])
    print(f"### Начальное приближение: {initial_point2}")
    answ2 = CombinedPenaltyMethod(initial_point2, eps_1=eps_1, eps_2=eps_2)
    print(f"### Точка локального экстремума: {answ2}\n")
