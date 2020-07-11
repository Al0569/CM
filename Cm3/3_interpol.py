import numpy as np
import matplotlib.pyplot as plt
import time

def sweep (a, b, c, f):
    n = f.size

    alpha = [0] * (n+1)
    betta = [0] * (n+1)
    x = [0] * (n+1)
    a[0] = 0
    c[n-1] = 0
    alpha[0] = 0
    betta[0] = 0

    for i in range(n):
        d = a[i] * alpha[i] + b[i]
        alpha[i+1] = -c[i] / d
        betta[i+1] = (f[i] - a[i] * betta[i]) / d

    x[n] = 0
    for i in range(n-1, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + betta[i+1]

    x = x[:-1]
    return x



def gencof(x, y):
    n = x.shape[0] - 1
    h = (x[n] - x[0]) / n

    a = np.array([0] + [1] * (n - 1) + [0])
    b = np.array([1] + [4] * (n - 1) + [1])
    c = np.array([0] + [1] * (n - 1) + [0])
    f = np.zeros(n + 1)

    for i in range(1, n):
        f[i] = 3 * (y[i-1] - 2 * y[i] + y[i+1]) / (h**2)
    s = sweep(a, b, c, f)

    A = np.zeros(n+1)
    B = np.zeros(n+1)
    C = np.zeros(n+1)
    D = np.zeros(n+1)

    for i in range(n):
        B[i] = s[i]

    for i in range(n):
        A[i] = (B[i+1] - B[i]) / (3 * h)
        C[i] = (y[i+1] - y[i]) / h - (B[i+1] + 2 * B[i]) * h / 3
        D[i] = y[i]

    return A, B, C, D



def inDot(dot, x, y, A, B, C, D):
    n = len(x)

    if dot == x[n - 1]:
        return y[n-1]

    if dot < x[0] or dot > x[n - 1]:
        print('Число %f выходит за область' % dot)
        return

    for i in range(1, n):
        if dot < x[i]:
            left = i - 1
            break

    P = A[left] * ((dot - x[left]) ** 3)
    P += B[left] * ((dot - x[left]) ** 2)
    P += C[left] * (dot - x[left])
    P += D[left]

    return P


x = np.array([2,3,4,5])
y = np.array([1,4,2,5])
z = np.linspace(np.min(x), np.max(x), 1000)
f = np.zeros(len(z))


f1 = open('train.dat')
f2 = open('train.ans')
f3 = open('test.dat')
f4 = open('test.ans', 'w')

x = [float(i) for i in f1.read().split()]
y = [float(i) for i in f2.read().split()]
z = [float(i) for i in f3.read().split()]

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

z = np.array(z, dtype=float)
f = np.zeros(len(z))


A, B, C, D = gencof(x, y)
print('A:', A)
print('B:', B)
print('C:', C)
print('D:', D)


start = time.time()
for i in range(len(z)):
    f[i] = inDot(z[i], x, y, A, B, C, D)
stop = time.time()




print('\nГрафик по интерполированным данным z, f:')
plt.plot(x, y, 'bo', label='Начальные данные', markersize=6)
plt.plot(z, f, 'ro-', label='Новые данные', markersize=2)
plt.grid(True)
plt.legend()
plt.show()


print('\nВремя:', stop - start)


f1.close()
f2.close()
f3.close()
f4.close()