import numpy as np
import matplotlib.pyplot as plt
import time


def interpol(dot, x, y):
    n = len(x)

    if dot == x[n - 1]:
        return y[n-1]

    if (dot < x[0] or dot > x[n - 1]):
        print('Выход за границы')
        return

    for i in range(n-1):
        if x[i] <= dot:
            left = i
        if x[i+1] > dot:
            right = i+1
            break

    dot_y = (y[right] - y[left]) / (x[right] - x[left])
    dot_y *= (dot - x[left])
    dot_y += y[left]

    return dot_y



f1 = open('train.dat')
f2 = open('train.ans')
f3 = open('test.dat')
f4 = open('test.ans', 'w')

x = [float(i) for i in f1.read().split()]
y = [float(i) for i in f2.read().split()]
z = [float(i) for i in f3.read().split()]

x = np.array(x)
y = np.array(y)
z = np.array(z)
f = np.zeros(len(z))

start = time.time()
for i in range(len(z)):
    f[i] = interpol(z[i], x, y)
stop = time.time()
t = stop - start
for i in range(len(f)):
    f4.write('%f\n' % f[i])

print('\nГрафик по интерполированным данным z, f:')
plt.plot(x, y, 'bo-', label='Начальные данные', markersize=6)
plt.plot(z, f, 'ro-', label='Новые данные', markersize=2)
plt.grid(True)
plt.legend()
plt.show()

f1.close()
f2.close()
f3.close()
f4.close()
print('Время вычисления:', t)
