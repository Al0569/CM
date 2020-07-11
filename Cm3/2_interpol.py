import numpy as np
import matplotlib.pyplot as plt
import time

def phi(i, dot):
    n = len(x)

    p = 1
    for j in range(0, n):
        if i != j:
            p = p * (dot - x[j]) / (x[i] - x[j])
    return p


def P(dot):
    n = len(x)

    s = 0
    for i in range(n):
        s = s + y[i] * phi(i, dot)
    return s


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

start = time.time()
for i in range(len(z)):
    f[i] = P(z[i])
stop = time.time()
t = stop - start

for i in range(len(z)):
    f4.write('%f\n' % f[i])



print('\nГрафик по интерполированным данным z, f:')
plt.plot(x, y, 'bo', label='Начальные данные', markersize=6)
plt.plot(z, f, 'ro-', label='Новые данные', markersize=4)
plt.grid(True)
plt.legend()
plt.show()

f1.close()
f2.close()
f3.close()
f4.close()

print('\nВремя вычисления для первого графика:', t)
