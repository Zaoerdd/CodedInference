import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# 测试指数分布的期望和期望表达式之差
def harmonic(n, k):
    ss = [1/i for i in range(n-k+1, n+1)]
    return sum(ss)

mu1 = 10
scale = 300000

n = 20
ks = range(2, n)
samples = random.exponential(mu1, (scale, n))
# E_k_os = [mu1 * harmonic(n, k) for k in ks]
E_k_os = [mu1 * np.log(n / (n - k)) for k in ks]

generated_E_K_os = []
for k in ks:
    samples_kth = np.partition(samples, k-1, axis=1)[:, k-1]
    mean_kth = samples_kth.mean()
    generated_E_K_os.append(mean_kth)

plt.plot(ks, generated_E_K_os, 'r*--', alpha=0.5, linewidth=1, label='Estimated')
plt.plot(ks, E_k_os, 'g*--', alpha=0.5, linewidth=1, label='Approx')

plt.legend()
plt.xlabel('k')
plt.ylabel('E[T]')

# draw the distribution with instances: what does x-axis means?

plt.show()

