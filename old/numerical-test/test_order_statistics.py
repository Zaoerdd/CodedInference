# 单条曲线：用于查看实际order statistic（只是order statistic，不包含master时延），判断实际order statistic的凹凸性，发现并非

import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

B = 1
# input shape
C_i, H_i, W_i = 64, 224, 224
# output shape
C_o, H_o, W_o = 64, 224, 224

kernel_size, stride, padding = 3, 1, 1

# MDS code
k = 3
n = 20
scale = 300000  # shape (scale, n) for generating random variables, n values per group

# rate parameter and shift parameter, should be set with proper values
#  master

mu_1, mu_2, mu_3 = 1, 1, 1
mus = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
labels = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1', '1e1', '1e2', '1e3', '1e4', '1e5']

Es_kth_Ts_w = []
Es_sum_kth = []
E_diffs = []

# for mu_tr in mus:
#     mu_1 = mu_2 = mu_tr
#     for mu_3 in mus:
for k in range(1, n + 1):

    # generate random variables following exponential distribution to estimate kth order statistic
    Ts_rec = random.exponential(1/mu_1, (scale, n))
    Ts_cmp = random.exponential(1/mu_2, (scale, n))
    Ts_sen = random.exponential(1/mu_3, (scale, n))

    Ts_w = Ts_rec + Ts_cmp + Ts_sen

    # kth order statistic of Ts_w, Ts_rec, Ts_cmp, Ts_sen
    kth_Ts_w = np.partition(Ts_w, k - 1, axis=1)[:, k - 1]
    E_kth_Ts_w = np.mean(kth_Ts_w)
    print(f'{n}:{k} order statistic: {E_kth_Ts_w}')
    Es_kth_Ts_w.append(E_kth_Ts_w)

    kth_Ts_rec = np.partition(Ts_rec, k - 1, axis=1)[:, k - 1]
    kth_Ts_cmp = np.partition(Ts_cmp, k - 1, axis=1)[:, k - 1]
    kth_Ts_sen = np.partition(Ts_sen, k - 1, axis=1)[:, k - 1]

    # estimation with
    sum_kth_Ts = kth_Ts_rec + kth_Ts_cmp + kth_Ts_sen
    E_sum_kth = np.mean(sum_kth_Ts)
    Es_sum_kth.append(E_sum_kth)

ks = list(range(1, n + 1))
not_convex = []
for i in range(1, n-1):
    if Es_kth_Ts_w[i-1] + Es_kth_Ts_w[i+1] < 2 * Es_kth_Ts_w[i]:
        not_convex.append(i)

print(not_convex)
plt.plot(ks, Es_kth_Ts_w, 'bv--', alpha=0.5, linewidth=1, label=f'mu1={mu_1},mu2={mu_2},mu3={mu_3}')
plt.plot([ks[i] for i in not_convex], [Es_kth_Ts_w[i] for i in not_convex], '^', color='r')
# plt.plot(ks, Es_sum_kth, 'r*--', alpha=0.5, linewidth=1, label='Estimated')
plt.legend()
plt.xlabel('k')
plt.ylabel('E[T]')

#         actual_min = np.min(Es_kth_Ts_w)
#         estimated_argmin = np.argmin(Es_sum_kth)
#         diff = Es_kth_Ts_w[estimated_argmin] - actual_min
#         relative = diff / actual_min
#         print(f'mu_tr:{mu_tr}, mu_cmp:{mu_3} ---- {relative}')
#         E_diffs.append(relative)
#
# # draw the distribution with instances: what does x-axis means?
# E_diffs = np.asarray(E_diffs).reshape((len(mus), len(mus)))
# plt.xticks(np.arange(len(mus)), labels=labels,
#                      rotation=45, rotation_mode="anchor", ha="right")
# plt.yticks(np.arange(len(mus)), labels=labels)
#
plt.title("Expectation of Order Statistic")
#
# plt.imshow(E_diffs)
# plt.colorbar()
# plt.tight_layout()
# plt.legend()
# plt.ylabel('mu_tr')
# plt.xlabel('mu_cmp')

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
