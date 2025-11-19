# 热力图：用于查看在特定计算场景下，不同mu_tr和mu_cmp取值时估计与实际在取到最优k*的差距
import sys
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

restart = True

mus = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
labels = ['1', '1e1', '1e2', '1e3', '1e4', '1e5', '1e6', '1e7', '1e8', '1e9']
if restart:
    # computation scenario
    B = 1
    # input shape
    C_i, H_i, W_i = 64, 224, 224
    # output shape
    C_o, H_o, W_o = 64, 224, 224

    kernel_size, stride, padding = 3, 1, 1

    # MDS code
    k = 3
    n = 20
    scale = 100000  # shape (scale, n) for generating random variables, n values per group

    # rate parameter and shift parameter, should be set with proper values
    #  master

    mu_m, theta_m = 5e9, 5e-9
    # #  worker: 4 CPU cores, 1.5 GHz = 1.5 * 10^9 Hz, maximum 10 MB/s
    # #  - mu for computation: floating point operation per second
    # #  - theta for computation: seconds per floating point operation
    # #  - mu for transmission: bytes per second (B/s)
    # #  - theta for computation: seconds per byte (s/B)
    mu_rec = mu_sen = 1e6
    theta_rec = theta_sen = 1e-7
    mu_cmp, theta_cmp = 1e8, 1e-8

    # mu_rec = mu_sen = 1e8
    # theta_rec = theta_sen = 1e-7
    # mu_cmp, theta_cmp = 1e5, 1e-8


    # mus = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    # labels = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1', '1e1', '1e2', '1e3', '1e4', '1e5']

    E_diffs = []
    k_diffs = []

    for mu_tr in mus:
        mu_rec = mu_sen = mu_tr
        for mu_cmp in mus:

            Es_kth_Ts_w = []
            Es_sum_kth = []

            for k in range(2, n + 1):
                # W_p_o = W_o // k
                W_p_o = W_o / k
                W_p_i = kernel_size + (W_p_o - 1) * stride

                N_enc = np.prod([(2 * k - 1), n, B, C_i, H_i, W_p_i])  # in FLOPs (floating point operations)
                N_dec = np.prod([(2 * k - 1), k, B, C_o, H_o, W_p_o])  # in FLOPs
                N_conv = np.prod([B, C_o, H_o, W_p_o, C_i, kernel_size ** 2]) \
                         + np.prod([B, C_o, H_o, W_p_o, C_i * (kernel_size ** 2)])  # in FLOPs
                N_in = np.prod([B, C_i, H_i, W_p_i, 4])  # in bytes
                N_out = np.prod([B, C_o, H_o, W_p_o, 4])  # in bytes

                # print(f'k:{k}, N_enc and N_dec: {N_enc+N_dec}, N_in: {N_in}, N_conv: {N_conv}, N_out: {N_out}')

                # latency of encoding and decoding on master
                T_enc = 2 / mu_m + N_enc * theta_m
                T_dec = 2 / mu_m + N_dec * theta_m
                T_m = T_enc + T_dec
                # print(f'Encoding and decoding on master: {T_m}')

                # update mu in exponential distribution for latency modeling
                _mu_rec = N_in / mu_rec
                _mu_cmp = N_conv / mu_cmp
                _mu_sen = N_out / mu_sen

                # print(_mu_rec, _mu_sen)
                # sys.exit(0)

                # _mu_rec = mu_rec / N_in
                # _mu_cmp = mu_cmp / N_conv
                # _mu_sen = mu_sen / N_out

                # generate random variables following exponential distribution to estimate kth order statistic
                Ts_rec = random.exponential(_mu_rec, (scale, n))
                Ts_cmp = random.exponential(_mu_cmp, (scale, n))
                Ts_sen = random.exponential(_mu_sen, (scale, n))

                Ts_w = Ts_rec + Ts_cmp + Ts_sen
                sum_offset = N_in * theta_rec + N_conv * theta_cmp + N_out * theta_sen

                # kth order statistic of Ts_w, Ts_rec, Ts_cmp, Ts_sen
                kth_Ts_w = np.partition(Ts_w, k - 1, axis=1)[:, k - 1]
                E_kth_Ts_w = np.mean(kth_Ts_w) + sum_offset + T_m
                Es_kth_Ts_w.append(E_kth_Ts_w)

                kth_Ts_rec = np.partition(Ts_rec, k - 1, axis=1)[:, k - 1]
                kth_Ts_cmp = np.partition(Ts_cmp, k - 1, axis=1)[:, k - 1]
                kth_Ts_sen = np.partition(Ts_sen, k - 1, axis=1)[:, k - 1]

                # estimation with
                sum_kth_Ts = kth_Ts_rec + kth_Ts_cmp + kth_Ts_sen
                E_sum_kth = np.mean(sum_kth_Ts) + sum_offset + T_m
                Es_sum_kth.append(E_sum_kth)

            # actual_min = np.min(Es_kth_Ts_w)
            # estimated_argmin = np.argmin(Es_sum_kth)
            # diff = Es_kth_Ts_w[estimated_argmin] - actual_min
            # relative = diff / actual_min
            # print(f'mu_tr:{mu_rec}, mu_cmp:{mu_cmp} ---- {relative}')
            # # print(f'Real k: {np.argmin(Es_kth_Ts_w)}, Estimated k {np.argmin(Es_sum_kth)}')
            # E_diffs.append(relative)

            actual_argmin = np.argmin(Es_kth_Ts_w)
            estimated_argmin = np.argmin(Es_sum_kth)
            k_diff = estimated_argmin - actual_argmin
            print(f'mu_tr:{mu_rec}, mu_cmp:{mu_cmp} ---- {k_diff}')
            k_diffs.append(k_diff)

else:
    file = open('k_diff_with_mus', 'rb')
    k_diffs = pickle.load(file)

# ks = list(range(2, n + 1))
# plt.plot(ks, Es_kth_Ts_w, 'b*--', alpha=0.5, linewidth=1, label='Actual')
# plt.plot(ks, Es_sum_kth, 'r*--', alpha=0.5, linewidth=1, label='Estimated')
# plt.legend()
# plt.xlabel('k')
# plt.ylabel('E[T]')

# draw the distribution with instances: what does x-axis means?
# E_diffs = np.asarray(E_diffs).reshape((len(mus), len(mus)))
k_diffs = np.asarray(k_diffs).reshape((len(mus), len(mus)))[5:, 5:]
with open('k_diff_with_mus_new', 'wb') as f:
    pickle.dump(k_diffs, f)

k_diffs = k_diffs

plt.xticks(np.arange(len(mus))[:5], labels=labels[5:],
                      rotation_mode="anchor", ha="center", size=15)
plt.yticks(np.arange(len(mus))[:5], labels=labels[5:], size=15)

# plt.title("Relative Diff of Estimation with Different Mu")

# plt.imshow(E_diffs)
plt.imshow(k_diffs)
cb=plt.colorbar()

cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
# plt.legend()
plt.ylabel(r'$\mu^{tr}$', rotation=90, size=20)
plt.xlabel(r'$\mu^{cmp}$', size=20)

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.tight_layout()
plt.savefig('Diff_approx_mus.pdf')  # save figure as pdf
plt.show()
