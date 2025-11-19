# 多条曲线：用于查看在特定计算场景下，mu和theta的变化对实际和估计的最优k*的影响

import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

# computation scenario
B = 1
# input shape
C_i, H_i, W_i = 64, 224, 224
# output shape
C_o, H_o, W_o = 64, 224, 224

kernel_size, stride, padding = 3, 1, 1

scale = 50000  # shape (scale, n) for generating random variables, n values per group

# rate parameter and shift parameter, should be set with proper values
#  master

mu_m, theta_m = 5e9, 5e-9
# #  worker: 4 CPU cores, 1.5 GHz = 1.5 * 10^9 Hz, maximum 10 MB/s
# #  - mu for computation: floating point operation per second
# #  - theta for computation: seconds per floating point operation
# #  - mu for transmission: bytes per second (B/s)
# #  - theta for computation: seconds per byte (s/B)

# tr parameters
# mus_tr = [1e5, 1e8, 1e7]
# thetas_tr = [1e-5, 1e-8, 1e-9]
mus_tr = [1e5, 1e6, 1e7]
thetas_tr = [1e-5, 1e-7, 1e-9]
# mu_theta_tr_pairs = [(mus_tr[0], thetas_tr[1]), (mus_tr[2], thetas_tr[1]), (mus_tr[1], thetas_tr[1]), (mus_tr[1], thetas_tr[0]), (mus_tr[1], thetas_tr[2])]
# mu_theta_tr_pairs_str = [('10^5', '10^{-8}'), ('10^7', '10^{-8}'), ('10^6', '10^{-8}'), ('10^6', '10^{-6}'), ('10^6', '10^{-10}')]
mu_theta_tr_pairs = [(mus_tr[0], thetas_tr[1]), (mus_tr[1], thetas_tr[0]), (mus_tr[1], thetas_tr[1]), (mus_tr[1], thetas_tr[2]), (mus_tr[2], thetas_tr[1])]
mu_theta_tr_pairs_str = [('10^5', '10^{-8}'), ('10^6', '10^{-6}'), ('10^6', '10^{-8}'), ('10^6', '10^{-10}'), ('10^7', '10^{-8}')]

# mu_theta_tr_pairs = mu_theta_tr_pairs
# mu_theta_tr_pairs_str = mu_theta_tr_pairs_str

# cmp parameters
mus_cmp = [1e7, 1e8, 1e9]
thetas_cmp = [1e-7, 1e-8, 1e-9]
# mu_theta_cmp_pairs = [(mus_cmp[0], thetas_cmp[1]), (mus_cmp[2], thetas_cmp[1]), (mus_cmp[1], thetas_cmp[1]), (mus_cmp[1], thetas_cmp[0]), (mus_cmp[1], thetas_cmp[2])]
# mu_theta_cmp_pairs_str = [('10^7', '10^{-8}'), ('10^9', '10^{-8}'), ('10^8', '10^{-8}'), ('10^8', '10^{-7}'), ('10^8', '10^{-9}')]
mu_theta_cmp_pairs = [(mus_cmp[0], thetas_cmp[1]), (mus_cmp[1], thetas_cmp[0]), (mus_cmp[1], thetas_cmp[1]), (mus_cmp[1], thetas_cmp[2]), (mus_cmp[2], thetas_cmp[1])]
mu_theta_cmp_pairs_str = [('10^7', '10^{-8}'), ('10^8', '10^{-7}'), ('10^8', '10^{-8}'), ('10^8', '10^{-9}'), ('10^9', '10^{-8}')]

mu_tr = mus_tr[1]
theta_tr = thetas_tr[1]
mu_cmp = 1e9
theta_cmp = 1e-8

# theta_rec = theta_sen = 1e-8
# theta_cmp = 5e-8

generate_data = False
actual = True

if generate_data:
    actual_curves_of_optimal_k = []
    estimated_curves_of_optimal_k = []
else:
    with open('actual_curves_of_optimal_k', 'rb') as record:
        actual_curves_of_optimal_k = pickle.load(record)
    with open('estimated_curves_of_optimal_k', 'rb') as record:
        estimated_curves_of_optimal_k = pickle.load(record)
approx_curves_of_optimal_k = []


# for mu_tr, theta_tr in mu_theta_tr_pairs:

for mu_tr, theta_tr in mu_theta_tr_pairs:
    print(f'mu_cmp={mu_cmp}, theta_cmp={theta_cmp}')

    mu_rec = mu_sen = mu_tr
    theta_rec = theta_sen = theta_tr
    # print(f'mu_tr={mu_tr}, theta_tr={theta_tr}')

    optimal_k_2_n_actual = []
    optimal_k_2_n_estimated = []
    optimal_k_2_n_approx = []

    # (n,k)-MDS code
    for n in range(10, 21):
        print(f' n={n}', end='')

        k0 = n // 2

        a = 2 * (1 / mu_m + theta_m) * (n * C_i * H_i * (kernel_size - stride) + C_o * H_o * W_o)
        b = 4 * W_o * (C_i * H_i * stride * theta_rec + C_o * H_o * theta_sen) + np.prod(
            [2, C_o, H_o, W_o, C_i, kernel_size ** 2, theta_cmp])
        c = 4 * W_o * (C_i * H_i * stride / mu_rec + C_o * H_o / mu_sen) + np.prod(
            [2, C_o, H_o, W_o, C_i, kernel_size ** 2, 1 / mu_cmp])
        d = 4 * C_i * H_i * (kernel_size - stride) / mu_rec

        approx = [a * k + b / k + c / k * np.log(n / (n - k)) + d * np.log(n / (n - k)) for k in
                    range(k0, n)]
        optimal_k_approx = k0 + np.argmin(approx)
        optimal_k_2_n_approx.append(optimal_k_approx)

        if generate_data:
            Es_kth_Ts_w = []
            Es_sum_kth = []

            for k in range(k0, n):
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

            optimal_k_actual = k0 + np.argmin(Es_kth_Ts_w)
            optimal_k_estimated = k0 + np.argmin(Es_sum_kth)
            optimal_k_2_n_actual.append(optimal_k_actual)
            optimal_k_2_n_estimated.append(optimal_k_estimated)
            # k_diff = estimated_argmin - actual_argmin
            print(f', actual k* {optimal_k_actual}, estimated k* {optimal_k_estimated}')

    if generate_data:
        actual_curves_of_optimal_k.append(optimal_k_2_n_actual)
        estimated_curves_of_optimal_k.append(optimal_k_2_n_estimated)
    approx_curves_of_optimal_k.append(optimal_k_2_n_approx)

with open('actual_curves_of_optimal_k', 'wb') as record:
    pickle.dump(actual_curves_of_optimal_k, record)

with open('estimated_curves_of_optimal_k', 'wb') as record:
    pickle.dump(estimated_curves_of_optimal_k, record)

ns = range(10, 21)

marks = ['+--', 'o--', 'o--', 'o--', 'x--']
markersize = [14,14,10,7,12]

plt.figure()
# actual tr
for i, mu_theta_tr_pair in enumerate(mu_theta_tr_pairs_str):
    mu_tr, theta_tr = mu_theta_tr_pair
    actual_curve_of_optimal_k = actual_curves_of_optimal_k[i]
    ms = markersize[i]
    if i in [1, 2, 3]:
        plt.plot(ns, actual_curve_of_optimal_k, marks[i], linewidth=1.5,
                 label=r'$\mu$,$\theta=$' + f'${mu_tr},{theta_tr}$', markerfacecolor='white', markersize=ms, zorder=0)
    else:
        plt.plot(ns, actual_curve_of_optimal_k, marks[i], linewidth=1.5,
                 label=r'$\mu$,$\theta=$' + f'${mu_tr},{theta_tr}$', markersize=ms)
plt.rcParams.update({'font.size': 15})
plt.legend(ncol=2)
plt.grid(zorder=-1)
# plt.title('Actual')
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylabel('$k^*$', size=20)
plt.xlabel('$n$', size=20)
plt.ylim(4, 30)
plt.tight_layout()
plt.savefig('actual_k_change_tr.pdf')
plt.show()


# estimated tr
for i, mu_theta_tr_pair in enumerate(mu_theta_tr_pairs_str):
    mu_tr, theta_tr = mu_theta_tr_pair
    estimated_curve_of_optimal_k = estimated_curves_of_optimal_k[i]
    ms = markersize[i]
    if i in [1, 2, 3]:
        plt.plot(ns, estimated_curve_of_optimal_k, marks[i], linewidth=1.5,
                 label=r'$\mu$,$\theta=$' + f'${mu_tr},{theta_tr}$', markerfacecolor='white', markersize=ms,
                 zorder=0)
    else:
        plt.plot(ns, estimated_curve_of_optimal_k, marks[i], linewidth=1.5,
                 label=r'$\mu$,$\theta=$' + f'${mu_tr},{theta_tr}$', markersize=ms)
plt.rcParams.update({'font.size': 15})
plt.legend(ncol=2)
plt.grid(zorder=-1)
# plt.title('Approx')
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylabel('$k^*$', size=20)
plt.xlabel('$n$', size=20)
plt.ylim(4, 30)
plt.tight_layout()
plt.savefig('approx_k_change_tr.pdf')
plt.show()


# actual tr
# for i, mu_theta_cmp_pair in enumerate(mu_theta_cmp_pairs_str):
#     mu_cmp, theta_cmp = mu_theta_cmp_pair
#     actual_curve_of_optimal_k = actual_curves_of_optimal_k[i]
#     ms = markersize[i]
#     if i in [1, 3]:
#         plt.plot(ns, actual_curve_of_optimal_k, marks[i], alpha=0.9, linewidth=1,
#                  label=r'$\mu$,$\theta=$' + f'${mu_cmp},{theta_cmp}$', markerfacecolor='white', markersize=ms, zorder=0)
#     else:
#         plt.plot(ns, actual_curve_of_optimal_k, marks[i], alpha=0.9, linewidth=1,
#              label=r'$\mu$,$\theta=$'+f'${mu_cmp},{theta_cmp}$', markersize=ms)
# plt.rcParams.update({'font.size': 15})
# plt.legend(ncol = 2)
# plt.grid(zorder=-1)
# # plt.title('Actual')
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.ylabel('$k^*$', size=20)
# plt.xlabel('$n$', size=20)
# plt.ylim(4, 30)
# plt.tight_layout()
# plt.savefig('actual_k_change_cmp.pdf')
# plt.show()
#
# # estimated tr
# for i, mu_theta_cmp_pair in enumerate(mu_theta_cmp_pairs_str):
#     mu_cmp, theta_cmp = mu_theta_cmp_pair
#     estimated_curve_of_optimal_k = estimated_curves_of_optimal_k[i]
#     ms = markersize[i]
#     if i in [1, 3]:
#         plt.plot(ns, estimated_curve_of_optimal_k, marks[i], linewidth=1,
#                  label=r'$\mu$,$\theta=$' + f'${mu_cmp},{theta_cmp}$', markerfacecolor='white',
#                  markersize=ms, zorder=0)
#     else:
#         plt.plot(ns, estimated_curve_of_optimal_k, marks[i], linewidth=1,
#                  label=r'$\mu$,$\theta=$' + f'${mu_cmp},{theta_cmp}$', markersize=ms)
# plt.rcParams.update({'font.size': 15})
# plt.legend(ncol=2)
# plt.grid(zorder=-1)
# # plt.title('Approx')
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.ylabel('$k^*$', size=20)
# plt.xlabel('$n$', size=20)
# plt.ylim(4, 30)
# plt.tight_layout()
# plt.savefig('approx_k_change_cmp.pdf')
# plt.show()





# actual cmp
# for i, mu_theta_cmp_pair in enumerate(mu_theta_cmp_pairs_str):
#     mu_cmp, theta_cmp = mu_theta_cmp_pair
#     actual_curve_of_optimal_k = actual_curves_of_optimal_k[i]
#     plt.plot(ns, actual_curve_of_optimal_k, '^--', alpha=0.5, linewidth=1,
#              label=r'$\mu^{tr}$,$\theta^{tr}=$' + f'${mu_tr},{mu_cmp}$')


# ks = list(range(2, n + 1))
# plt.plot(ks, Es_kth_Ts_w, 'b*--', alpha=0.5, linewidth=1, label='Actual')
# plt.plot(ks, Es_sum_kth, 'r*--', alpha=0.5, linewidth=1, label='Estimated')



# draw the distribution with instances: what does x-axis means?
# E_diffs = np.asarray(E_diffs).reshape((len(mus), len(mus)))
# k_diffs = np.asarray(k_diffs).reshape((len(mus), len(mus)))
# with open('k_diff_with_mus', 'wb') as f:
#     pickle.dump(k_diffs, f)

# plt.xticks(np.arange(len(mus)), labels=labels,
#                      rotation=45, rotation_mode="anchor", ha="right", size=10)
# plt.yticks(np.arange(len(mus)), labels=labels, size=10)

# plt.title("Relative Diff of Estimation with Different Mu")

# plt.imshow(E_diffs)
# plt.imshow(k_diffs)
# plt.colorbar()
# plt.tight_layout()
# # plt.legend()
# plt.ylabel(r'$\mu^{tr}$', rotation=90, size=12)
# plt.xlabel(r'$\mu^{cmp}$', size=12)

# plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.tight_layout()
# plt.savefig('Diff_approx_mus.pdf')  # save figure as pdf
# plt.show()
