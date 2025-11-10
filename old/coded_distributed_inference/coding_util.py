import torch
import numpy as np

def encode_conv_MDS(xs: list[torch.Tensor], n: int, k: int,
                    G: torch.Tensor):  # generate n encoded inputs from k inputs of same size
    '''
    :param xs: partitioned inputs in k pieces
    :param n: n in MDS code
    :param k: k in MDS code
    :param G: generation matrix, vandermonde matrix
    :param shape: shape of partitioned inputs
    :return: encoded inputs
    '''
    input_shape = xs[0].shape
    split_inputs = torch.concat(xs, dim=0)
    # print(f'split_inputs shape: {split_inputs.shape}')  # k*(1,C,I,I/k) -> (k,C,I,I/k)
    split_inputs = split_inputs.view(k, -1)  # (k,C*I*I/k)
    coded_inputs = torch.matmul(G, split_inputs)  # (n,C*I*I/k)
    coded_inputs = coded_inputs.view(n, *input_shape)  # (n,C,I,I/k)
    coded_inputs = [coded_inputs[i].clone().detach() for i in range(n)]
    return coded_inputs


def decode_conv_MDS(coded_outputs: list[torch.Tensor], n: int, k: int, G: torch.Tensor, kset: list[int]):
    assert len(kset) == k
    output_shape = coded_outputs[0].shape
    Gk = G[kset]
    Ginv = Gk
    # Ginv = torch.linalg.inv(Gk)
    # Ck = torch.concat([coded_outputs[i] for i in kset], dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = torch.concat(coded_outputs, dim=0)  # k*(1,Co,O,O/k) -> (k,Co,O,O/k)
    Ck = Ck.view(k, -1)  # (k,Co,O,O/k) -> (k,Co*O*O/k)
    decoded_outputs = torch.matmul(Ginv, Ck)
    decoded_outputs = decoded_outputs.view(k, *output_shape)
    decoded_outputs = [decoded_outputs[i] for i in range(k)]
    return decoded_outputs


def MDS_conv_optimal_k(perf_params: tuple, task_params: tuple, n: int):
    assert n > 2
    assert len(perf_params) == 5  # mu_m, mu_cmp, theta_cmp, mu_tr, theta_tr
    assert len(task_params) == 8  # C_i, H_i, W_i, C_o, H_o, W_o, kernel, stride
    mu_m, mu_cmp, theta_cmp, mu_tr, theta_tr = perf_params
    C_i, H_i, W_i, C_o, H_o, W_o, kernel, stride = task_params
    a = 2 * (np.prod([n, C_i, H_i, kernel - stride]) + np.prod([C_o, H_o, W_o]))
    b = 4 * (np.prod([C_i, H_i, stride]) + C_o * H_o) * W_o
    c = np.prod([2, C_o, H_o, C_i, kernel, kernel, W_o])
    d = np.prod([n, C_i, H_i, W_o, stride])
    A = a / mu_m
    B = b * theta_tr + c * theta_cmp - d / mu_m
    C = b / mu_tr + c / mu_cmp
    L_k = [A*k + B/k + C/k * np.log(n/(n-k)) for k in range(2, n)]
    return np.argmax(L_k)
