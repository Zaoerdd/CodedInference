import numpy as np
import pickle
from gen_params import load_records, MLE_shifted_exponential_distribution, mu_variance_estimation
from gen_params import adjust_latency


# test the estimation results from the derived system parameters
cmp_parameters = False
tr_parameters = True


def test_cmp_parameters():
    record_file_name = 'N_time_records_vgg16.txt'
    layers_records = load_records(record_file_name)
    layer_ids = layers_records.keys()

    all_latency = []

    layers_params = {}
    for layer_id in layer_ids:
        layer_record = layers_records[layer_id]
        N_convs = layer_record[0]
        ks_records = layer_record[1:]
        layer_t_Nconv = [np.asarray(ks_records[i]) / N_conv for i, N_conv in enumerate(N_convs)]
        all_latency.extend(layer_t_Nconv)
        layers_params[layer_id] = MLE_shifted_exponential_distribution(layer_t_Nconv)

    print(layers_params)
    all_latency = np.concatenate(all_latency, axis=0)
    params_overall = MLE_shifted_exponential_distribution(all_latency)
    print('ML Estimation:')
    print(f'Computation params: {params_overall}')
    params_overall = mu_variance_estimation(all_latency)
    print('Direct Estimation:')
    print(f'Computation params: {params_overall}')

    # test for error
    mu_cmp, theta_cmp = params_overall
    for layer_id in layer_ids:
        layer_record = layers_records[layer_id]
        N_convs = layer_record[0]
        layer_k_records = layer_record[1:]
        for Nk in N_convs:
            mean_estimated = Nk / mu_cmp + Nk * theta_cmp  # unit: second
# 看起来总体还行


def test_tr_parameters():
    # record_file = open('transmission100_device4_2.rec', 'rb')
    record_file = open('transmission100_device10_110.rec', 'rb')
    record_file2 = open('transmission100_device10_114.rec', 'rb')
    transmission_records = pickle.load(record_file) + pickle.load(record_file2)

    n = len(transmission_records)
    Ns_tr = [rs[0] for rs in transmission_records]
    recs = [rs[1] for rs in transmission_records]
    send_recs = np.asarray([rs[0] for rs in recs])
    recv_recs = np.asarray([rs[1] for rs in recs])
    rtt_recs = np.asarray([rs[2] for rs in recs])

    send_recs, recv_recs = adjust_latency(send_recs, recv_recs)
    recv_recs, send_recs = adjust_latency(recv_recs, send_recs)

    # 对send和recv分别表示参数
    send_recs_norm = [send_recs[i] / Ns_tr[i] for i in range(n)]
    recv_recs_norm = [recv_recs[i] / Ns_tr[i] for i in range(n)]
    all_send_r = np.concatenate(send_recs_norm)
    all_recv_r = np.concatenate(recv_recs_norm)
    params_send = MLE_shifted_exponential_distribution(all_send_r)
    params_recv = MLE_shifted_exponential_distribution(all_recv_r)
    print('MLE Estimation:')
    print(f'Send params: {params_send}')
    print(f'Recv params: {params_recv}')

    params_send = mu_variance_estimation(all_send_r)
    params_recv = mu_variance_estimation(all_recv_r)
    print('Direct Estimation:')
    print(f'Send params: {params_send}')
    print(f'Recv params: {params_recv}')

    mu_sen, theta_sen = params_send
    mu_rec, theta_rec = params_recv
    for i in range(len(Ns_tr)):
        N_tr = Ns_tr[i]
        sen_recs = recs[i][0]
        rec_recs = recs[i][1]
        sen_mean = np.mean(sen_recs)
        rec_mean = np.mean(rec_recs)
        estimated_mean_sen = N_tr / mu_sen + N_tr * theta_sen
        estimated_mean_rec = N_tr / mu_rec + N_tr * theta_rec


if __name__ == '__main__':
    # test_cmp_parameters()
    test_tr_parameters()