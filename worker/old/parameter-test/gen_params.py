import numpy as np
import pickle


def load_records(file_name):
    records_dir = {}
    with open(file_name, 'r') as file_to_read:
        line = file_to_read.readline()
        while line:
            params = line.split(' ')
            layer_id = int(params[0])
            N_convs = [int(x) for x in params[1:]]
            layer_records = [N_convs]
            for N_conv in N_convs:
                rs = file_to_read.readline().strip()
                rs = rs[1:-1].split(', ')
                rs = [float(x) for x in rs]
                # k_records = [float(x) for x in file_to_read.readline()[1:-1].split(', ')]
                layer_records.append(rs)

            records_dir[layer_id] = layer_records
            line = file_to_read.readline()
    return records_dir


'''
(mu^, theta^) = (1/(x_avg-x_min), x_min)
'''


def MLE_shifted_exponential_distribution(x_set):
    x_avg = np.mean(x_set)
    x_min = np.min(x_set)
    return 1 / (x_avg - x_min), x_min


def mu_variance_estimation(x_set):
    sigma = np.std(x_set)
    x_avg = np.mean(x_set)
    return 1 / sigma, x_avg - sigma


def adjust_latency(latency_records: np.ndarray, the_other_records: np.ndarray):
    """
    correct the negative latency to positive records
    :param latency_records: with shape (types of transmission sizes, device num, repetitions)
    :param the_other_records: corresponding latency where sum of these two latency is the RTT
    :return: the corrected latency
    """
    assert len(latency_records.shape) == len(the_other_records.shape) == 3
    if latency_records.min() > 0:
        return latency_records, the_other_records
    transmission_types, n_device, repetition = latency_records.shape
    for i in range(transmission_types):
        type_recs = latency_records[i]
        if type_recs.min() < 0:
            type_mins = type_recs.min(axis=-1)
            negatives, positives = 0, []
            for value in type_mins:
                if value > 0:
                    positives.append(value)
                else:
                    negatives += 1
                positives.sort()
            if negatives > 0:
                for device in range(n_device):
                    type_device_recs = type_recs[device]
                    if type_device_recs.min() < 0:
                        min_positive = min(positives)
                        print(f'negative {type_device_recs.min()} positive {min_positive}')
                        random_offset = np.random.uniform(min_positive*0.95, min_positive*1.1) - type_device_recs.min()
                        type_device_recs += random_offset
                        the_other_records[i, device] -= random_offset
    return latency_records, the_other_records


def gen_cmp_parameters():
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

def gen_tr_parameters():
    record_file = open('transmission100_device4.rec', 'rb')
    # record_file = open('transmission100_device10_110.rec', 'rb')
    # record_file2 = open('transmission100_device10_114.rec', 'rb')
    transmission_records = pickle.load(record_file)
    # transmission_records = pickle.load(record_file) + pickle.load(record_file2)

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


if __name__ == '__main__':
    gen_cmp_parameters()
    gen_tr_parameters()
