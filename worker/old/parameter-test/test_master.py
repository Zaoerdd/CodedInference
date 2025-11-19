import asyncio
import threading
import sys
from util import *
from functions import output_input
import pickle
# from functions import *

record_file_name = 'N_time_records_vgg16.txt'

''' records format:
layer_id1, N_conv1,..., N_convn
[list of records corresponds to N_conv1]
...
[list of records corresponds to N_convn]
layer_id2, N_conv1,..., N_convn
[list of records corresponds to N_conv1]
...
[list of records corresponds to N_convn]
......
'''


def save_records(records_dir: dict, file_name):
    with open(file_name, 'w') as file_to_write:
        for layer_id in records_dir.keys():
            N_record = records_dir[layer_id]
            Ns = N_record[0]
            records = N_record[1:]
            file_to_write.write(f'{layer_id} ' + ' '.join([str(N) for N in Ns]) + '\n')
            for record in records:
                file_to_write.write(f'{record}\n')


model = load_model('VGG16')
B = 1
layers = model.layers
repetition = 100
# connect to workers
# build connections
self_ip = '192.168.1.155'
__port__ = 49999
server_socket = create_server((self_ip, __port__), family=AF_INET)
# connect_to_other(ip_list, __port__, )
server_socket.settimeout(8)
worker_sockets = []
stop_thread = False
available_workers = []
print(f'Master ip is {self_ip}, recv connections from workers...')
recv_thread = threading.Thread(target=accept_connection,
                               args=[server_socket, available_workers, lambda: stop_thread])
recv_thread.start()
time.sleep(10)  # waiting for 10s
stop_thread = True
recv_thread.join()

num_ip = []
for conn, addr in available_workers:
    # conn.settimeout(5)
    # ip, port = addr
    if addr not in num_ip:  # first time to meet the addr
        num_ip.append(addr)
        worker_sockets.append(conn)
    else:
        conn.close()
n_workers = len(num_ip)
print(f'Recv connections from {n_workers} workers')

loop = asyncio.get_event_loop()  # for async sending
records = {}
layers_id = []

for i, layer in enumerate(layers):

    if isinstance(layer, BasicConv2d):
        layers_id.append(i)
        print(f'Start sending task of layer{i}')
        conv = layer.conv
        input_shape = model.get_feature_shape(i, True)
        output_shape = model.get_feature_shape(i, False)
        # input shape
        B, C_i, H_i, W_i = input_shape
        # output shape
        _, C_o, H_o, W_o = output_shape
        # conv
        kernel_size, stride = conv.kernel_size[-1], conv.stride[-1]

        N_convs = []
        records_layer = [N_convs]

        ks = list(range(2, 10))
        for k in ks:
            W_p_o = W_o // k
            W_p_i = kernel_size + (W_p_o - 1) * stride
            if k > 1:
                input_shape = (*input_shape[:-1], W_p_i)
            N_conv = W_p_o * C_o * H_o * (2 * C_i * kernel_size * kernel_size - 1)
            N_convs.append(N_conv)
            # send layer_id and input_shape to worker and repetition times to workers
            config = (i, input_shape, repetition)
            print(config)
            send_tasks = [async_send_data(conn, config) for conn in worker_sockets]
            loop.run_until_complete(asyncio.gather(*send_tasks))
            layer_k_record = []
            recv_queue = SimpleQueue()  # receive the records from workers
            for conn in worker_sockets:
                recv_thread = threading.Thread(target=put_recv_data, args=[conn, recv_queue])
                recv_thread.start()
                recv_thread.join()
            while not recv_queue.empty():
                layer_k_record += (recv_queue.get())

            records_layer.append(layer_k_record)
            time.sleep(10)  # 每轮计算让树莓派休息60s别炸了
        records[i] = records_layer
        # print(f'Layer{i}: {record_this_layer}')

for conn in worker_sockets:
    conn.close()
# after receive all records from workers, disconnect to the workers and calculate the parameters
# records_dir = {}
# for record in records:
#     N_conv = record[0]
#     if N_conv not in records_dir.keys():
#         records_dir[N_conv] = []
#     for rec in record[1:]:
#         records_dir[N_conv] += rec

with open(f'Computation{repetition}_device{n_workers}.rec', 'wb') as record_file:
    pickle.dump(records, record_file)
# save_records(records, record_file_name)
