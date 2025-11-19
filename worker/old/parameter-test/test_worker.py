import argparse

import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from util import *
# from functions import

__subnet__ = '192.168.1'
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--master', type=str, required=True)
args = parser.parse_args()
master_ip = args.master
__port__ = 49999

model_name = 'VGG16'
model = load_model('VGG16')
print(f'{model_name} has been loaded.')
B = 1
layers = model.layers

# first connect to master
self_ip = get_ip_addr(__subnet__)
master_socket = None
try:
    master_socket = create_connection(address=(master_ip, __port__))
except Exception as e:
    print('Erroe occurred when connecting to master: \n' + e.__str__())
if master_socket is not None:
    print('Connected to master')

task_cnt = 0
# accept tasks from master and record computation time of each times
while True:
    task_cnt += 1
    print(f'Repetition task{task_cnt}')
    try:
        layer_id, input_shape, repetition = recv_data(master_socket)
        layer = layers[layer_id]
        assert isinstance(layer, BasicConv2d)
        conv = layer.conv

        input_tensor = torch.randn(input_shape)
        this_records = []
        for i in tqdm(range(repetition)):

            consumption = time.time()
            _ = conv(input_tensor)
            consumption = time.time() - consumption
            this_records.append(consumption)

        send_data(master_socket, this_records)
        print(np.asarray(this_records).mean())
    except Exception as e:
        print('Error occurred when receiving or sending: \n' + e.__str__())
        break

# send the records to master
