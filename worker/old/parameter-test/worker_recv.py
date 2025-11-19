import argparse
from socket import create_connection
from comm import send_data, recv_data
import numpy as np
import time
from tqdm import tqdm

__port__ = 49999

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--master', type=str, required=True)
args = parser.parse_args()
master_ip = args.master

# connect
master_socket = create_connection((master_ip, __port__))
print('Connected to master.')

try:
    while True:
        recv = recv_data(master_socket)
        print(recv, end=' ')
        feature_size, k, repetition = recv
        recv_time = []
        send_time = []
        data_size = feature_size // k
        for _ in tqdm(range(repetition)):
            data = recv_data(master_socket)
            end = time.time()
            recv_time.append(end)
            assert len(data) == data_size
            # time.sleep(0.2)
            start = time.time()
            send_data(master_socket, data)
            send_time.append(start)

        send_data(master_socket, (recv_time, send_time))

except Exception as e:
    print(e)



