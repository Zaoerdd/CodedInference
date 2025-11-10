import pickle
import threading
import time
import traceback
from socket import create_server, AF_INET
import torch
from cmd_util import *
from comm import accept_connection
from models.AlexNet import AlexNet, AlexNetbig
from models.VGG16 import vgg16
from models.googlenet import GoogLeNet
from models.ResNet import ResNet, ResBlock


# from scipy.optimize import linprog

# def union_interval(intervals):  # 只考虑单个最大区间
#     lefts = [interval[0] for interval in intervals]
#     rights = [interval[1] for interval in intervals]
#     return min(lefts), max(rights)

def get_intersection(interval1, interval2):
    left_max = max(interval1[0], interval2[0])
    right_min = min(interval1[1], interval2[1])
    if left_max < right_min:  # overlap
        return left_max, right_min
    else:
        return None


def get_union(interval1, interval2):
    left_max = max(interval1[0], interval2[0])
    right_min = min(interval1[1], interval2[1])
    if left_max <= right_min:  # overlap or connected
        return min(interval1[0], interval2[0]), max(interval1[1], interval2[1])
    else:
        return None


def get_set_union(intervals):  # 考虑多个区间取并集后仍有多个区间
    intervals.sort(key=lambda range_item: range_item[0])
    i, j = 0, len(intervals) - 1
    while True:
        if i == j:
            break
        range1, range2 = intervals.pop(i), intervals.pop(i)
        union = get_union(range1, range2)
        if union is not None:
            intervals.insert(i, union)
        else:
            intervals.insert(i, range1)
            intervals.insert(i + 1, range2)
            i = i + 1
        j = len(intervals) - 1
    return intervals


def save_object(obj, path: str):
    assert obj is not None
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except:
        traceback.print_exc()


def load_object(path: str):
    obj = None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except:
        traceback.print_exc()
    assert obj is not None
    return obj


def input_satisfactory(required_input: tuple, recv_list: list):  # 判断required_input是否满足
    """
    judge whether the required input of sub-task is satisfied with recv inputs
    :param required_input: a tuple (dependent layers in list, required input range)
    :param recv_list: recv input stored by layers
    :return: return the required concat input else None
    """
    dependent_layers, input_range = required_input
    # print(required_input)
    if len(dependent_layers) > 1:  # concat output from execution units of several layers
        collect = []
        for i, dl in enumerate(dependent_layers):
            outputs = recv_list[dl]
            if len(outputs) == input_range[i]:
                outputs.sort(key=lambda x: x[0][0])
                outputs = [data[1] for data in outputs]
                collect.append(torch.cat(outputs, -1))  # item in recv_list is (global range, data)
            else:
                return None
        return collect

    else:  # collect paddings len(dependent_layers) == 1 or == 0
        c, d = input_range
        if len(dependent_layers) == 0:  # input of the first layer
            if len(recv_list[-1]) == 0:
                return None
            return recv_list[-1][0]
        else:
            input_list = recv_list[dependent_layers[0]]
        input_list.sort(key=lambda item: item[0][0])
        collect = []
        for interval, data in input_list:
            a, b = interval
            if a <= c < b:
                if d <= b:
                    collect.append(data[..., c - a:d - a])
                    c = d
                    break
                else:
                    collect.append(data[..., c - a:])
                c = b
        if c == d:
            # print(torch.concat(collect, dim=-1).shape)
            return torch.concat(collect, dim=-1)
        return None


def input_satisfactory2(required_input: tuple, recv_list: list):  # 判断required_input是否满足
    """
    judge whether the required input of sub-task is satisfied with recv inputs
    :param required_input: a tuple (dependent layers in list, required input range)
    :param recv_list: recv input stored by layers
    :return: return the required concat input else None
    """
    dependent_layers, input_range = required_input
    if len(dependent_layers) == 0:  # input of the first layer
        if len(recv_list[-1]) == 0:
            return None
        return recv_list[-1][0]
    # print(required_input)
    # if len(dependent_layers) > 1:  # concat output from execution units of several layers
    collects = []
    if len(dependent_layers) != len(input_range):
        for last_last in dependent_layers:
            c, d = input_range
            outputs = recv_list[last_last]
            outputs.sort(key=lambda x: x[0][0])
            collect = []
            for interval, data in outputs:
                a, b = interval
                if a <= c < b:
                    if d <= b:
                        collect.append(data[..., c - a:d - a])
                        c = d
                        break
                    else:
                        collect.append(data[..., c - a:])
                    c = b
            if c == d:
                collects.append(collect)
            else:
                return None
        if len(dependent_layers) == 1:
            return torch.cat(collects[0], -1)
        return torch.cat([torch.cat(concat, -1) for concat in collects], 1)

    else:  # concat layer
        for i, dl in enumerate(dependent_layers):
            outputs = recv_list[dl]
            if len(outputs) == input_range[i]:
                outputs.sort(key=lambda x: x[0][0])
                outputs = [data[1] for data in outputs]
                collects.append(torch.cat(outputs, -1))  # item in recv_list is (global range, data)
            else:
                return None
        return collects


def input_satisfactory3(required_input: tuple, recv_list: list):  # split concat layer
    """
    judge whether the required input of sub-task is satisfied with recv inputs
    :param required_input: a tuple (dependent layers in list, required input range)
    :param recv_list: recv input stored by layers
    :return: return the required concat input else None
    """
    dependent_layers, input_range = required_input
    # print(required_input)
    if len(dependent_layers) > 1:  # concat output from execution units of several layers
        collects = []
        for dl in dependent_layers:
            c, d = input_range
            input_list = recv_list[dl]
            input_list.sort(key=lambda item: item[0][0])
            collect = []
            for interval, data in input_list:
                a, b = interval
                if a <= c < b:
                    if d <= b:
                        collect.append(data[..., c - a:d - a])
                        c = d
                        break
                    else:
                        collect.append(data[..., c - a:])
                        c = b
            if c == d:
                collects.append(torch.concat(collect, dim=-1))
            else:
                return None
        return collects

    else:  # collect paddings len(dependent_layers) == 1 or == 0
        c, d = input_range
        if len(dependent_layers) == 0:  # input of the first layer
            if len(recv_list[-1]) == 0:
                return None
            return recv_list[-1][0]
        else:
            input_list = recv_list[dependent_layers[0]]
        input_list.sort(key=lambda item: item[0][0])
        collect = []
        for interval, data in input_list:
            a, b = interval
            if a <= c < b:
                if d <= b:
                    collect.append(data[..., c - a:d - a])
                    c = d
                    break
                else:
                    collect.append(data[..., c - a:])
                c = b
        if c == d:
            # print(torch.concat(collect, dim=-1).shape)
            return torch.concat(collect, dim=-1)
        return None


def recv_connections(server_ip: str, port: str):
    server_socket = create_server((server_ip, port), family=AF_INET)
    # connect_to_other(ip_list, __port__, )
    server_socket.settimeout(8)
    worker_sockets = []
    stop_thread = False
    available_workers = []
    print(f'Master ip is {server_ip}, recv connections from workers...')
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

    return worker_sockets, n_workers


def load_model(model_name: str, dict_path=None, device='cpu'):
    model_name = model_name.lower()
    if model_name == 'googlenet':
        model = GoogLeNet()
    elif model_name == 'vgg16':
        model = vgg16()
    elif model_name == 'alexnet':
        model = AlexNet()
    elif model_name == 'alexnetbig':
        model = AlexNetbig()
    elif model_name == 'resnet':
        model = ResNet(ResBlock)
    else:
        raise Exception('Invalid model name!')
    model.to(device)
    if dict_path is not None:
        model.load_state_dict(torch.load(dict_path, map_location=device))
    return model


def product(num_list):
    res = 1
    for i in num_list:
        res *= i
    return res


def cal_tensor_size(tensor_shape, fix=False):
    size = product(tensor_shape) << 2
    if fix:
        size += 48
    return size

# def solve_linprog_no_transmission(n_device, Yls, H, l):
#     n = n_device
#     n_variables = n + 2
#     n_inequalities = 2 * n
#     # objective function (min)
#     objective = np.zeros(n_variables)
#     objective[-1] = 1
#     # matrix A (inequalities)
#     A = np.zeros((n_inequalities, n_variables))
#     b = np.zeros(n_inequalities)
#     recv_cnt = 0
#     for i in range(1, n+1):
#         A[recv_cnt, i-1] = 1
#         A[recv_cnt, i] = -1
#         recv_cnt += 1
#     for i in range(1, n+1):
#         w_il, b_il = Yls[i-1][l]
#         A[recv_cnt, i-1] = -w_il
#         A[recv_cnt, i] = w_il
#         A[recv_cnt, -1] = -1
#         b[recv_cnt] = -b_il
#         recv_cnt += 1
#     # matrix Aeq (equalities)
#     Aeq = np.zeros((2, n_variables))
#     beq = np.asarray([0, H])
#     #  x0 = 0, xn = H
#     Aeq[0, 0] = 1
#     Aeq[1, n] = 1
#
#     print(objective, A.shape, b.shape, Aeq.shape, beq.shape)
#     result = linprog(objective, A, b, Aeq, beq)
#     if result.get('status') == 0:
#         return result.get('x')
#     else:
#         print(result.get('message'))
#         return None
#
#
# def solve_linprog_edgeflow(n_device, xm, Bij, Tmj, Yls, l, H, layer_params: dict, output_shape: tuple):
#     '''
#     decision variables: x0,x1,...,xn,lambda,p_{i,j}(i,j in {1,...,n})
#     :param n_device: number of devices
#     :param xm: output partition of last layer m
#     :param Bij: bandwidth between devices
#     :param Tmj: the time that device j finished its execution unit of layer m
#     :param Yil: the parameters of linear regression on execution time of layer l in device j
#     :param layer_params: layers parameters (e.g. for conv: kernel_size, stride, padding)
#     :return: The output partition of layer l
#     '''
#     n = n_device
#     n_variables = n*n + n + 2
#     n_inequalities = 5*n*(n-1) + n
#     output_size = cal_tensor_size(output_shape[:-1])
#     # objective function (min)
#     objective = np.zeros(n_variables)
#     objective[n+1] = 1
#     for i in range(1, n+1):
#         for j in range(1, n+1):
#             if i != j:
#                 objective[n + 1 + i*j] = 1
#     # matrix A (inequalities)
#     A = np.zeros((n_inequalities, n_variables))
#     b = np.zeros(n_inequalities)
#     #  x0,...xn
#     recv_cnt = 0
#     for i in range(1, n+1):
#         A[recv_cnt, i-1] = 1
#         A[recv_cnt, i] = -1
#         recv_cnt += 1
#     #  p_{m,i,j}
#     #  in the case of convolutional or pooling layers
#     k, s, p = layer_params['kernel_size'], layer_params['stride'], layer_params['padding']
#     if not isinstance(k, int):
#         k = k[-1]
#     if not isinstance(s, int):
#         s = s[-1]
#     if not isinstance(p, int):
#         p = p[-1]
#     for i in range(1, n+1):
#         for j in range(1, n+1):
#             if i == j:
#                 continue
#             A[recv_cnt, i-1] = s
#             A[recv_cnt, i] = -s
#             A[recv_cnt, n + 1 + i * j] = 1
#             b[recv_cnt] = k - s
#             recv_cnt += 1
#             A[recv_cnt, i] = -s
#             A[recv_cnt, n + 1 + i * j] = 1
#             b[recv_cnt] = -s + k - p - xm[j-1]
#             recv_cnt += 1
#             A[recv_cnt, i - 1] = s
#             A[recv_cnt, n + 1 + i * j] = 1
#             b[recv_cnt] = xm[j] + p
#             recv_cnt += 1
#             A[recv_cnt, n + 1 + i * j] = 1
#             b[recv_cnt] = xm[j] - xm[j-1]
#             recv_cnt += 1
#     #  lambda
#     for i in range(1, n + 1):
#         w_il, b_il = Yls[i-1][l]
#         for j in range(1, n + 1):
#             if i == j:
#                 continue
#             A[recv_cnt, i-1] = -w_il
#             A[recv_cnt, i] = w_il
#             A[recv_cnt, n+1] = -1
#             A[recv_cnt, n + 1 + i * j] = output_size/Bij[i-1, j-1]
#             b[recv_cnt] = -Tmj[j-1] - b_il
#             recv_cnt += 1
#
#     # equality
#     Aeq = np.zeros((2, n_variables))
#     beq = np.asarray([0, H])
#     #  x0 = 0, xn = H
#     Aeq[0, 0] = 1
#     Aeq[1, n] = 1
#
#     # bounds
#     bound_x = [(0, H) for _ in range(n+1)]
#     bound_lambda_p = [(0, None) for _ in range(n*n + 1)]
#     bounds = bound_x + bound_lambda_p
#
#     print(objective.shape, A.shape, b.shape, Aeq.shape, beq.shape, len(bounds))
#     result = linprog(objective, A, b, Aeq, beq, bounds)
#     if result.get('status') == 0:
#         return result.get('x')
#     else:
#         print(result.get('message'))
#         return None
