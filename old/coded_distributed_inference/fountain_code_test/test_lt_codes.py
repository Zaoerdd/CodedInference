from tqdm import tqdm
import itertools
from coded_distributed_inference.fountain_code_test.LTCodes import *


def test_encoding_decoding(split_cnt):  # done
    assert isinstance(split_cnt, int) and split_cnt > 0
    x = torch.randn(1, 3, split_cnt)
    xs = [xi for xi in x.split(1, dim=-1)]
    coded_computation = LTCodedTask(xs)

    for i in range(coded_computation.max_encoded_num):
        encoded_symbol = next(coded_computation.gen_encoded_symbol)
        cur_rank = coded_computation.put_recv(i, encoded_symbol.data)
        if cur_rank == coded_computation.source_num:
            break
    decoded_data = coded_computation.decode_ge()
    decoded_data = torch.concat(decoded_data, dim=-1)
    # print(torch.allclose(x, decoded_data, atol=1e-2))
    return len(coded_computation.recv_symbols), coded_computation.source_num

def test_decoding_success_cnt():
    repeat = 100
    decoded_cnts = []
    source_cnt = 14
    for i in tqdm(range(repeat)):
        random.seed(i)
        decoded_symbol_cnt, _ = test_encoding_decoding(source_cnt)
        # print(decoded_cnts)
        decoded_cnts.append(decoded_symbol_cnt)
    print(decoded_symbol_cnt, decoded_cnts.count(decoded_symbol_cnt))


def test_decoding_success_rate(split_cnt, encoded_cnt):
    assert isinstance(split_cnt, int) and split_cnt > 0
    x = torch.randn(1, 3, split_cnt)
    xs = [xi for xi in x.split(1, dim=-1)]
    coded_computation = LTCodedTask(xs)

    print('Generating symbols...')
    for i in tqdm(range(coded_computation.max_encoded_num)):
        encoded_symbol = next(coded_computation.gen_encoded_symbol)
        coded_computation.put_recv(i, encoded_symbol.data)
    symbol_idxs = range(len(coded_computation.recv_symbols))

    print('Generating combinations...')
    combinations = itertools.combinations(symbol_idxs, encoded_cnt)
    combinations_cnt = 0.0
    # print('All combinations:', combinations_cnt)

    available_cnt = 0.0
    print('Testing combinations...')
    if split_cnt == 9:
        a = 1

    for idxs in tqdm(combinations):
        combinations_cnt += 1
        encoding_matrix = coded_computation.encoding_matrix[idxs, :]
        if np.linalg.matrix_rank(encoding_matrix) == split_cnt:
            available_cnt += 1
    success_rate = available_cnt/combinations_cnt
    print(f'Decoding success rate: {success_rate: .5f}')
    return success_rate

def test_k_decoding_success_rate():
    x_range = range(2, 10)
    n_reach_99 = []
    for k in x_range:
        for n in range(k, 13):
            print(f'k={k}, n={n}')
            success_rate = test_decoding_success_rate(k, n)
            if success_rate > 0.99:
                n_reach_99.append((k, n))
                break
    print(n_reach_99)



if __name__ == '__main__':
    test_decoding_success_rate(224, 225)
