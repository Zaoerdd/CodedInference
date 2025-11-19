import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_obj(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def show_one():
    record_file_t1 = 'record/vgg16_layer1_r100_t2_pi_fail1_4.rec'
    records = load_obj(record_file_t1)
    print(records)
    # record_file_t2 = 'record/vgg16_layer1_r100_t2_pi_fail1_2.rec'
    # records = load_obj(record_file_t2)
    # print(records)
    # record_file_t3 = 'record/vgg16_layer1_r100_t2_pi_fail1_3.rec'
    # records = load_obj(record_file_t3)
    # print(records)
    print(np.mean(records))
    # for i in records:
    #     print(np.mean(i))

def show_all():
    record_dir = 'record/'
    # WIFI pi
    # record_file_t1 = [f'record/vgg16_layer1_r100_t1_{i}.rec' for i in range(1, 3)]
    # record_file_t2 = [f'record/vgg16_layer1_r100_t2_{i}.rec' for i in range(1, 3)]
    # record_file_t3 = [f'record/vgg16_layer1_r100_t3_{i}.rec' for i in range(1, 3)]
    # Wired PC
    record_file_t1 = [f'record/vgg16_layer1_r100_t1_pc_{i}.rec' for i in range(1, 5)]
    record_file_t2 = [f'record/vgg16_layer1_r100_t2_pc_{i}.rec' for i in range(1, 4)]
    record_file_t3 = [f'record/vgg16_layer1_r100_t3_pc_{i}.rec' for i in range(1, 4)]
    # Wired Pi
    # record_file_t1 = [f'record/vgg16_layer1_r100_t1_pi_{i}.rec' for i in range(1, 4)]
    # record_file_t2 = [f'record/vgg16_layer1_r100_t2_pi_{i}.rec' for i in range(1, 2)]
    # record_file_t3 = [f'record/vgg16_layer1_r100_t3_pi_{i}.rec' for i in range(1, 2)]

    # record_file_t1 = [f'record/vgg16_layer1_r100_t1_pi_fail1.rec'] # for i in range(1, 4)]
    # record_file_t2 = [f'record/vgg16_layer1_r100_t2_pi_fail1_{i}.rec' for i in range(1, 5)]
    # record_file_t3 = [f'record/vgg16_layer1_r100_t3_pi_fail1.rec'] # for i in range(1, 2)]


    record_t1 = []
    record_t2 = []
    record_t3 = []

    for rec_file in record_file_t1:
        with open(rec_file, 'rb') as f:
            record_t1.append(pickle.load(f))

    for rec_file in record_file_t2:
        with open(rec_file, 'rb') as f:
            record_t2.append(pickle.load(f))

    for rec_file in record_file_t3:
        with open(rec_file, 'rb') as f:
            record_t3.append(pickle.load(f))

    # print(np.asarray(record_t1).shape, np.asarray(record_t2).shape, np.asarray(record_t3).shape)
    # records of t1
    for cnt, rec_t1 in enumerate(record_t1):
        print(f'Record{cnt+1}')
        # for k in range(5, 10):
        for i, k in enumerate([5, 8, 9]):
            # i = k - 5
            record = rec_t1[i]
            print(f'Coded: n={10}, k={k}')
            print(np.mean(record))

    print()

    # records of t2
    for cnt, rec_t2 in enumerate(record_t2):
        print(f'Uncoded: n=10, k=5, record{cnt+1}, expectation: {np.mean(rec_t2)}')

    print()

    # records of t3
    for cnt, rec_t3 in enumerate(record_t3):
        # plt.boxplot()
        print(f'Uncoded: n=10, record{cnt+1}, expectation: {np.mean(rec_t3)}')

    return record_t1[0], record_t2[-1], record_t3[0]

def show_fail2():
    record_dir = 'record/'

    record_file_t1 = [f'record/vgg16_layer1_r100_t1_fail2.rec'] # for i in range(1, 4)]
    record_file_t2 = [f'record/vgg16_layer1_r100_t2_fail2.rec'] # for i in range(1, 5)]
    record_file_t3 = [f'record/vgg16_layer1_r100_t3_pi_fail2.rec'] # for i in range(1, 2)]


    record_t1 = []
    record_t2 = []
    record_t3 = []

    for rec_file in record_file_t1:
        with open(rec_file, 'rb') as f:
            record_t1.append(pickle.load(f))

    for rec_file in record_file_t2:
        with open(rec_file, 'rb') as f:
            record_t2.append(pickle.load(f))

    for rec_file in record_file_t3:
        with open(rec_file, 'rb') as f:
            record_t3.append(pickle.load(f))

    # print(np.asarray(record_t1).shape, np.asarray(record_t2).shape, np.asarray(record_t3).shape)
    # records of t1
    for cnt, rec_t1 in enumerate(record_t1):
        print(f'Record{cnt+1}')
        # for k in range(5, 10):
        for i, k in enumerate([5, 8]):
            # i = k - 5
            record = rec_t1[i]
            print(f'Coded: n={10}, k={k}')
            print(np.mean(record))

    print()

    # records of t2
    for cnt, rec_t2 in enumerate(record_t2):
        print(f'Uncoded: n=10, k=5, record{cnt+1}, expectation: {np.mean(rec_t2)}')

    print()

    # records of t3
    for cnt, rec_t3 in enumerate(record_t3):
        # plt.boxplot()
        print(f'Uncoded: n=10, record{cnt+1}, expectation: {np.mean(rec_t3)}')

    return record_t1[0], record_t2[0], record_t3[0]


def box_plot_latency():
    record_dir = 'D:/华为云盘/project/DistributedInference/coded_distributed_inference/record/'
    coded_k5 = record_dir + 'vgg16_layer1_r100_t1_pc_3.rec'
    coded_k = record_dir + 'vgg16_layer1_r100_t1_pc_3.rec'
    repetition = record_dir + 'vgg16_layer1_r100_t2_pc_3.rec'
    uncoded = record_dir + 'vgg16_layer1_r100_t3_pc_3.rec'


    file_list = [repetition, coded_k5, uncoded, coded_k]

    # data_list = [[6.18793511390686]]
    data_list = []

    data = load_obj(repetition)
    data_list.append(data)

    # data = load_obj(coded_k5)
    # data_list.append(data[0])

    data = load_obj(uncoded)
    data_list.append(data)

    data = load_obj(coded_k)
    data_list.append(data[-1])


    # print(data_list)
    plt.figure(figsize=(3, 4))
    plt.grid(zorder=-1)
    # plt.boxplot(data_list, labels=['Repetition', 'Coded k=5', 'Uncoded', 'Coded k*'], showmeans=True)
    plt.boxplot(data_list, labels=['Repetition', 'Uncoded', 'CoCoI'], showmeans=True)
    plt.xticks(rotation=30, size=12)
    plt.yticks(size=12)
    plt.ylabel('Latency (in seconds)', size=12)
    plt.tight_layout()
    plt.savefig('vgg16_layer1_fail0.pdf')
    plt.show()
    return [np.mean(item) for item in data_list]

def box_plot_fail1():
    coded = load_obj('vgg16_layer1_r100_t1_fail1.rec')
    repetition = load_obj('vgg16_layer1_r100_t2_fail1.rec')
    uncoded = load_obj('vgg16_layer1_r100_t3_fail1.rec')

    coded_k5 = coded[0]
    coded_k = coded[-1]
    # data_list = [repetition, coded_k5, uncoded, coded_k]
    data_list = [repetition, uncoded, coded_k]

    plt.figure(figsize=(3, 4))
    plt.grid(zorder=-1)
    plt.xticks(rotation=30, size=12)
    plt.yticks(size=12)
    # plt.boxplot(data_list, labels=['Repetition', 'Coded k=5', 'Uncoded', 'Coded k*'], showmeans=True)
    plt.boxplot(data_list, labels=['Repetition', 'Uncoded', 'CoCoI'], showmeans=True)
    plt.tight_layout()
    plt.savefig('vgg16_layer1_fail1.pdf')
    plt.show()
    return [np.mean(item) for item in data_list]

def box_plot_fail2():
    coded = load_obj('vgg16_layer1_r100_t1_fail2.rec')
    repetition = load_obj('vgg16_layer1_r100_t2_fail2.rec')
    uncoded = load_obj('record/vgg16_layer1_r100_t3_fail2.rec')

    coded_k5 = coded[0]
    coded_k = coded[-1]
    # data_list = [repetition, coded_k5, uncoded, coded_k]
    data_list = [repetition, uncoded, coded_k]
    plt.figure(figsize=(3, 4))
    plt.grid(zorder=-1)
    plt.xticks(rotation=30, size=12)
    plt.yticks(size=12)
    # plt.boxplot(data_list, labels=['Repetition', 'Coded k=5', 'Uncoded', 'Coded k*'], showmeans=True)
    plt.boxplot(data_list, labels=['Repetition', 'Uncoded', 'CoCoI'], showmeans=True)
    plt.tight_layout()
    plt.savefig('vgg16_layer1_fail2.pdf')
    plt.show()
    return [np.mean(item) for item in data_list]


if __name__ == '__main__':
    # show_all()
    # show_one()
    es0 = box_plot_latency()
    es1 = box_plot_fail1()
    es2 = box_plot_fail2()

    # means = [es0, es1, es2]
    # print(means)
    # xs = ['Fail=0', 'Fail=1', 'Fail=2']
    # methods = ['Repetition', 'Coded k=5', 'Uncoded', 'Coded k*']
    #
    # ess = [[es0[i], es1[i], es2[i]] for i in range(4)]
    # plt.figure(figsize=(3,4))
    # for i, es in enumerate(ess):
    #     plt.plot(xs, es, label=methods[i])
    # plt.ylabel('Latency Expectation')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('layer_means.pdf')
    # plt.show()
