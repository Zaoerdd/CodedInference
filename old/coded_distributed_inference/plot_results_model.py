import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_obj(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# vgg16
total_vgg = 50.60205912590027
local_latency_layers_vgg = [0.319610595703125, 6.18793511390686, 0.013349294662475586, 2.8713788986206055, 5.970198631286621,
                        0.0062410831451416016, 3.2753548622131348, 6.294331312179565, 6.17841100692749,
                        0.003329753875732422, 3.1216278076171875, 6.162682294845581, 6.1657116413116455,
                        0.0019109249114990234, 1.175356149673462, 1.175065279006958, 1.176065444946289,
                        0.0003490447998046875, 0.2157268524169922, 0.00012612342834472656, 0.0003216266632080078,
                        0.03555011749267578, 0.00012946128845214844, 0.0003199577331542969, 0.00879669189453125]
print(total_vgg, sum(local_latency_layers_vgg))
# resnet18
# total_resnet = 106.12221646308899
# local_latency_layers_resnet = []
# print()

layer_indexes = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # vgg16
layer_record_dir1 = 'record/other_layers/vgg/pi/1/'
layer_record_dir2 = 'record/other_layers/vgg/pi/2/'

# t1: coded

layer_record_coded = load_obj('record/vgg16_r30_t1_fail1.rec')
layer_record_repetition = load_obj('record/vgg16_r30_t2_fail1.rec')
layer_record_uncoded = load_obj('vgg16_r30_t3_fail1.rec')

# ks = [5, 8, 9]
# # record_files = [f'vgg16_layer{layer_idx}_r30_t1.rec' for layer_idx in layer_indexes]
# k5_latency = local_latency_layers_vgg.copy()
k_latency_min = local_latency_layers_vgg.copy()
k_latency_mean = local_latency_layers_vgg.copy()
k_latency_max = local_latency_layers_vgg.copy()
# for layer_idx in layer_indexes:
i = 0
for layer_idx, latency in enumerate(local_latency_layers_vgg):
    if layer_idx in layer_indexes:
        layer_latency = layer_record_coded[i]
        i += 1
        min_layer_k, mean_layer_k, max_layer_k = np.min(layer_latency), np.mean(layer_latency), np.max(layer_latency)
        k_latency_min[layer_idx] = min(k_latency_min[layer_idx], min_layer_k)
        k_latency_mean[layer_idx] = min(k_latency_mean[layer_idx], mean_layer_k)
        k_latency_max[layer_idx] = min(k_latency_max[layer_idx], max_layer_k)

        # k5_latency[layer_idx] = min(k5_latency[layer_idx], mean_layer_k5)
        # k_latency_mean[layer_idx] = min(k_latency_mean[layer_idx], mean_layer_k)
#
# k5_latency_sum = np.sum(k5_latency)
k_latency_min_sum = np.sum(k_latency_min)
k_latency_sum = np.sum(k_latency_mean)
k_latency_max_sum = np.sum(k_latency_max)
# print('Coded k=5', k5_latency_sum)
print('Coded k*', k_latency_min_sum, k_latency_sum, k_latency_max_sum)



# t2: repetition
repetition_latency_min = local_latency_layers_vgg.copy()
repetition_latency = local_latency_layers_vgg.copy()
repetition_latency_max = local_latency_layers_vgg.copy()
i = 0
for layer_idx, latency in enumerate(local_latency_layers_vgg):
    if layer_idx in layer_indexes:
        layer_latency = layer_record_repetition[i]
        i+= 1
        layer_min = np.min(layer_latency)
        layer_mean = np.mean(layer_latency)
        layer_max = np.max(layer_latency)
        repetition_latency_min[layer_idx] = min(repetition_latency_min[layer_idx], layer_min)
        repetition_latency[layer_idx] = min(repetition_latency[layer_idx], layer_mean)
        repetition_latency_max[layer_idx] = min(repetition_latency_max[layer_idx], layer_max)

repetition_latency_min_sum = np.sum(repetition_latency_min)
repetition_latency_sum = np.sum(repetition_latency)
repetition_latency_max_sum = np.sum(repetition_latency_max)
print('Repetition', repetition_latency_min_sum, repetition_latency_sum, repetition_latency_max_sum)

# t3: uncoded
uncoded_latency_min = local_latency_layers_vgg.copy()
uncoded_latency = local_latency_layers_vgg.copy()
uncoded_latency_max = local_latency_layers_vgg.copy()

i = 0
for layer_idx, latency in enumerate(local_latency_layers_vgg):
    # print(f'    Layer {layer_idx}')
    if layer_idx in layer_indexes:
        layer_latency = layer_record_uncoded[i]
        i += 1
        layer_min = np.min(layer_latency)
        layer_mean = np.mean(layer_latency)
        layer_max = np.max(layer_latency)
        uncoded_latency_min[layer_idx] = min(uncoded_latency_min[layer_idx], layer_min)
        uncoded_latency[layer_idx] = min(uncoded_latency[layer_idx], layer_mean)
        uncoded_latency_max[layer_idx] = min(uncoded_latency_max[layer_idx], layer_max)
        # uncoded_latency[layer_idx] = min(uncoded_latency[layer_idx], layer_min)

uncoded_latency_min_sum = np.sum(uncoded_latency_min)
uncoded_latency_sum = np.sum(uncoded_latency)
uncoded_latency_max_sum = np.sum(uncoded_latency_max)
print('Uncoded', uncoded_latency_min_sum, uncoded_latency_sum, uncoded_latency_max_sum)

mins = np.asarray([total_vgg-0.3, repetition_latency_min_sum, uncoded_latency_min_sum, k_latency_min_sum])
results = np.asarray([total_vgg, repetition_latency_sum, uncoded_latency_sum, k_latency_sum])
maxs = np.asarray([total_vgg+0.1, repetition_latency_max_sum, uncoded_latency_max_sum, k_latency_max_sum])

low = results - mins
high = maxs - results

labels = ['Local', 'Repetition', 'Uncoded', 'CoCoI']
# plt.bar(labels, results, width=0.5, )

# fig = plt.figure(figsize=(6,4))
fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(zorder=0)
ax.bar(
    x=labels,  # Matplotlib自动将非数值变量转化为x轴坐标
    height=results,  # 柱子高度，y轴坐标
    width=0.5,  # 柱子宽度，默认0.8，两根柱子中心的距离默认为1.0
    align="center",  # 柱子的对齐方式，'center' or 'edge'
    # color="green",  # 柱子颜色
    edgecolor="k",  # 柱子边框的颜色
    linewidth=1.0,  # 柱子边框线的大小
    # hatch='/',
    zorder=5,
)
plt.xticks(size=15)
plt.yticks(size=20)
plt.ylabel('Latency (in seconds)', size=15)
plt.ylim(0, 60)
plt.errorbar(labels, results, fmt='none', yerr=[low, high], ecolor='black', elinewidth=3, capsize=16, capthick=1, zorder=10)
print(results, low, high)

# xticks = ax.get_xticks(total_vgg)
# for i in range(len(results)):
#     xy = (xticks[i], results[i] * 1.020)
#     s = str('{:.3f}'.format(results[i]))
#     ax.annotate(
#         text=s,  # 要添加的文本
#         xy=xy,  # 将文本添加到哪个位置
#         fontsize=13,  # 标签大小
#         color="black",  # 标签颜色
#         ha="center",  # 水平对齐
#         va="baseline"  # 垂直对齐
#     )
plt.tight_layout()
# plt.savefig('vgg_latency.pdf')
plt.show()


