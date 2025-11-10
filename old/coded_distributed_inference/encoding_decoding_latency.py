import time
import torch
import torch.nn.functional as F
from util import load_model
import numpy as np
from coding_util import encode_conv_MDS, decode_conv_MDS
from functions import translate_next_array, next_to_last, generate_layerConfig, conv_output_input
import argparse
from models.googlenet import BasicConv2d

def get_input_output_shape(layer_id, input_shape, output_shapes, last_array):
    if layer_id == 0:
        return (1, *input_shape), output_shapes[layer_id]
    last_layer = last_array[layer_id][0]
    return output_shapes[last_layer], output_shapes[layer_id]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='vgg16')

    args = parser.parse_args()
    model_name = args.model

    model = load_model(model_name)

    next_array = model.next
    translate_next_array(next_array)
    last_array = next_to_last(next_array)

    layers = model.layers

    vgg_conv_idxes = []
    for i, layer in enumerate(layers):
        if isinstance(layer, BasicConv2d | torch.nn.Conv2d):
            vgg_conv_idxes.append(i)

    vgg_distributed_conv_idxes = [1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    resnet_conv_idxs = [0, 3, 6, 10, 13, 17, 20, 22, 26, 29, 33, 36, 38, 42, 45, 49, 52, 54, 58, 61]
    resnet_distributed_conv_idxes = [3, 6, 10, 13, 17, 20, 26, 29, 33, 36, 42, 45, 49, 52, 58, 61]

    new_idxes_vgg = []
    new_idxes_resnet = []
    for vgg_distributed_conv_idx in vgg_distributed_conv_idxes:
        idx = vgg_conv_idxes.index(vgg_distributed_conv_idx)
        new_idxes_vgg.append(idx+1)
    for resnet_distributed_conv_idx in resnet_distributed_conv_idxes:
        idx = resnet_conv_idxs.index(resnet_distributed_conv_idx)
        new_idxes_resnet.append(idx+1)

    print(new_idxes_vgg)
    print(new_idxes_resnet)

    if model_name == 'vgg16':
        conv_idxs = vgg_distributed_conv_idxes
    else:
        conv_idxs = resnet_distributed_conv_idxes

    encoding_latency = []
    decoding_latency = []

    n, k = 10, 9
    G = torch.vander(torch.arange(1, n + 1), k).float()

    for conv_idx in conv_idxs:
        layer = layers[conv_idx]
        if isinstance(layer, BasicConv2d):
            layer = layer.conv
        layer_config = generate_layerConfig(layer)
        input_shape, output_shape = get_input_output_shape(conv_idx, model.input_shape, model.output_shapes, last_array)
        x = torch.randn(input_shape)

        padding = layer.padding
        if padding != 0:
            padding = [padding[1] for i in range(2)] + [padding[0] for i in range(2)]
            x = F.pad(x, padding)
            # x_padded = F.pad(x, padding)
            layer_config['padding'] = 0

        O = output_shape[-1]
        worker_task = O // k  # master_task < worker_task
        master_task = O % k

        output_ranges = [(i, i + worker_task) for i in range(0, O - master_task, worker_task)]
        input_ranges = [conv_output_input(output_range, layer_config) for output_range in output_ranges]
        xs = [x[..., s:e] for s, e in input_ranges]  # k*(1,C,I,I/k)

        # testing for encoding
        start = time.time()

        y = encode_conv_MDS(xs, n, k , G)

        consumption = time.time() - start
        encoding_latency.append(consumption)

        ys = [torch.randn(*output_shape[:-1], worker_task) for i in range(k)]

        # testing for decoding
        start = time.time()

        decoded_y = decode_conv_MDS(ys, n, k, G, list(range(k)))

        consumption = time.time() - start
        decoding_latency.append(consumption)

    for i in range(len(conv_idxs)):
        conv_idx = conv_idxs[i]
        t_enc = encoding_latency[i]
        t_dec = decoding_latency[i]
        print(f'layer {conv_idx}, encoding <= {t_enc}, decoding <= {t_dec}', get_input_output_shape(conv_idx, model.input_shape, model.output_shapes, last_array))

    print('Sum:', sum(encoding_latency) + sum(decoding_latency))

# results: resnet18
# layer 3, encoding <= 0.19541239738464355, decoding <= 0.1440725326538086
# layer 6, encoding <= 0.1841726303100586, decoding <= 0.12380719184875488
# layer 10, encoding <= 0.1777801513671875, decoding <= 0.12211871147155762
# layer 13, encoding <= 0.162153959274292, decoding <= 0.13036561012268066
# layer 17, encoding <= 0.17209672927856445, decoding <= 0.06150937080383301
# layer 20, encoding <= 0.09166383743286133, decoding <= 0.06620597839355469
# layer 26, encoding <= 0.09173202514648438, decoding <= 0.062088727951049805
# layer 29, encoding <= 0.09100151062011719, decoding <= 0.06381964683532715
# layer 33, encoding <= 0.09145045280456543, decoding <= 0.027771472930908203
# layer 36, encoding <= 0.05123639106750488, decoding <= 0.027301788330078125
# layer 42, encoding <= 0.052800655364990234, decoding <= 0.02670121192932129
# layer 45, encoding <= 0.05015850067138672, decoding <= 0.028212785720825195
# layer 49, encoding <= 0.04482436180114746, decoding <= 0.013255834579467773
# layer 52, encoding <= 0.032584190368652344, decoding <= 0.013527393341064453
# layer 58, encoding <= 0.033128976821899414, decoding <= 0.013389110565185547
# layer 61, encoding <= 0.03242778778076172, decoding <= 0.013411760330200195
# Sum: 2.4921836853027344

# results: vgg16
# layer 1, encoding <= 0.1939244270324707, decoding <= 0.1387157440185547
# layer 3, encoding <= 0.048238515853881836, decoding <= 0.07011079788208008
# layer 4, encoding <= 0.10098123550415039, decoding <= 0.0594639778137207
# layer 6, encoding <= 0.024212360382080078, decoding <= 0.026593923568725586
# layer 7, encoding <= 0.05275583267211914, decoding <= 0.0278933048248291
# layer 8, encoding <= 0.054300546646118164, decoding <= 0.02733302116394043
# layer 10, encoding <= 0.014374494552612305, decoding <= 0.014009237289428711
# layer 11, encoding <= 0.03171944618225098, decoding <= 0.013364553451538086
# layer 12, encoding <= 0.03244733810424805, decoding <= 0.013276100158691406
# layer 14, encoding <= 0.010057926177978516, decoding <= 0.0017583370208740234
# layer 15, encoding <= 0.009286880493164062, decoding <= 0.0017132759094238281
# layer 16, encoding <= 0.009629011154174805, decoding <= 0.0017194747924804688
# Sum: 0.9778797626495361

