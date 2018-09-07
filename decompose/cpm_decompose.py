import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tensorly as tl
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
import cpm_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", dest="cp", action="store_true",  help="Use cp decomposition. uses tucker by default")
    parser.set_defaults(cp=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')

    model = torch.load('model')

    model.eval()
    model.cpu()
    N = len(model.module._modules.keys())
    for i, key in enumerate(model.module._modules.keys()):

        if i >= N - 2:
            break
        if isinstance(model.module._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = model.module._modules[key]

            if args.cp:
                rank = max(conv_layer.weight.data.numpy().shape)//3
                decomposed = cp_decomposition_conv_layer(conv_layer, rank)
            else:
                if conv_layer.kernel_size[1] > 1:
                     decomposed = tucker_decomposition_conv_layer(conv_layer)
                else:
                     decomposed = conv_layer

            model.module._modules[key] = decomposed

        torch.save(model, 'decomposed_model')



