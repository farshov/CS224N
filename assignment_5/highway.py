#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
import torch
import sys
import numpy as np
"""
CS224N 2018-19: Homework 5
"""
### YOUR CODE HERE for part 1h
EMBED_SIZE = 2
BATCH_SIZE = 3
SENT_LEN = 2
PROB_DROPOUT = 0.5


class Highway(nn.Module):
    def __init__(self, e_word=256, prob=0):
        super(Highway, self).__init__()
        self.e_word = e_word
        self.prob = prob
        self.W_proj = nn.Linear(e_word, e_word, bias=True)
        self.W_gate = nn.Linear(e_word, e_word, bias=True)
        self.dropout = nn.Dropout(prob)

    def forward(self, x_conv_out):
        """
        :param x_conv_out: shape = (max sentence length, batch size, e_word)
        :return: x_high: shape = (max sentence length, batch size, e_word)
        """
        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))
        x_high = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return self.dropout(x_high)

### END YOUR CODE


def highway_shapes_check(model, x_conv_out):
    assert tuple(model.W_proj.weight.shape) == (EMBED_SIZE, EMBED_SIZE)
    assert tuple(model.W_proj.bias.shape) == (EMBED_SIZE, )
    assert tuple(model.W_gate.weight.shape) == (EMBED_SIZE, EMBED_SIZE)
    assert tuple(model.W_gate.bias.shape) == (EMBED_SIZE, )
    res = model(x_conv_out)
    assert res.shape == x_conv_out.shape


def highway_numeric_check(model):
    x = torch.tensor([[[1, 1], [1, 1], [1, 1]],
                      [[2, 2], [2, 2], [2, 2]]])
    x_proj_gold = torch.tensor([[[4, 3], [4, 3], [4, 3]],
                                [[6, 5], [6, 5], [6, 5]]])
    x_gate_gold = torch.tensor([[[5, 6], [5, 6], [5, 6]],
                                [[9, 10], [9, 10], [9, 10]]])
    res_gold = torch.tensor([[[16, 13], [16, 13], [16, 13]],
                            [[38, 32], [38, 32], [38, 32]]])

    model.W_proj.weight = nn.Parameter(torch.tensor([[1, 1],
                                                    [1, 1]]), False)
    model.W_proj.bias = nn.Parameter(torch.tensor([2, 1]), False)
    model.W_gate.weight = nn.Parameter(torch.tensor([[2, 2],
                                                    [2, 2]]), False)
    model.W_gate.bias = nn.Parameter(torch.tensor([1, 2]), False)

    x_proj = model.W_proj(x)
    assert torch.equal(x_proj, x_proj_gold), 'proj'
    x_gate = model.W_gate(x)
    assert torch.equal(x_gate, x_gate_gold), 'gate'

    assert torch.equal(res_gold, x_gate * x_proj + (1 - x_gate) * x), 'res'


def main():
    """ Main func.
    """
    args = sys.argv

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Create NMT Model
    model = Highway(
        e_word=EMBED_SIZE,
        prob=PROB_DROPOUT
    )

    x_conv_out = torch.randn(SENT_LEN, BATCH_SIZE, EMBED_SIZE)
    if args[1] == 'test':
        highway_shapes_check(model, x_conv_out)
        print("SHAPE_CHECK_DONE")
        highway_numeric_check(model)
        print("NUMBERS_CHECK_DONE")
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
