#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
import torch
import sys
import numpy as np
### YOUR CODE HERE for part 1i
EMBED_WORD = 256
EMBED_CHAR = 50
BATCH_SIZE = 3
SENT_LEN = 4
MAX_SENT_LEN = 23
MAX_WORD_LEN = 21


class CNN(nn.Module):
    def __init__(self, e_word=256, e_char=50, kernel=5):
        super(CNN, self).__init__()
        self.k = kernel
        self.e_char = e_char
        self.e_word = e_word
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=kernel)

    def forward(self, x_reshaped):
        """
        :param x_reshaped: shape = (max_sentence_length, batch_size, e_char, max_word_length)
        :return: x_conv_out: shape = (max sentence length, batch size, e_word)
        """
        max_sentence_length, batch_size, e_char, max_word_length = x_reshaped.shape
        x_conv = self.conv(x_reshaped.view(-1, e_char, max_word_length)).view(max_sentence_length, batch_size,
                                                                                   self.e_word, -1)
        x_conv = F.relu(x_conv)
        # shape = (max_sentence_length, batch_size, e_word, max_word_length - kernel + 1)
        x_conv_out = torch.max(x_conv, dim=-1)[0]
        x_conv_out = x_conv_out.view(max_sentence_length, batch_size, self.e_word)
        return x_conv_out

### END YOUR CODE


def cnn_shapes_check(model, x_reshaped):
    x_conv_out = model(x_reshaped)
    x_gold = torch.randn(MAX_SENT_LEN, BATCH_SIZE, EMBED_WORD)
    assert x_conv_out.shape == x_gold.shape


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
    model = CNN()

    x_reshaped = torch.randn(MAX_SENT_LEN, BATCH_SIZE, EMBED_CHAR, MAX_WORD_LEN)
    if args[1] == 'test':
        cnn_shapes_check(model, x_reshaped)
        print("SHAPE_CHECK_DONE")
        # highway_numeric_check(model)
        # print("NUMBERS_CHECK_DONE")
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()

