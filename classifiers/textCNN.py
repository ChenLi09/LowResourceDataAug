#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   textCNN.py
@Time    :   2020-07-15
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mixup import text_mixup as tm


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        class_num = args.class_num
        channel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding = self.embedding.from_pretrained(args.vectors, freeze=True)

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes)*filter_num, class_num)

    def forward(self, x, is_training=False):
        x = self.embedding(x)
        lam = None
        if is_training and self.args.mixup == 'word':
            batch_mix_index = int(x.size()[0] * self.args.rate_mixup)
            x_mix, lam = tm.wordMixup(x[:batch_mix_index, :, :], x[-batch_mix_index:, :, :], self.args.alpha)
            x = torch.cat((x, x_mix), 0)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        if is_training and self.args.mixup == 'sen':
            batch_mix_index = int(x.size()[0] * self.args.rate_mixup)
            x_mix, lam = tm.sentenceMixup(x[:batch_mix_index, :], x[-batch_mix_index:, :], self.args.alpha)
            x = torch.cat((x, x_mix), 0)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits, lam
