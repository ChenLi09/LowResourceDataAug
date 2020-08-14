#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   text_mixup.py
@Time    :   2020/8/13
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :   
"""

import torch
from torch.distributions import Beta
import torch.nn.functional as F


def get_lam_beta(alpha, batch_size):
    """
    --return:
        type: torch.Tensor
        value: the value of beta distribution
    """
    m = Beta(torch.FloatTensor([alpha]), torch.FloatTensor([alpha]))
    return m.sample(sample_shape=torch.Size([batch_size]))


def wordMixup(x0, x1, alpha=1.0):
    x_size = x0.size()

    lam = get_lam_beta(alpha, x_size[0])
    lam_re_x = lam.repeat([1, x_size[2]])
    lam_re_x = torch.reshape(lam_re_x, [x_size[0],1,x_size[2]])
    lam_re_x= lam_re_x.repeat([1, x_size[1], 1])

    aug = lam_re_x * x0 + (1-lam_re_x) * x1
    return aug, lam


def sentenceMixup(sen0, sen1, alpha=1.0):
    sen_size = sen0.size()
    lam = get_lam_beta(alpha, sen_size[0])
    lam_re_sen = lam.repeat([1, sen_size[1]])

    aug = lam_re_sen * sen0 + (1-lam_re_sen) * sen1
    return aug, lam


def loss_mixup(logits, target, batch_mix, lam):
    lam = lam.squeeze(1)
    loss = F.cross_entropy(logits[:-batch_mix], target)

    loss_x0 = F.cross_entropy(logits[-batch_mix:], target[:batch_mix], reduce=False)
    loss_x1 = F.cross_entropy(logits[-batch_mix:], target[-batch_mix:], reduce=False)

    total_loss = loss + torch.mean(lam *loss_x0 + (1-lam) * loss_x1)
    return total_loss
