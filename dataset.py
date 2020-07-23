#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   dataset.py
@Time    :   2020/7/17
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :   
"""

import re
from torchtext import data
import jieba

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def tokenizer(text):
    text = regex.sub(' ', text)
    result = [token for token in jieba.cut(text) if token.strip()]
    return result


def make_dataset(train_path, val_path, text_field, label_field):
    text_field.tokenize = tokenizer
    train = data.TabularDataset(
        path=train_path, format='csv', skip_header=True,
        fields=[('label', label_field), ('text', text_field)]
    )
    val = data.TabularDataset(
        path=val_path, format='csv', skip_header=True,
        fields=[('label', label_field), ('text', text_field)]
    )
    return train, val
