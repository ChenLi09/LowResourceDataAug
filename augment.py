#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    :   augment.py
@Time    :   2020-07-15
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :
"""

from eda import eda_gen
from bt import bt_gen
from cvae import cvae_gen
import argparse
import os
import csv
import random

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, type=str, help='增强方法选择')
parser.add_argument('--input_file', required=True, type=str, help='原始数据的文件路径')
parser.add_argument('--output', required=True, type=str, help='增强数据的输出路径')
parser.add_argument('--num_aug', required=False, type=int, default=9, help='每条原始语句增强的语句数')
parser.add_argument('--alpha', required=False, type=float, default=0.1, help='每条语句中将会被改变的单词数占比')
args = parser.parse_args()


file_name = args.method + '_' + os.path.basename(args.input_file)
output_file = os.path.join(args.output, file_name)


def augment(method, original_data, o_file, n_aug, p_change):
    print("正在使用{}生成增强语句...".format(method))
    if method == 'cvae':
        with open(original_data, 'r', encoding='utf-8') as input_f:
            reader = csv.reader(input_f)
            with open('cvae/tmp_data/all_ori', 'w', encoding='utf-8') as output_f:
                for item in reader:
                    if reader.line_num == 1:
                        continue
                    label = item[0]
                    sentence = item[1]
                    sen_sep = ' '.join([c for c in sentence])
                    output_f.write('\t'.join(['root', label, sen_sep]) + '\n')
        cvae_gen.cvae(o_file)
    else:
        result = []
        with open(original_data, 'r') as file:
            reader = csv.reader(file)
            for item in reader:
                if reader.line_num == 1:
                    continue
                label = item[0]
                sentence = item[1]
                if method == 'eda':
                    aug_sentences = eda_gen.eda(sentence, p_change, p_change, p_change, p_change, n_aug)
                elif method == 'bt':
                    aug_sentences = bt_gen.back_translate(sentence)
                for aug_sentence in aug_sentences:
                    result.append([label, aug_sentence])
        random.shuffle(result)
        result = [['label', 'text']] + result
        with open(o_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for item in result:
                writer.writerow(item)
    print("已生成增强语句!")
    print('存储路径：', o_file)


if __name__ == '__main__':
    augment(args.method, args.input_file, output_file, args.num_aug, args.alpha)
