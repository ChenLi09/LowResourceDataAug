#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    :   augment.py
@Time    :   2020-07-15
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :
"""

from utils import eda_gen
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, type=str, help='原始数据的文件路径')
parser.add_argument('--output', required=False, type=str, help='增强数据的输出文件路径')
parser.add_argument('--num_aug', required=False, type=int, default=9, help='每条原始语句增强的语句数')
parser.add_argument('--alpha', required=False, type=float, default=0.1, help='每条语句中将会被改变的单词数占比')
args = parser.parse_args()

num_aug = args.num_aug
alpha = args.alpha
output_path = None
if args.output:
    output_path = args.output
else:
    dir_name = os.path.dirname(args.input)
    file_name = 'eda_' + os.path.basename(args.input)
    output_path = os.path.join(dir_name, file_name)


def augment(original_data, output_file, n_aug, p_change):
    writer = open(output_file, 'w')
    print("正在使用EDA生成增强语句...")
    with open(original_data, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            label = parts[0]
            sentence = parts[1]
            aug_sentences = eda_gen.eda(sentence, p_change, p_change, p_change, p_change, n_aug)
            for aug_sentence in aug_sentences:
                writer.write(label + '\t' + aug_sentence + '\n')
    writer.close()
    print("已生成增强语句!")
    print('存储路径：', output_file)


if __name__ == '__main__':
    augment(args.input, output_path, num_aug, alpha)
