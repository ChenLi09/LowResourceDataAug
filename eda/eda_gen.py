#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    :   eda_gen.py
@Time    :   2020-07-15
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :
"""

import jieba
import synonyms
import random

random.seed(2020)

# 停用词列表，默认使用哈工大停用词表
stop_words = []
with open('data/stopwords/hit_stopwords.txt') as file:
    for line in file:
        line = line.strip()
        stop_words.append(line)


########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def get_synonyms(word):
    return synonyms.nearby(word)[0]


def synonyms_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms_list = get_synonyms(random_word)
        if synonyms_list:
            random_synonym = random.choice(synonyms_list)
            new_words = [random_synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def add_word(new_words):
    synonyms_list = []
    count = 0
    while len(synonyms_list) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms_list = get_synonyms(random_word)
        count += 1
        if count >= 5:
            return new_words
    random_synonym = random.choice(synonyms_list)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = add_word(new_words)
    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    count = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        count += 1
        if count > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if not new_words:
        random_int = random.randint(0, len(words)-1)
        new_words.append(words[random_int])
    return new_words


########################################################################
# main data augmentation function
########################################################################
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = ' '.join(seg_list)
    words = seg_list.split()
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonyms_replacement(words, n_sr)
        augmented_sentences.append(''.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    random.shuffle(augmented_sentences)
    if num_aug >= 1 and type(num_aug) == int:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        assert False, 'should give a right num_aug'
    augmented_sentences.append(seg_list)
    return augmented_sentences


# print(eda(sentence="我们就像蒲公英，我也祈祷着能和你飞去同一片土地"))
