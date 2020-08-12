#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   cvae_gen.py
@Time    :   2020/8/7
@Software:   PyCharm
@Author  :   Li Chen
@Desc    :   
"""

import tensorflow as tf
import os
import logging
import cvae.utils as utils
from cvae.model_bert import CVAEModel_bert
from cvae.data_helper import TFRData
from cvae.gen_tfrecord import gen_tfrecord


def cvae(output_file):
    config = utils.load_config('cvae/config.json')
    data_dir = os.path.join(config['poj_base'], config['data_dir'])
    best_model = os.path.join(config['poj_base'], config['best_model'])
    word_vocab_file = os.path.join(data_dir, config['bert_vocab_file'])
    intent_vocab_file = os.path.join(data_dir, config['intent_vocab_file'])
    ori_files = [os.path.join(data_dir, i) for i in config['ori_file']]
    ori_prep_file = [os.path.join(os.path.dirname(i), os.path.basename(i) + '.tfrecord') for i in ori_files]

    logger = logging.getLogger()
    ckpt_dir = best_model
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    if not ckpt_file:
        logger.error("cannot find checkpoint in {}".format(ckpt_dir))
        exit(1)

    word2id, id2word = utils.read_vocab(word_vocab_file, logger)
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    config = utils.update_vocab_size(config, intent2id, dict())
    gen_tfrecord(word2id, intent2id, ori_files, ori_prep_file, logger, config['max_utter_len'], word_level=True)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    try:
        with tf.Session(config=tf_config) as sess:
            logger.info('loading model from {}'.format(ckpt_file))
            model = CVAEModel_bert(config, utils.GO_ID, utils.EOS_ID)
            model.saver.restore(sess, ckpt_file)
            logger.info('Loading original data')
            ori_dataset = TFRData(utils.PAD_ID, repeat=False)
            ori_dataset.init(sess, ori_prep_file, config['batch_size'])
            data_handler = ori_dataset.get_handler(sess)
            while True:
                try:
                    intents, input_utter, greedy_infer, beam_infer, beam_infer_topk = sess.run(
                        [model.intents, model.utter, model.infer_utter, model.beam_utter, model.beam_utter_topk],
                        feed_dict={model.data_handler: data_handler, model.keep_rate: 1.0})
                    utils.peep_output(intents, id2intent, id2word, input_utter, greedy_infer, beam_infer, beam_infer_topk,
                                      logger, output_file_name=output_file)
                except Exception:
                    break
    except Exception:
        exit(1)
