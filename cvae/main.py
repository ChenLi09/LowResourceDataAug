import tensorflow as tf
import cvae.utils as utils
import argparse
from cvae.model_bert import CVAEModel_bert
import os
import time
import traceback
import numpy as np
from cvae.data_helper import TFRData
from cvae.gen_tfrecord import gen_tfrecord

logger = utils.get_logger('main.log')
parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file', default='cvae/config.json')
parser.add_argument('--gpu', help='which gpu to use', default='1')
parser.add_argument("--is_train", type=utils.str2bool, default=True, help="is_train or infer&evaluate")
parser.add_argument("--use_best", type=utils.str2bool, default=True, help="whether to use best model")
parser.add_argument("--csv_file_name", help='output csv file name', default='bert_result.csv')
parser.add_argument("--use_bert", type=utils.str2bool, default=True, help="whether to use bert model")

args = parser.parse_args()

config = utils.load_config(args.config)

train_dir = os.path.join(config['poj_base'], config['train_dir'])
data_dir = os.path.join(config['poj_base'], config['data_dir'])
eval_dir = os.path.join(config['poj_base'], config['eval_dir'])
log_dir = os.path.join(config['poj_base'], config['log_dir'])
best_model = os.path.join(config['poj_base'], config['best_model'])

word_vocab_file = os.path.join(data_dir, config['bert_vocab_file'])
intent_vocab_file = os.path.join(data_dir, config['intent_vocab_file'])

train_files = [os.path.join(data_dir, i) for i in config['train_file']]
valid_files = [os.path.join(data_dir, i) for i in config['valid_file']]
test_files = [os.path.join(data_dir, i) for i in config['test_file']]

csv_file = os.path.join(data_dir, config['csv_file'])

train_prep_file = [os.path.join(os.path.dirname(i), os.path.basename(i) + '.tfrecord') for i in train_files]
valid_prep_file = [os.path.join(os.path.dirname(i), os.path.basename(i) + '.tfrecord') for i in valid_files]
test_prep_file = [os.path.join(os.path.dirname(i), os.path.basename(i) + '.tfrecord') for i in test_files]


def add_summary(writer, step, data):
    for i in data:
        name, value = i
        writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)]), global_step=step)


np.random.seed(config['seed'])

try:

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    dirs = [train_dir, eval_dir, log_dir, best_model]
    for d in dirs:
        if not os.path.isdir(d):
            logger.info('cannot find {}, mkdiring'.format(d))
            os.makedirs(d)

    if args.is_train:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = True
        with tf.Session(config=tf_config) as sess:
            check_list = {'word_vocab': word_vocab_file, 'intent_vocab': intent_vocab_file}
            print("-----------------------------")
            if tf.train.get_checkpoint_state(train_dir):
                # load trained model
                print("--------load trained model---------")
                logger.info('Found check point in {}'.format(train_dir))
                for file in check_list:
                    if not os.path.isfile(check_list[file]):
                        logger.error('Can not find {}'.format(check_list[file]))
                        exit(1)
                logger.info('Loading vocab files {}'.format(check_list))
                word2id, id2word = utils.read_vocab(check_list['word_vocab'], logger, limit=config['word_vocab_size'])
                intent2id, id2intent = utils.read_vocab(check_list['intent_vocab'], logger)

                config = utils.update_vocab_size(config, intent2id, dict())
                logger.info('Loading checkpoint from {}'.format(tf.train.latest_checkpoint(train_dir)))
                model = CVAEModel_bert(config, utils.GO_ID, utils.EOS_ID)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                # Begin from scratch
                print('--------------Starting from scratch-------------')
                miss_files = [file for file in check_list.values() if not os.path.isfile(file)]
                logger.info(miss_files)
                if len(miss_files) != 0:
                    logger.info('{} files '.format(len(miss_files)) +
                                'are missing: \n {}'.format('\n'.join(miss_files)))
                    logger.info('re-building vocab')
                    utils.build_data(train_files, test_files, csv_file, logger)
                    utils.build_vocab(
                        train_files, check_list['word_vocab'], check_list['intent_vocab'], logger, word_level=True)

                print('-------Loading vocab files--------')
                word2id, id2word = utils.read_vocab(check_list['word_vocab'], logger, limit=config['word_vocab_size'])
                intent2id, id2intent = utils.read_vocab(check_list['intent_vocab'], logger)

                config = utils.update_vocab_size(config, intent2id, dict())
                embed = utils.load_embed(word2id, config['word_vocab_size'], config['word_embed_size'],
                                         config['pretrained_embed'], logger)
                model = CVAEModel_bert(config, utils.GO_ID, utils.EOS_ID)
                sess.run(tf.global_variables_initializer())


            print('----------Preparing data------------')
            for i, file in enumerate(train_files):
                if not os.path.isfile(train_prep_file[i]):
                    logger.info("{}/{}: {} not found, generating from {}".format(i + 1, len(train_prep_file),
                                                                                 train_prep_file[i], train_files[i]))
                    gen_tfrecord(word2id, intent2id, [train_files[i]], [train_prep_file[i]], logger,
                                 config['max_utter_len'], word_level=True, shuffle=True)

            for i, file in enumerate(valid_files):
                if not os.path.isfile(valid_prep_file[i]):
                    logger.info("{}/{}: {} not found, generating from {}".format(i + 1, len(valid_prep_file),
                                                                                 valid_prep_file[i], valid_files[i]))
                    gen_tfrecord(word2id, intent2id, [valid_files[i]], [valid_prep_file[i]], logger,
                                   config['max_utter_len'], word_level=True)

            print('-----------Loading data----------')
            train_dataset = TFRData(utils.PAD_ID, repeat=config['max_epoch'], shuffle_buffer=10000, seed=config['seed'])
            valid_dataset = TFRData(utils.PAD_ID, repeat=False)
            peep_dataset = TFRData(utils.PAD_ID, repeat=True, shuffle_buffer=100)
            train_dataset.init(sess, train_prep_file, config['batch_size'])
            train_handler = train_dataset.get_handler(sess)

            peep_dataset.init(sess, valid_prep_file, 5)
            peep_handler = peep_dataset.get_handler(sess)

            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'), sess.graph)

            train_loss = 0.0
            train_kl_loss = 0.0
            train_ppl_loss = 0.0
            train_kl_w = 0.0
            train_anneal_kl_loss = 0.0
            train_reconstruct_loss = 0.0
            train_time = 0.0
            last_improved = 0
            start_time = time.time()

            best_ppl_loss = 1e18
            prev_ppl_loss = [1e18 for _ in range(5)]

            logger.info('Training')
            while True:
                try:
                    start_time = time.time()
                    output = [model.KL_weight, model.train_loss, model.KL_loss, model.anneal_KL_loss,
                              model.ppl_loss, model.reconstruct_loss, model.train_op]
                    input_dict = {model.data_handler: train_handler, model.keep_rate: config['keep_rate']}
                    kl_w, loss, kl_loss, anneal_kl_loss, ppl_loss, reconstruct_loss, _ = sess.run(
                        output, feed_dict=input_dict)

                    train_loss += loss
                    train_kl_loss += kl_loss
                    train_anneal_kl_loss += anneal_kl_loss
                    train_kl_w += kl_w
                    train_reconstruct_loss += reconstruct_loss
                    train_ppl_loss += ppl_loss
                    train_time += (time.time() - start_time)

                    total_step = model.global_step.eval(sess)

                    if total_step % config['save_per_iter'] == 0:
                        # save
                        print("====================save model=====================")
                        model.saver.save(sess, os.path.join(train_dir, 'model.ckpt'), global_step=total_step)

                        train_loss /= config['save_per_iter']
                        train_kl_loss /= config['save_per_iter']
                        train_anneal_kl_loss /= config['save_per_iter']
                        train_kl_w /= config['save_per_iter']
                        train_ppl_loss /= config['save_per_iter']
                        train_time /= config['save_per_iter']
                        train_reconstruct_loss /= config['save_per_iter']

                        learning_rate = model.learning_rate.eval(sess)

                        add_summary(train_writer, total_step, data=[('loss/loss', train_loss),
                                                                    ('loss/reconstruct_loss', train_reconstruct_loss),
                                                                    ('loss/kl_loss', train_kl_loss),
                                                                    ('loss/anneal_kl_loss', train_anneal_kl_loss),
                                                                    ('loss/ppl', np.exp(train_ppl_loss)),
                                                                    ('weight/kl_w', train_kl_w),
                                                                    ('weight/lr', learning_rate)])

                        if train_ppl_loss > max(prev_ppl_loss):
                            sess.run(model.learning_rate_decay_op)
                        prev_ppl_loss = prev_ppl_loss[1:] + [train_ppl_loss]

                        valid_dataset.init(sess, valid_prep_file, config['batch_size'])
                        valid_data_handler = valid_dataset.get_handler(sess)
                        valid_kl_w, valid_loss, valid_kl_loss, valid_anneal_kl_loss, valid_ppl_loss = model.eval(
                            sess, valid_data_handler)

                        add_summary(valid_writer, total_step, data=[('loss/loss', valid_loss),
                                                                    ('loss/kl_loss', valid_kl_loss),
                                                                    ('loss/anneal_kl_loss', valid_anneal_kl_loss),
                                                                    ('loss/ppl', np.exp(valid_ppl_loss)),
                                                                    ('weight/kl_w', valid_kl_w)])

                        if valid_ppl_loss < best_ppl_loss:
                            best_ppl_loss = valid_ppl_loss
                            model.best_saver.save(sess=sess, save_path=os.path.join(best_model, 'best_model.ckpt'),
                                                  global_step=total_step)
                            last_improved = total_step
                            improved_str = '*'
                        else:
                            improved_str = ' '

                        format_str = 'It: {0:>5} t loss: {1:>4.4f} t kl loss: {2:>4.4f} ' + \
                                     't anneal kl loss: {3:>4.4f} t ppl: {4:>4.4f} t kl_w: {5:3.4f} t: {6:>3.4f}ms ' + \
                                     'lr: {7:>.6}'
                        out1 = format_str.format(total_step, train_loss, train_kl_loss, train_anneal_kl_loss,
                                                 np.exp(train_ppl_loss), train_kl_w, train_time * 1000, learning_rate)

                        format_str = 'v loss: {0:>4.4f} v kl loss: {1:>4.4f} ' + \
                                     'v anneal kl loss: {2:>4.4f} v ppl: {3:>4.4f} v kl_w: {4:>4.4f} last best:{5} {6}'
                        out2 = format_str.format(valid_loss, valid_kl_loss, valid_anneal_kl_loss,
                                                 np.exp(valid_ppl_loss), valid_kl_w, last_improved, improved_str)

                        logger.info(out1)
                        logger.info(out2)

                        train_loss = 0.0
                        train_kl_loss = 0.0
                        train_ppl_loss = 0.0
                        train_kl_w = 0.0
                        train_anneal_kl_loss = 0.0
                        train_time = 0.0

                        intents, input_utter, greedy_infer, beam_infer, beam_infer_topk = sess.run(
                            [model.intents, model.utter, model.infer_utter, model.beam_utter, model.beam_utter_topk],
                            feed_dict={model.data_handler: peep_handler, model.keep_rate: 1.0})

                        utils.peep_output(intents, id2intent, id2word, input_utter, greedy_infer, beam_infer, beam_infer_topk, logger)

                except tf.errors.OutOfRangeError:
                    logger.info('Training complete')
                    break
    else:
        if args.use_best:
            ckpt_dir = best_model
        else:
            ckpt_dir = train_dir

        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if not ckpt_file:
            logger.error("cannot find checkpoint in {}".format(ckpt_dir))
            exit(1)

        logger.info("found checkpoint {}".format(ckpt_file))

        word2id, id2word = utils.read_vocab(word_vocab_file, logger, config['word_vocab_size'])
        intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
        config = utils.update_vocab_size(config, intent2id, dict())


        logger.info('Processing testing data')
        gen_tfrecord(word2id, intent2id, test_files, test_prep_file, logger, config['max_utter_len'], word_level=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        try:
            with tf.Session(config=tf_config) as sess:
                logger.info('loading model from {}'.format(ckpt_file))
                model = CVAEModel_bert(config, utils.GO_ID, utils.EOS_ID)
                model.saver.restore(sess, ckpt_file)
                logger.info('Loading test data')
                test_dataset = TFRData(utils.PAD_ID, repeat=False)
                test_dataset.init(sess, test_prep_file, config['batch_size'])
                test_handler = test_dataset.get_handler(sess)
                logger.info('evaluating test file')
                while True:
                    try:
                        intents, input_utter, greedy_infer, beam_infer, beam_infer_topk = sess.run(
                            [model.intents, model.utter, model.infer_utter, model.beam_utter, model.beam_utter_topk],
                            feed_dict={model.data_handler: test_handler, model.keep_rate: 1.0})
                        utils.peep_output(intents, id2intent, id2word, input_utter, greedy_infer, beam_infer, beam_infer_topk, logger, write_csv=True, output_file_name=args.csv_file_name)
                    except:
                        break
        except:
            exit(1)

except:
    logger.error(traceback.format_exc())
