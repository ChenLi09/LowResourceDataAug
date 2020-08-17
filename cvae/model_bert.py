import numpy as np
import tensorflow as tf
import cvae.my_modeling_bert as my_modeling_bert
from cvae.beam_search_decoder import BeamSearchDecoder_topk
import os


class CVAEModel_bert(object):
    def __init__(self, config=None, go_id=None, eos_id=None, embed=None):
        self.config = config
        self.go_id = go_id
        self.eos_id = eos_id
        # initialize the training process
        with tf.name_scope('inputs'):
            self.data_handler = tf.placeholder(tf.string, shape=[], name='dataset_handler')
            iterator = tf.data.Iterator.from_string_handle(
                self.data_handler, {"utter": tf.int32, "id": tf.int32, "len": tf.int32, "intent": tf.int32},
                output_shapes={"utter": [None, None], "id": [None], "len": [None], "intent": [None]}
            )
            data = iterator.get_next(name='data_input')
            self.intents = data['intent']# [?,]
            self.utter = data['utter']# [?,?]
            self.utter_len = data['len']# [?,]

        with tf.name_scope('common_variables'):
            self.learning_rate = tf.Variable(float(self.config['learning_rate']), trainable=False,
                                             dtype=tf.float32, name='lr', use_resource=True)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.config['lr_decay'],
                                                                    name='lr_decay')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.keep_rate = tf.placeholder(tf.float32, shape=[], name='keep_rate')
            max_utter_len = tf.reduce_max(self.utter_len)

        with tf.name_scope("embedding"):
            # build the embedding table and embedding input
            if embed is None:
                # initialize the embedding randomly
                self.embed = tf.get_variable('embed', [self.config['word_vocab_size'],
                                                       self.config['word_embed_size']], tf.float32,
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
            else:
                # initialize the embedding by pre-trained word vectors
                self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

            self.intent_embed = tf.get_variable('intent_embed',
                                                [self.config['intent_num'], self.config['intent_embed_size']],
                                                tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.intent_resp = tf.nn.embedding_lookup(self.intent_embed, self.intents)   # [bs, intent_embed_size]
            self.utter_resp = tf.nn.embedding_lookup(self.embed, self.utter)   # [bs, ?, embed_size]
            self.utter_resp = tf.nn.dropout(self.utter_resp, keep_prob=self.keep_rate, name='dropped_utter_resp')
            batch_size = tf.shape(self.intent_resp)[0]
            self.batch_size = batch_size

            self.dec_input = tf.split(self.utter, [max_utter_len - 1, 1], 1)[0]   # without <EOS>
            self.dec_target = tf.split(self.utter, [1, max_utter_len - 1], 1)[1]   # without <GO>
            self.dec_input_resp = tf.nn.embedding_lookup(self.embed, self.dec_input)

        with tf.name_scope("rnn_cells"):
            cell_enc = self._get_rnn_cell(self.config['rnn_size'], self.config['keep_rate'],
                                          self.config['num_layers'])
            cell_dec = self._get_rnn_cell(self.config['rnn_size'], self.config['keep_rate'],
                                          self.config['num_layers'])
        print('================BERT==================')
        with tf.name_scope("encoder"):
            bert_dir = 'cvae/chinese_bert/'
            bert_config_file = os.path.join(bert_dir, 'bert_config.json')
            bert_config = my_modeling_bert.BertConfig.from_json_file(bert_config_file)
            is_training = False
            model = my_modeling_bert.BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=self.utter,
                use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
            )
            enc_state = model.get_pooled_output()
            '''
            enc_output, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.utter_resp,
                                                      sequence_length=self.utter_len - 1, dtype=tf.float32,
                                                      scope='encoder_rnn')
            enc_state = [tf.concat((enc_state[i].c, enc_state[i].h), 1) for i in range(config['num_layers'])]
            enc_state = tf.concat(enc_state, 1)
            '''

        with tf.name_scope("recognition_network"):
            recog_input = tf.concat((self.intent_resp, enc_state), 1)
            recog_output = tf.layers.dense(recog_input, 2 * config['latent_size'], name='recog_net')
            recog_mu, recog_logvar = tf.split(recog_output, 2, 1)
            recog_z = self._sample_gaussian((batch_size, config['latent_size']), recog_mu, recog_logvar)

        with tf.name_scope("prior_network"):
            # prior_input = self.intent_resp
            # print(self.intent_resp)
            prior_input = tf.concat((self.intent_resp, enc_state), 1)
            prior_fc_1 = tf.layers.dense(prior_input, config['latent_size'], activation=tf.tanh, name='prior2latent')
            prior_output = tf.layers.dense(prior_fc_1, 2 * config['latent_size'], name='prior_net')
            prior_mu, prior_logvar = tf.split(prior_output, 2, 1)
            self.prior_z = self._sample_gaussian((batch_size, config['latent_size']), prior_mu, prior_logvar)

        with tf.name_scope("decode"):
            dec_init_fn = tf.layers.Dense(config['rnn_size'] * 2, use_bias=True, activation=None, name='dec_state_init')
            output_layer = tf.layers.Dense(config['word_vocab_size'], use_bias=True, name='dec_output_layer',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            with tf.name_scope("training"):
                dec_init = self._get_dec_state(recog_z, dec_init_fn, self.intent_resp, config['num_layers'])
                train_helper = tf.contrib.seq2seq.TrainingHelper(self.dec_input_resp, self.utter_len - 1)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(cell_dec, train_helper, dec_init, output_layer)
                train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True,
                                                                        scope="decoder")
                self.decoder_logits = train_outputs.rnn_output

                with tf.name_scope("losses"):
                    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_logits,
                                                                              labels=self.dec_target)
                    decoder_mask = tf.sequence_mask(self.utter_len - 1, max_utter_len - 1, dtype=tf.float32)
                    loss_mask = tf.cast(tf.greater(self.utter_len - 1, 0), dtype=tf.float32)

                    self.reconstruct_loss = tf.reduce_sum(crossent * decoder_mask) / tf.cast(batch_size, tf.float32)
                    self.ppl_loss = tf.reduce_sum(crossent * decoder_mask) / tf.reduce_sum(decoder_mask)
                    self.KL_loss = tf.reduce_sum(
                        loss_mask * self._KL_divergence(
                            prior_mu, prior_logvar, recog_mu, recog_logvar)) / tf.cast(batch_size, tf.float32)
                    self.KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / config['full_kl_step'])
                    self.anneal_KL_loss = self.KL_weight * self.KL_loss

                    self.train_loss = tf.square(self.anneal_KL_loss + self.reconstruct_loss - config['min_reconstruct_loss'])

            with tf.name_scope("inference"):
                self.infer_init = self._get_dec_state(self.prior_z, dec_init_fn, self.intent_resp, config['num_layers'])

                infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embed, tf.fill([tf.shape(self.intent_resp)[0]], self.go_id), self.eos_id)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell_dec, infer_helper, self.infer_init, output_layer)
                infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    infer_decoder, impute_finished=True, maximum_iterations=config['max_decode_len'],
                    scope='infer_decoder')
                self.infer_utter = infer_output.sample_id

            with tf.name_scope("beam_topk"):
                tiled_infer_init = tf.contrib.seq2seq.tile_batch(self.infer_init, multiplier=config['beam_width'])
                beam_decoder = BeamSearchDecoder_topk(
                    cell=cell_dec, embedding=self.embed, initial_state=tiled_infer_init, output_layer=output_layer,
                    start_tokens=tf.fill([tf.shape(self.intent_resp)[0]], self.go_id), end_token=self.eos_id,
                    beam_width=config['beam_width'])
                beam_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    beam_decoder, impute_finished=False, maximum_iterations=config['max_decode_len'])
                self.beam_utter_topk = beam_output.predicted_ids

            with tf.name_scope("beam"):
                tiled_infer_init = tf.contrib.seq2seq.tile_batch(self.infer_init, multiplier=config['beam_width'])
                beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell_dec, embedding=self.embed, initial_state=tiled_infer_init, output_layer=output_layer,
                    start_tokens=tf.fill([tf.shape(self.intent_resp)[0]], self.go_id), end_token=self.eos_id,
                    beam_width=config['beam_width'])
                beam_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    beam_decoder, impute_finished=False, maximum_iterations=config['max_decode_len'])
                self.beam_utter = beam_output.predicted_ids

        with tf.name_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, params)
            clipped_grad, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = self.opt.apply_gradients(
                zip(clipped_grad, params), global_step=self.global_step, name='train_op')

        self.saver = tf.train.Saver(
            write_version=tf.train.SaverDef.V2, max_to_keep=5, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.best_saver = tf.train.Saver(max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        # for var in tf.trainable_variables():
        #     print(var)

    def _get_rnn_cell(self, size, keep_rate, layer_num):
        return tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size), variational_recurrent=True,
                dtype=tf.float32, output_keep_prob=keep_rate, state_keep_prob=keep_rate)) for _ in range(layer_num)])

    def _sample_gaussian(self, shape, mu, logvar):
        normal = tf.random_normal(shape=shape, dtype=tf.float32)
        z = tf.exp(logvar / 2) * normal + mu
        return z

    def _get_dec_state(self, sample_z, dec_init_fn, context, layers):
        state = tf.concat((sample_z, context), 1)
        state = dec_init_fn(state)
        return tuple([tf.nn.rnn_cell.LSTMStateTuple(tf.split(state, 2, 1)[0],
                                                    tf.split(state, 2, 1)[1]) for _ in range(layers)])

    def _KL_divergence(self, prior_mu, prior_logvar, recog_mu, recog_logvar):
        KL_divergence = 0.5 * (tf.exp(recog_logvar - prior_logvar) + tf.pow(recog_mu - prior_mu, 2) / tf.exp(
            prior_logvar) - 1 - (recog_logvar - prior_logvar))
        return tf.reduce_sum(KL_divergence, axis=1)

    def eval(self, sess, data_handler):
        step_count = 0
        valid_loss = 0.0
        valid_kl_loss = 0.0
        valid_ppl_loss = 0.0
        valid_kl_w = 0.0
        valid_anneal_kl_loss = 0.0
        output = [self.KL_weight, self.train_loss, self.KL_loss, self.anneal_KL_loss, self.ppl_loss]
        input_dict = {self.data_handler: data_handler, self.keep_rate: 1.0}
        while True:
            try:
                kl_w, loss, kl_loss, anneal_kl_loss, ppl_loss = sess.run(output, feed_dict=input_dict)
                valid_loss += loss
                valid_kl_loss += kl_loss
                valid_anneal_kl_loss += anneal_kl_loss
                valid_kl_w += kl_w
                valid_ppl_loss += ppl_loss
                step_count += 1
            except tf.errors.OutOfRangeError:
                break

        valid_loss /= step_count
        valid_kl_loss /= step_count
        valid_anneal_kl_loss /= step_count
        valid_kl_w /= step_count
        valid_ppl_loss /= step_count

        return valid_kl_w, valid_loss, valid_kl_loss, valid_anneal_kl_loss, valid_ppl_loss