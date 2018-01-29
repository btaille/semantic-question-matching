import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
from data_utils import DataIterator, DataIteratorAE

max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class HybridNet(object):
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.vocab_size = len(embeddings)

    def bilstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, sequence_length=seq_len,
                                                                        dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output, state

    def lstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        output, state = tf.nn.dynamic_rnn(cell_fw, seq, sequence_length=seq_len, dtype=tf.float32)
        return output, state

    def decode(self, initial_state):
        cell_fw = tf.nn.rnn_cell.LSTMCell(2 * self.config.hidden_size)
        dec_inputs = tf.zeros([self.config.batch_size, self.config.padlen, 1])
        output, state = tf.nn.dynamic_rnn(cell_fw, dec_inputs, initial_state=initial_state)

        return output, state

    def activation(self, x):
        assert self.config.fc_activation in ["sigmoid", "relu", "tanh"]
        if self.config.fc_activation == "sigmoid":
            return tf.nn.sigmoid(x)
        elif self.config.fc_activation == "relu":
            return tf.nn.relu(x)
        elif self.config.fc_activation == "tanh":
            return tf.nn.tanh(x)

    def build(self):
        ### Placeholders
        self.q1 = tf.placeholder_with_default(tf.zeros([self.config.batch_size, self.config.padlen], tf.int64),
                                              shape=[None, None], name="question1")
        self.l1 = tf.placeholder_with_default(tf.zeros([self.config.batch_size], tf.int64), shape=[None], name="len1")

        self.q2 = tf.placeholder_with_default(tf.zeros([self.config.batch_size, self.config.padlen], tf.int64),
                                              shape=[None, None], name="question2")
        self.l2 = tf.placeholder_with_default(tf.zeros([self.config.batch_size], tf.int64), shape=[None], name="len2")

        self.y = tf.placeholder_with_default(tf.zeros([self.config.batch_size], tf.int64), shape=[None],
                                             name="is_duplicate")

        self.q_ae = tf.placeholder_with_default(tf.zeros([self.config.batch_size, self.config.padlen], tf.int64),
                                                shape=[None, None], name="questionAE")
        self.l_ae = tf.placeholder_with_default(tf.zeros([self.config.batch_size], tf.int64), shape=[None],
                                                name="lenAE")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        ### Optimizer
        with tf.variable_scope("train_step") as scope:
            if self.config.lr_method == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.lr_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.lr)

                ### Embedding layer
        with tf.variable_scope("word_embeddings") as scope:
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            we1 = tf.nn.embedding_lookup(_word_embeddings, self.q1, name="q1_embedded")
            we2 = tf.nn.embedding_lookup(_word_embeddings, self.q2, name="q2_embedded")
            we_ae = tf.nn.embedding_lookup(_word_embeddings, self.q_ae, name="q_ae_embedded")

            we1 = tf.nn.dropout(we1, keep_prob=1 - self.dropout)
            we2 = tf.nn.dropout(we2, keep_prob=1 - self.dropout)
            we_ae = tf.nn.dropout(we_ae, keep_prob=1 - self.dropout)

        ### ENCODER
        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.l1)
            scope.reuse_variables()
            lstm2, state2 = self.bilstm(we2, self.l2)
            scope.reuse_variables()
            lstm_ae, state_ae = self.bilstm(we_ae, self.l_ae)

        ### DECODER
        state_ae = tf.contrib.rnn.LSTMStateTuple(tf.concat([state_ae[0].c, state_ae[1].c], axis=1),
                                                 tf.concat([state_ae[0].h, state_ae[1].h], axis=1))

        decoded, _ = self.decode(state_ae)

        ### logits
        with tf.variable_scope("decoder_projection") as scope:
            W = tf.Variable(tf.random_normal([1, 2 * self.config.hidden_size, self.vocab_size], stddev=1e-3), name="w")
            self.logits = tf.nn.conv1d(decoded, W, 1, "VALID", name="logits")

        ### Loss
        losses_mask = tf.sequence_mask(lengths=self.l_ae, maxlen=self.config.padlen, dtype=tf.float32)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.q_ae, logits=self.logits)

        self.cross_entropy_ae = tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)

        self.train_step_ae = optimizer.minimize(self.cross_entropy_ae)

        ### Evaluation
        correct_prediction_ae = tf.equal(tf.argmax(self.logits, axis=-1), self.q_ae)

        self.accuracy_ae = tf.reduce_sum(tf.cast(correct_prediction_ae, tf.float32) * losses_mask) / tf.reduce_sum(
            losses_mask)

        ### Summaries
        with tf.variable_scope("summaries") as scope:

            # train
            tf.summary.scalar('cross_entropy_ae', self.cross_entropy_ae)
            tf.summary.scalar('accuracy_ae', self.accuracy_ae)
            self.merged_ae = tf.summary.merge_all()

            # test
            self.acc_value_ae = tf.placeholder_with_default(tf.constant(0.0), shape=())
            self.ce_value_ae = tf.placeholder_with_default(tf.constant(0.0), shape=())
            acc_summary_ae = tf.summary.scalar('accuracy_ae', self.acc_value_ae)
            ce_summary_ae = tf.summary.scalar('cross_entropy_ae', self.ce_value_ae)
            self.merged_eval_ae = tf.summary.merge([acc_summary_ae, ce_summary_ae])

        ### INFERENCE
        ### Max pooling
        lstm1_pool = max_pool(lstm1)
        lstm2_pool = max_pool(lstm2)

        ### Features
        flat1 = tf.contrib.layers.flatten(lstm1_pool)
        flat2 = tf.contrib.layers.flatten(lstm2_pool)
        mult = tf.multiply(flat1, flat2)
        diff = tf.abs(tf.subtract(flat1, flat2))

        if self.config.feats == "raw":
            concat = tf.concat([flat1, flat2], axis=-1)
        elif self.config.feats == "dist":
            concat = tf.concat([mult, diff], axis=-1)
        elif self.config.feats == "all":
            concat = tf.concat([flat1, flat2, mult, diff], axis=-1)

        ### FC layers
        concat_size = int(concat.get_shape()[1])
        intermediary_size = 2 + (concat_size - 2) // 2
        # intermediary_size = 512

        with tf.variable_scope("fc1") as scope:
            W1 = tf.Variable(tf.random_normal([concat_size, intermediary_size], stddev=1e-3), name="w_fc")
            b1 = tf.Variable(tf.zeros([intermediary_size]), name="b_fc")

            z1 = tf.matmul(concat, W1) + b1

            if self.config.batch_norm:
                epsilon = 1e-3
                batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
                scale1, beta1 = tf.Variable(tf.ones([intermediary_size])), tf.Variable(tf.zeros([intermediary_size]))
                z1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, scale1, epsilon)

            fc1 = tf.nn.dropout(self.activation(z1), keep_prob=1 - self.dropout)
            tf.summary.histogram('fc1', fc1)
            tf.summary.histogram('W1', W1)
            tf.summary.histogram('b1', b1)

        with tf.variable_scope("fc2") as scope:
            W2 = tf.Variable(tf.random_normal([intermediary_size, 2], stddev=1e-3), name="w_fc")
            b2 = tf.Variable(tf.zeros([2]), name="b_fc")

            z2 = tf.matmul(fc1, W2) + b2

            if self.config.batch_norm:
                epsilon = 1e-3
                batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
                scale2, beta2 = tf.Variable(tf.ones([2])), tf.Variable(tf.zeros([2]))
                z2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, epsilon)

            self.fc2 = z2
            tf.summary.histogram('fc2', self.fc2)
            tf.summary.histogram('W2', W2)
            tf.summary.histogram('b2', b2)

        ### Loss
        self.cross_entropy_inf = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc2))

        self.train_step_inf = optimizer.minimize(self.cross_entropy_inf)

        ### Evaluation
        correct_prediction_inf = tf.equal(tf.argmax(self.fc2, 1), self.y)
        self.accuracy_inf = tf.reduce_mean(tf.cast(correct_prediction_inf, tf.float32))

        ### Init
        self.init = tf.global_variables_initializer()

        ### Summaries
        with tf.variable_scope("summaries") as scope:

            # train
            tf.summary.scalar('cross_entropy_inf', self.cross_entropy_inf)
            tf.summary.scalar('accuracy_inf', self.accuracy_inf)
            self.merged_inf = tf.summary.merge_all()

            # test
            self.acc_value_inf = tf.placeholder_with_default(tf.constant(0.0), shape=())
            self.ce_value_inf = tf.placeholder_with_default(tf.constant(0.0), shape=())
            acc_summary_inf = tf.summary.scalar('accuracy_inf', self.acc_value_inf)
            ce_summary_inf = tf.summary.scalar('cross_entropy_inf', self.ce_value_inf)
            self.merged_eval_inf = tf.summary.merge([acc_summary_inf, ce_summary_inf])

        self.saver = tf.train.Saver()

    def run_epoch_inf(self, sess, train, dev, test, epoch, lr, test_step=1000, restrict=0):
        iterator = DataIterator(train, self.config.batch_size, strict=1, restrict=restrict)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size
        dev_acc = 0

        accuracy, cross = 0, 0
        for i in tqdm(range(nbatches)):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.q2: q2, self.l1: l1, self.l2: l2, self.y: y,
                  self.dropout: self.config.dropout, self.lr: lr}
            _, summary = sess.run([self.train_step_inf, self.merged_inf], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.train_writer.add_summary(summary, epoch * nbatches + i)

            if i % test_step == 0 and i > 0 or i == nbatches - 1:
                summary, dev_acc = self.run_evaluate_inf(sess, dev)
                self.dev_writer.add_summary(summary, epoch * nbatches + i)
                print("Step {}/{}".format(i, nbatches))
                print("dev acc inf : {:04.2f}".format(100 * dev_acc))

                summary, test_acc = self.run_evaluate_inf(sess, test)
                self.test_writer.add_summary(summary, epoch * nbatches + i)
                print("test acc inf : {:04.2f}".format(100 * test_acc))

        return dev_acc

    def run_evaluate_inf(self, sess, data):
        iterator = DataIterator(data, self.config.batch_size, strict=1)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size

        accuracy, cross = 0, 0
        for i in range(nbatches):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.q2: q2, self.l1: l1, self.l2: l2, self.y: y,
                  self.dropout: 0,
                  self.lr: 0}
            acc, ce = sess.run([self.accuracy_inf, self.cross_entropy_inf], feed_dict=fd)

            accuracy += acc * len(q1)
            cross += ce * len(q1)

        accuracy /= iterator.max
        cross /= iterator.max

        summary = sess.run(self.merged_eval_inf, feed_dict={self.acc_value_inf: accuracy, self.ce_value_inf: cross})

        return summary, accuracy

    def run_epoch_ae(self, sess, train, dev, test, epoch, lr, test_step=1000):
        iterator = DataIteratorAE(train, self.config.batch_size, strict=1)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size
        dev_acc = 0

        accuracy, cross = 0, 0
        for i in tqdm(range(nbatches)):
            q, l = iterator.__next__()
            fd = {self.q_ae: q, self.l_ae: l, self.dropout: self.config.dropout, self.lr: lr}
            _, summary = sess.run([self.train_step_ae, self.merged_ae], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.train_writer.add_summary(summary, epoch * nbatches + i)

            if i % test_step == 0 and i > 0 or i == nbatches - 1:
                summary, dev_acc = self.run_evaluate_ae(sess, dev)
                self.dev_writer.add_summary(summary, epoch * nbatches + i)
                print("Step {}/{}".format(i, nbatches))
                print("dev acc ae : {:04.2f}".format(100 * dev_acc))

                summary, test_acc = self.run_evaluate_ae(sess, test)
                self.test_writer.add_summary(summary, epoch * nbatches + i)
                print("test acc ae : {:04.2f}".format(100 * test_acc))

        return dev_acc

    def run_evaluate_ae(self, sess, data):
        iterator = DataIteratorAE(data, self.config.batch_size, strict=1)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size

        accuracy, cross = 0, 0
        for i in range(nbatches):
            q, l = iterator.__next__()
            fd = {self.q_ae: q, self.l_ae: l, self.dropout: 0, self.lr: 0}
            acc, ce = sess.run([self.accuracy_ae, self.cross_entropy_ae], feed_dict=fd)

            accuracy += acc * len(q)
            cross += ce * len(q)

        accuracy /= iterator.max
        cross /= iterator.max

        summary = sess.run(self.merged_eval_ae, feed_dict={self.acc_value_ae: accuracy, self.ce_value_ae: cross})

        return summary, accuracy

    def run_epoch_mixed(self, sess, train, dev, test, epoch, lr, test_step=1000, restrict=0, ratio=1):
        inf_iterator = DataIterator(train, self.config.batch_size, strict=1, restrict=restrict, ratio=ratio)
        ae_iterator = DataIteratorAE(train, self.config.batch_size, strict=1)

        nbatches_inf = (inf_iterator.max + self.config.batch_size - 1) // self.config.batch_size
        nbatches_ae = (ae_iterator.max + self.config.batch_size - 1) // self.config.batch_size

        i_inf = 0
        i_ae = 0
        i = 0
        nbatches = nbatches_inf + nbatches_ae

        for _ in tqdm(range(nbatches)):
            rand = np.random.random()
            threshold = (nbatches_ae - i_ae) / ((nbatches_ae - i_ae) + (nbatches_inf - i_inf))

            if rand > threshold:
                task = "inf"
            else:
                task = "ae"

            if task == "inf":
                q1, q2, l1, l2, y = inf_iterator.__next__()
                fd = {self.q1: q1, self.q2: q2, self.l1: l1, self.l2: l2, self.y: y,
                      self.dropout: 0,
                      self.lr: 0}
                _, summary = sess.run([self.train_step_inf, self.merged_inf], feed_dict=fd)

                i_inf += 1

            elif task == "ae":
                q, l = ae_iterator.__next__()
                fd = {self.q_ae: q, self.l_ae: l, self.dropout: self.config.dropout, self.lr: lr}
                _, summary = sess.run([self.train_step_ae, self.merged_ae], feed_dict=fd)

                i_ae += 1

            i += 1

            # tensorboard
            if i_inf % 10 == 0:
                self.train_writer.add_summary(summary, epoch * nbatches + i)

            if i % test_step == 0 and i > 0 or i == nbatches - 1:
                summary, dev_acc_inf = self.run_evaluate_inf(sess, dev)
                self.dev_writer.add_summary(summary, epoch * nbatches + i)
                print("Step inf {}/{}".format(i_inf, nbatches_inf))
                print("dev acc inf{:04.2f}".format(100 * dev_acc_inf))

                summary, test_acc_inf = self.run_evaluate_inf(sess, test)
                self.test_writer.add_summary(summary, epoch * nbatches + i)
                print("test acc inf{:04.2f}".format(100 * test_acc_inf))

                summary, dev_acc_ae = self.run_evaluate_ae(sess, dev)
                self.dev_writer.add_summary(summary, epoch * nbatches + i)
                print("Step ae {}/{}".format(i_ae, nbatches_ae))
                print("dev acc ae{:04.2f}".format(100 * dev_acc_ae))

                summary, test_acc_ae = self.run_evaluate_ae(sess, test)
                self.test_writer.add_summary(summary, epoch * nbatches + i)
                print("test acc ae{:04.2f}".format(100 * test_acc_ae))

        return dev_acc_inf, dev_acc_ae

    def train(self, train_data, dev_data, test_data, restrict=0, task="joint", ratio=1):
        assert task in ["inference", "autoencoder", "joint"]

        best_acc = 0
        nepoch_no_improv = 0

        with tf.Session(config=tf_config) as sess:
            self.train_writer = tf.summary.FileWriter(self.config.log_path + "train", sess.graph)
            self.dev_writer = tf.summary.FileWriter(self.config.log_path + "dev", sess.graph)
            self.test_writer = tf.summary.FileWriter(self.config.log_path + "test", sess.graph)

            sess.run(self.init)

            lr = self.config.lr

            print("Training in {}".format(self.config.conf_dir))
            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{} :".format(epoch + 1, self.config.n_epochs))

                if task == "joint":
                    print("Unsupervised training of autoencoder")
                    for _ in range(ratio):
                        dev_acc_ae = self.run_epoch_ae(sess, train_data, dev_data, test_data, epoch, lr)
                    print("Supervised training of inference")
                    dev_acc = self.run_epoch_inf(sess, train_data, dev_data, test_data, epoch, lr, restrict=restrict)

                elif task == "autoencoder":
                    print("Unsupervised training of autoencoder")
                    dev_acc = self.run_epoch_ae(sess, train_data, dev_data, test_data, epoch, lr)

                elif task == "inference":
                    print("Supervised training of inference")
                    dev_acc = self.run_epoch_inf(sess, train_data, dev_data, test_data, epoch, lr, restrict=restrict)

                lr *= self.config.lr_decay

                if dev_acc > best_acc:
                    nepoch_no_improv = 0
                    if not os.path.exists(self.config.model_path):
                        os.makedirs(self.config.model_path)
                    self.saver.save(sess, self.config.model_path)
                    best_acc = dev_acc
                    print("New best score on dev !")

                else:
                    lr /= self.config.lr_divide
                    nepoch_no_improv += 1
                    if nepoch_no_improv >= self.config.nepochs_no_improv:
                        print("Early stopping after {} epochs without improvements".format(nepoch_no_improv))
                        break


    def train_mixed(self, train_data, dev_data, test_data, restrict=0, ratio=1):

        best_acc = 0
        nepoch_no_improv = 0

        with tf.Session(config=tf_config) as sess:
            self.train_writer = tf.summary.FileWriter(self.config.log_path + "train", sess.graph)
            self.dev_writer = tf.summary.FileWriter(self.config.log_path + "dev", sess.graph)
            self.test_writer = tf.summary.FileWriter(self.config.log_path + "test", sess.graph)

            sess.run(self.init)

            lr = self.config.lr

            print("Training in {}".format(self.config.conf_dir))
            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{} :".format(epoch + 1, self.config.n_epochs))
                print("Joint training of inference and autoencoder")
                dev_acc_inf, dev_acc_ae = self.run_epoch_mixed(sess, train_data, dev_data, test_data, epoch, lr,
                                                               restrict=restrict, ratio=ratio)

                lr *= self.config.lr_decay

                if dev_acc_inf > best_acc:
                    nepoch_no_improv = 0
                    if not os.path.exists(self.config.model_path):
                        os.makedirs(self.config.model_path)
                    self.saver.save(sess, self.config.model_path)
                    best_acc = dev_acc_inf
                    print("New best score on dev !")

                else:
                    lr /= self.config.lr_divide
                    nepoch_no_improv += 1
                    if nepoch_no_improv >= self.config.nepochs_no_improv:
                        print("Early stopping after {} epochs without improvements".format(nepoch_no_improv))
                        break
