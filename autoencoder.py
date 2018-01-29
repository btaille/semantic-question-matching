import tensorflow as tf
from tqdm import tqdm
import os
from data_utils import DataIterator

max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class AutoEncoder(object):
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

    #     def decode(self, code, initial_state):
    #         decoder_output_list = []
    #         cell_fw = tf.nn.rnn_cell.LSTMCell(2 * self.config.hidden_size)
    #         print(code)
    #         initial_states = []
    #         for i in range(self.config.batch_size):
    #             initial_states.append((initial_state.c[i], initial_state.h[i]))

    # #         output, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=dec_input, initial_state=initial_state, cell=cell_fw)
    #         for i, init in enumerate(initial_states):
    #             if i > 0:
    #                 tf.get_variable_scope().reuse_variables()
    #             dec_input = [tf.zeros([1, self.config.hidden_size])]
    #             decoder_output, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=dec_input, initial_state=init, cell=cell_fw)
    #             decoder_output_list.append(tf.stack(decoder_output, axis=1))

    #         return decoder_output_list


    def decode(self, initial_state):
        cell_fw = tf.nn.rnn_cell.LSTMCell(2 * self.config.hidden_size)
        dec_inputs = [tf.zeros(2 * self.config.hidden_size) for _ in range(self.config.batch_size)]
        dec_inputs = tf.zeros([self.config.batch_size, self.config.padlen, 1])
        output, state = tf.nn.dynamic_rnn(cell_fw, dec_inputs, initial_state=initial_state)

        return output, state

    def build(self):
        ### Placeholders
        self.q1 = tf.placeholder(tf.int64, shape=[self.config.batch_size, None], name="question1")
        self.l1 = tf.placeholder(tf.int64, shape=[self.config.batch_size], name="len1")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        ### Embedding layer
        with tf.variable_scope("word_embeddings") as scope:
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            we1 = tf.nn.embedding_lookup(_word_embeddings, self.q1, name="q1_embedded")

            we1 = tf.nn.dropout(we1, keep_prob=1 - self.dropout)

        ### Shared layer
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.l1)

        state = tf.contrib.rnn.LSTMStateTuple(tf.concat([state1[0].c, state1[1].c], axis=1),
                                              tf.concat([state1[0].h, state1[1].h], axis=1))

        ### Decoder
        decoded, _ = self.decode(state)

        ### logits
        with tf.variable_scope("linear_projection") as scope:
            W = tf.Variable(tf.random_normal([1, 2 * self.config.hidden_size, self.vocab_size], stddev=1e-3), name="w")
            self.logits = tf.nn.conv1d(decoded, W, 1, "VALID", name="logits")

        ### Loss
        loss_mask = tf.sequence_mask(lengths=self.l1 - 1, maxlen=self.padlen - 1, dtype=tf.float32)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.q1, logits=self.logits))


        ### Optimizer
        with tf.variable_scope("train_step") as scope:
            if self.config.lr_method == "adam":
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.lr_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.lr)

            self.train_step = optimizer.minimize(self.cross_entropy)

        ### Evaluation
        correct_prediction = tf.equal(tf.argmax(self.logits, axis=-1), self.q1)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ### Init
        self.init = tf.global_variables_initializer()

        ### Summaries
        with tf.variable_scope("summaries") as scope:

            # train
            tf.summary.scalar('cross_entropy', self.cross_entropy)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()

            # test
            self.acc_value = tf.placeholder_with_default(tf.constant(0.0), shape=())
            self.ce_value = tf.placeholder_with_default(tf.constant(0.0), shape=())
            acc_summary = tf.summary.scalar('accuracy', self.acc_value)
            ce_summary = tf.summary.scalar('cross_entropy', self.ce_value)
            self.merged_eval = tf.summary.merge([acc_summary, ce_summary])

        self.saver = tf.train.Saver()

    def run_epoch(self, sess, train, dev, test, epoch, lr, test_step=1000):
        iterator = DataIterator(train, self.config.batch_size, strict=1)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size
        dev_acc = 0

        accuracy, cross = 0, 0
        for i in tqdm(range(nbatches)):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.l1: l1, self.dropout: self.config.dropout, self.lr: lr}
            _, summary = sess.run([self.train_step, self.merged], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.train_writer.add_summary(summary, epoch * nbatches + i)

            if i % test_step == 0:
                summary, dev_acc = self.run_evaluate(sess, dev)
                self.dev_writer.add_summary(summary, epoch * nbatches + i)
                print("Step {}/{}".format(i, nbatches))
                print("dev acc {:04.2f}".format(100 * dev_acc))

                summary, test_acc = self.run_evaluate(sess, test)
                self.test_writer.add_summary(summary, epoch * nbatches + i)
                print("test acc {:04.2f}".format(100 * test_acc))

        return dev_acc

    def run_evaluate(self, sess, data):
        iterator = DataIterator(data, self.config.batch_size, strict=1)
        nbatches = (iterator.max + self.config.batch_size - 1) // self.config.batch_size

        accuracy, cross = 0, 0
        for i in range(nbatches):
            q1, q2, l1, l2, y = iterator.__next__()
            fd = {self.q1: q1, self.l1: l1, self.dropout: 0, self.lr: 0}
            acc, ce = sess.run([self.accuracy, self.cross_entropy], feed_dict=fd)

            accuracy += acc * len(q1)
            cross += ce * len(q1)

        accuracy /= iterator.max
        cross /= iterator.max

        summary = sess.run(self.merged_eval, feed_dict={self.acc_value: accuracy, self.ce_value: cross})

        return summary, accuracy

    def train(self, train_data, dev_data, test_data):

        best_acc = 0
        nepoch_no_improv = 0

        with tf.Session(config=tf_config) as sess:
            self.train_writer = tf.summary.FileWriter(self.config.log_path + "autoencoder/" + "train", sess.graph)
            self.dev_writer = tf.summary.FileWriter(self.config.log_path + "autoencoder/" + "dev", sess.graph)
            self.test_writer = tf.summary.FileWriter(self.config.log_path + "autoencoder/" + "test", sess.graph)

            sess.run(self.init)

            lr = self.config.lr

            print("Training in {}".format(self.config.conf_dir))
            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{} :".format(epoch + 1, self.config.n_epochs))
                dev_acc = self.run_epoch(sess, train_data, dev_data, test_data, epoch, lr)

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