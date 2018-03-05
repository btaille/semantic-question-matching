import argparse
import os

import numpy as np
from src.data_utils import QuoraDataset, corrupt_sequences
from src.embeddings import load_embeddings, convert_glove

from src.model import Model

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model", default="siamese")
parser.add_argument("-res", "--restrict", type=int, help="size of supervised seed", default=0)
parser.add_argument("-tr", "--training", help="training method of hybrid network", default="successive")
parser.add_argument("-ra", "--ratio", type=int, help="repetition of seed to balance joint training", default=1)
parser.add_argument("-ep", "--epochs", type=int, help="number of epochs", default=10)
parser.add_argument("-es", "--early_stopping", type=int, help="number of epochs without improvement max", default=3)
parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=32)
parser.add_argument("-ts", "--test_step", type=int, help="test step", default=1000)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-3)
parser.add_argument("-lrd", "--lr_decay", type=float, help="learning rate decay", default=0.9)
parser.add_argument("-opt", "--optimizer", help="optimizer", default="adam")
parser.add_argument("-bn", "--batch_norm", type=int, help="batch norm", default=1)
parser.add_argument("-f", "--feats", help="features", default="dist")
parser.add_argument("-hid", "--hidden", type=int, help="dimension of lstm hidden layer", default=256)
parser.add_argument("-w2v", "--w2v_dim", type=int, help="dimension of pretrained word embeddings", default=300)
parser.add_argument("-d", "--dropout", type=float, help="dropout", default=0.)
parser.add_argument("-emb", "--train_emb", type=int, help="fine_tune word embeddings", default=0)
parser.add_argument("-act", "--activation", help="activation function of FC layers", default="relu")
parser.add_argument("-pad", "--padlen", type=int, help="padding length", default=40)
parser.add_argument("-cor", "--corruption", type=int, help="use Denoising AE", default=1)

args = parser.parse_args()


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.glove_filename):
            for file in os.listdir(os.path.dirname(self.glove_filename)):
                convert_glove(os.path.join(os.path.dirname(self.glove_filename), file))

    # model
    model_name = args.model
    restrict = args.restrict
    ratio = args.ratio

    # embeddings
    we_dim = args.w2v_dim
    glove_filename = "word_embeddings/glove.6B/glove.6B.{}d_w2vformat.txt".format(we_dim)

    # data
    train_filename = "data/train.csv"
    dev_filename = "data/dev.csv"
    test_filename = "data/test.csv"
    train_save = "data/train.pkl"
    dev_save = "data/dev.pkl"
    test_save = "data/test.pkl"

    # training
    train_embeddings = args.train_emb
    task = args.training
    n_epochs = args.epochs
    dropout = args.dropout
    batch_size = args.batch_size
    batch_norm = args.batch_norm
    lr_method = args.optimizer
    feats = args.feats
    fc_activation = args.activation
    lr = args.learning_rate
    lr_decay = args.lr_decay
    nepochs_no_improv = args.early_stopping
    test_step = args.test_step
    corruption = args.corruption

    lr_divide = 1
    reload = False

    # hyperparameters
    padlen = args.padlen
    hidden_size = args.hidden

    assert lr_method in ["adam", "sgd"]
    assert fc_activation in ["relu", "tanh", "sigmoid"]
    assert feats in ["raw", "dist", "all"]
    assert model_name in ["siamese", "hybrid"]
    assert task in ["autoencoder", "inference", "joint", "successive"]

    if model_name == "siamese":
        conf_dir = "{}-hid-{}_feats-{}_lr-{}-{}-{}_bs-{}_drop-{}_bn-{}_emb-{}".format(model_name, hidden_size,
                                                                                      feats,
                                                                                      lr_method, lr,
                                                                                      fc_activation, batch_size,
                                                                                      dropout,
                                                                                      batch_norm,
                                                                                      train_embeddings)

    elif model_name == "hybrid":
        conf_dir = "{}-hid-{}_feats-{}_lr-{}-{}-{}_bs-{}_drop-{}_bn-{}_emb-{}_res-{}_tr-{}_corr-{}/".format(model_name,
                                                                                                            hidden_size,
                                                                                                            feats,
                                                                                                            lr_method,
                                                                                                            lr,
                                                                                                            fc_activation,
                                                                                                            batch_size,
                                                                                                            dropout,
                                                                                                            batch_norm,
                                                                                                            train_embeddings,
                                                                                                            restrict,
                                                                                                            task,
                                                                                                            corruption)

    # general config
    output_path = "results/" + conf_dir
    model_path = output_path + "model/"
    log_path = output_path + "logs/"


if __name__ == "__main__":
    # Loading config and pretrained Glove embeddings
    config = Config()
    loaded_embeddings, (w2idx, idx2w) = load_embeddings(config.glove_filename, binary=False)

    # Loading Quora Dataset
    qd_train = QuoraDataset(config.train_filename, save_path=config.train_save)
    w2idx_train, idx2w_train = qd_train.w2idx, qd_train.idx2w

    embeddings = np.random.normal(scale=0.001, size=(len(w2idx_train), config.we_dim))

    for w, i in w2idx_train.items():
        idx = w2idx.get(w)
        if idx is not None:
            embeddings[i] = loaded_embeddings[idx]

    qd_dev = QuoraDataset(config.dev_filename, w2idx=w2idx_train, save_path=config.dev_save)
    qd_test = QuoraDataset(config.test_filename, w2idx=w2idx_train, save_path=config.test_save)

    train_data = qd_train.data()
    dev_data = qd_dev.data()
    test_data = qd_test.data()

    # Training Neural network
    model = Model(config, embeddings)
    model.build()

    if config.model_name == "siamese":
        model.train(train_data, dev_data, test_data, task="inference")

    elif config.model_name == "hybrid":
        if config.corruption:
            corrupt = corrupt_sequences
        else:
            corrupt = lambda x: x

        if config.task in ["autoencoder", "inference"]:
            model.train(train_data, dev_data, test_data, restrict=config.restrict, task=config.task, ratio=config.ratio,
                        corrupt=corrupt)

        elif config.task == "successive":
            model.train(train_data, dev_data, test_data, restrict=config.restrict, task="autoencoder",
                        ratio=config.ratio, corrupt=corrupt)
            model.train(train_data, dev_data, test_data, restrict=config.restrict, task="inference",
                        ratio=config.ratio)
        elif config.task == "joint":
            model.train_mixed(train_data, dev_data, test_data, restrict=config.restrict, ratio=config.ratio)
