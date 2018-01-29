import numpy as np
import argparse
import os

from model import SiameseNet
from data_utils import QuoraDataset
from embeddings import load_embeddings

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-ep", "--epochs", type=int, help="number of epochs", default=10)
parser.add_argument("-es", "--early_stopping", type=int, help="number of epochs without improvement max", default=3)
parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=64)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-3)
parser.add_argument("-lrd", "--lr_decay", type=float, help="learning rate decay", default=0.9)
parser.add_argument("-opt", "--optimizer", help="optimizer", default="adam")
parser.add_argument("-bn", "--batch_norm", type=int, help="batch norm", default=1)
parser.add_argument("-f", "--feats", help="features", default="all")
parser.add_argument("-hid", "--hidden", type=int, help="dimension of lstm hidden layer", default=256)
parser.add_argument("-w2v", "--w2v_dim", type=int, help="dimension of pretrained word embeddings", default=300)
parser.add_argument("-d", "--dropout", type=float, help="dropout", default=0.)
parser.add_argument("-emb", "--train_emb", type=int, help="fine_tune word embeddings", default=0)
parser.add_argument("-act", "--activation", help="activation function of FC layers", default="relu")
parser.add_argument("-pad", "--padlen", type=int, help="padding length", default=40)

args = parser.parse_args()


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

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

    # vocab
    # TODO saving and quick reloading of dicts and formatted embeddings ???

    # training
    train_embeddings = args.train_emb
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

    lr_divide = 1
    reload = False

    # hyperparameters
    padlen = args.padlen
    hidden_size = args.hidden

    assert lr_method in ["adam", "sgd"]
    assert fc_activation in ["relu", "tanh", "sigmoid"]
    assert feats in ["raw", "dist", "all"]

    conf_dir = "hid-{}_feats-{}_lr-{}-{}-{}_bs-{}_drop-{}_bn-{}_emb-{}/".format(hidden_size, feats, lr_method, lr,
                                                                                fc_activation, batch_size, dropout,
                                                                                int(batch_norm),
                                                                                int(train_embeddings))

    # general config
    output_path = "results/" + conf_dir
    model_path = output_path + "model/"
    log_path = output_path + "logs/"


if __name__ == "__main__":
    ### Loading config and pretrained Glove embeddings
    config = Config()
    loaded_embeddings, (w2idx, idx2w) = load_embeddings(config.glove_filename, binary=False)

    ### Loading Quora Datasets
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

    ### SiameseNet
    model = SiameseNet(config, embeddings)
    model.build()

    model.train(train_data, dev_data, test_data)
