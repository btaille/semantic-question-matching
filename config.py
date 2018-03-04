import os
from src.embeddings import convert_glove

class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.glove_filename):
            for file in os.listdir(os.path.dirname(self.glove_filename)):
                convert_glove(os.join(os.path.dirname(self.glove_filename), file))

    # model
    model_name = "siamese"

    # embeddings
    we_dim = 300
    glove_filename = "word_embeddings/glove.6B/glove.6B.{}d_w2vformat.txt".format(we_dim)

    # data
    train_filename = "data/train.csv"
    dev_filename = "data/dev.csv"
    test_filename = "data/test.csv"
    train_save = "data/train.pkl"
    dev_save = "data/dev.pkl"
    test_save = "data/test.pkl"

    # training
    train_embeddings = False
    n_epochs = 10
    dropout = 0.
    batch_size = 32
    batch_norm = True
    lr_method = "adam"
    fc_activation = "relu"
    feats = "dist"
    lr = 0.001
    lr_decay = 0.9
    lr_divide = 1
    reload = False
    nepochs_no_improv = 3
    test_step = 6000

    # hybrid
    task = "successive"
    corruption = True
    ratio = 1
    restrict = 0

    # hyperparameters
    padlen = 40
    hidden_size = 256

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
