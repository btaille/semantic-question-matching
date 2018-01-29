import os


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

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

    # vocab
    # TODO saving and quick reloading of dicts and formatted embeddings ???

    # training
    train_embeddings = False
    n_epochs = 10
    dropout = 0.
    batch_size = 16
    batch_norm = True
    lr_method = "adam"
    fc_activation = "relu"
    feats = "dist"
    lr = 0.001
    lr_decay = 0.9
    lr_divide = 1
    reload = False
    nepochs_no_improv = 3

    # hyperparameters
    padlen = 40
    hidden_size = 128

    assert lr_method in ["adam", "sgd"]
    assert fc_activation in ["relu", "tanh", "sigmoid"]
    assert feats in ["raw", "dist", "all"]

    conf_dir = "hid-{}_feats-{}_lr-{}-{}-{}_bs-{}_drop-{}_bn-{}_emb-{}_padlen-{}/quora/".format(hidden_size, feats,
                                                                                                lr_method, lr,
                                                                                                fc_activation,
                                                                                                batch_size, dropout,
                                                                                                int(batch_norm),
                                                                                                int(train_embeddings),
                                                                                                padlen)

    # general config
    output_path = "results/" + conf_dir
    model_path = output_path + "model/model.ckpt"
    log_path = output_path + "logs/"
