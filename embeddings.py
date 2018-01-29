import numpy as np
import json
import os
from gensim.models import KeyedVectors

path = "word_embeddings/glove.6B/"


def load_embeddings(embeddings_path, binary=False):
    """ Load word embeddings"""
    assert os.path.isfile(embeddings_path)
    saving_path = embeddings_path + "_embeddings.p"
    vocab_path = embeddings_path + "_vocab.p"

    if not os.path.exists(saving_path):
        print("Loading word word_embeddings from %s" % embeddings_path)
        w2v = KeyedVectors.load_word2vec_format(embeddings_path, binary=binary)

        weights = w2v.syn0
        np.save(open(saving_path, 'wb'), weights)

        vocab = dict([(k, v.index) for k, v in w2v.vocab.items()])
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(vocab))
        print("Created word word_embeddings from %s" % embeddings_path)

    else:
        print("Loading from saved word_embeddings")
        with open(saving_path, 'rb') as f:
            weights = np.load(f)

    vocab = load_vocab(embeddings_path)

    return weights, vocab


def load_vocab(embeddings_path):
    """ Load vocabulary mappings corresponding to an embedding"""
    print("Loading vocab")
    vocab_path = embeddings_path + "_vocab.p"

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def convert_glove(embeddings_path):
    """ Creates a new Word2Vec formatted embedding file from a Glove formatted one """
    assert os.path.isfile(embeddings_path)
    os.system("python3 -m gensim.scripts.glove2word2vec --input  %s --output %s_w2vformat.txt"
              % (embeddings_path, embeddings_path.split(".txt")[0]))


def word_index(sents, lower=False, add_unknown=True):
    """
    Create mappings between words and indices of vocabulary
    :param sents: list of tokenized sentences as lists of words
    :param lower: if True capitalization is not taken into account
    :param add_unknown: if True adds <UNK> token at index 0
    :return: w2idx and idx2w mapping dictionaries
    """
    w2idx = {}
    if add_unknown:
        w2idx["<UNK>"] = 0
    for s in sents:
        for w in s:
            if lower:
                w = w.lower()
            if w not in w2idx:
                w2idx[w] = len(w2idx)
    idx2w = reverse_dict(w2idx)
    return w2idx, idx2w


def reverse_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


if __name__ == "__main__":
    for file in os.listdir(path):
        convert_glove(path + file)
