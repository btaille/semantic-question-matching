import numpy as np
import pandas as pd
import pickle
import os

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

from embeddings import word_index, reverse_dict


def check_str(s):
    if type(s) == str:
        return s
    else:
        return "NaN"


def lower_list(l):
    return ([elt.lower() for elt in l])


def tokenize_dict(q_dict, lower=True, stemmer=None):
    if stemmer == "english":
        snowball = SnowballStemmer("english")
        return {k: lower_list([snowball.stem(token) for token in word_tokenize(q_dict[k])]) if lower
        else [snowball.stem(token) for token in word_tokenize(q_dict[k])]
                for k in q_dict.keys()}
    else:
        return {k: lower_list(word_tokenize(q_dict[k])) if lower
        else word_tokenize(q_dict[k])
                for k in q_dict.keys()}


def sent2ids(sent, w2idx):
    return [w2idx[w] if w in w2idx.keys() else w2idx["<UNK>"] for w in sent]


def ids2sent(ids, idx2w):
    return [idx2w[i] for i in ids]


def pad_sequence(ids, padlen, pad_tok=0):
    return ids[:padlen] + [pad_tok] * max(padlen - len(ids), 0)


def sequence_dict(tok_dict, w2idx):
    seq_dict = {k: sent2ids(tok_dict[k], w2idx) for k in tok_dict.keys()}
    return seq_dict


def corrupt_sequence(seq, p=0.1, d=3):
    l = len(seq)
    mask = np.greater(np.random.random(l), [p] * l).astype(int)
    seq = mask * seq

    return np.array(seq)[sorted(range(l), key=lambda i: i + d * np.random.random())]


def corrupt_sequences(sequences, p=0.1, d=3):
    corrupted = np.zeros(sequences.shape, int)

    print(corrupted.shape)

    for i, seq in enumerate(sequences):
        corrupted[i] = corrupt_sequence(seq, p=p, d=d)

    return corrupted


class DataIterator(object):
    def __init__(self, data, batch=1, strict=0, restrict=0, ratio=1):
        self.q1, self.q2, self.l1, self.l2, self.y = data
        self.batch = batch

        if restrict:
            np.random.seed(0)
            ids = np.random.randint(len(self.q1), size=restrict)
            self.q1, self.q2 = self.q1[ids], self.q2[ids]
            self.l1, self.l2 = np.array(self.l1)[ids], np.array(self.l2)[ids]
            self.y = self.y[ids]

        if ratio:
            self.q1 = np.repeat(self.q1, ratio, axis=0)
            self.q2 = np.repeat(self.q2, ratio, axis=0)
            self.l1 = np.repeat(self.l1, ratio, axis=0)
            self.l2 = np.repeat(self.l2, ratio, axis=0)
            self.y = np.repeat(self.y, ratio, axis=0)

        self.i = 0
        self.max = len(self.q1)
        if strict:
            self.max -= self.max % self.batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            ranged = (self.i, min(self.i + self.batch, self.max))
            self.i += self.batch
            return self.q1[ranged[0]:ranged[1]], self.q2[ranged[0]:ranged[1]], self.l1[ranged[0]:ranged[1]], \
                   self.l2[ranged[0]:ranged[1]], self.y[ranged[0]:ranged[1]]
        raise StopIteration


class DataIteratorAE(object):
    def __init__(self, data, batch=1, strict=0):
        self.q1, self.q2, self.l1, self.l2, self.y = data
        self.batch = batch
        self.i = 0

        q1, ids1 = np.unique(self.q1, return_index=True, axis=0)
        q2, ids2 = np.unique(self.q2, return_index=True, axis=0)

        self.q = np.concatenate([q1, q2], axis=0)
        self.l = np.concatenate([np.array(self.l1)[ids1], np.array(self.l2)[ids2]], axis=0)

        self.max = len(self.q)
        if strict:
            self.max -= self.max % self.batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            ranged = (self.i, min(self.i + self.batch, self.max))
            self.i += self.batch
            return self.q[ranged[0]:ranged[1]], self.l[ranged[0]:ranged[1]]
        raise StopIteration


class QuoraDataset(object):
    def __init__(self, filename, sep=',', w2idx=None, save_path=None):
        self.filename = filename
        self.w2idx = w2idx
        self.sep = sep
        self.save_path = save_path

        if self.save_path is not None and os.path.exists(self.save_path):
            self.reload()
        else:
            self.build()
            if self.save_path is not None:
                self.save()

    def save(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.__dict__, file)

    def reload(self):
        with open(self.save_path, "rb") as file:
            tmp_dict = pickle.load(file)

        self.__dict__.update(tmp_dict)

    def build(self):
        self.df = pd.read_csv(self.filename, sep=self.sep)

        print("Building question dictionary...")
        self.q_dict = {}

        for _, row in self.df.iterrows():
            i, qid1, q1 = row["id"], row["qid1"], row["question1"]
            if qid1 not in self.q_dict:
                self.q_dict[qid1] = check_str(q1)

        for _, row in self.df.iterrows():
            i, qid2, q2 = row["id"], row["qid2"], row["question2"]
            if qid2 not in self.q_dict:
                self.q_dict[qid2] = check_str(q2)

        print("Tokenizing questions...")
        self.tok_dict = tokenize_dict(self.q_dict)

        if self.w2idx is None:
            self.w2idx, self.idx2w = word_index(self.tok_dict.values())
        else:
            self.idx2w = reverse_dict(self.w2idx)

        self.seq_dict = sequence_dict(self.tok_dict, self.w2idx)

        self.len = len(self.df["qid1"])

        self.l1 = [len(self.seq_dict[np.array(self.df["qid1"])[k]]) for k in range(self.len)]
        self.l2 = [len(self.seq_dict[np.array(self.df["qid2"])[k]]) for k in range(self.len)]

        self.maxlen = max([max(self.l1), max(self.l2)])

        self.q1 = np.array(
            [pad_sequence(self.seq_dict[np.array(self.df["qid1"])[k]], self.maxlen) for k in range(self.len)])
        self.q2 = np.array(
            [pad_sequence(self.seq_dict[np.array(self.df["qid2"])[k]], self.maxlen) for k in range(self.len)])

        self.y = np.array(self.df["is_duplicate"])

        print("Done")

    def data(self, padlen=40, len=None):
        if len is not None:
            return self.q1[:len, :padlen], self.q2[:len, :padlen], self.l1[:len], self.l2[:len], self.y[:len]
        else:
            return self.q1[:, :padlen], self.q2[:, :padlen], self.l1, self.l2, self.y
