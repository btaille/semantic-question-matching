Semantic Question Matching
====

A Tensorflow implementation of a BiLSTM-MaxPooling Siamese Network [1] for Paraphrase Detection on the Quora question pairs dataset [2]. The encoder can be pretrained with a Sequential Denoising Autoencoder (SDAE)
to tackle the semi-supervised setting (as in [3]).

### Requirements
The code is written in Python 3 with the following dependencies:

* Tensorflow (== 1.4)
* NLTK
* Gensim
* tqdm

### Data
The provided split is the standard partition from [4] in the original Quora format.

**None of question_ids and ids match the original release from Quora.**

### Download GloVe

To get GloVE pretrained word embeddings, run (into data/):
```bash
./get_data.sh
```
This will download GloVE.6B.

### Training
Supervised Siamese Network:
```bash
python training.py -m siamese
```
Semi-supervised SDAE-Siamese Network:
```bash
# -r is the size of the labeled seed of question pairs

python training.py -m hybrid -r 1000
```

### References

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

[2] [Quora question pairs dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

[3] D. Shen, Y. Zhang, R. Henao, Q. Su, L. Carin, [*Deconvolutional Latent-Variable Model for Text Sequence Matching*](https://arxiv.org/abs/1709.07109)

[4] Z. Wang, W. Hamza, R.Florian, [*Bilateral Multi-Perspective Matching for Natural Language Sentences*](https://arxiv.org/abs/1702.03814)