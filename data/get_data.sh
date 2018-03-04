#!/usr/bin/env bash

glovepath='http://nlp.stanford.edu/data/glove.6B.zip'

# Download GloVe
mkdir ../word_embeddings/glove.6B
curl -LO $glovepath
unzip glove.6B.zip -d ../word_embeddings/glove.6B/
rm glove.6B.zip
