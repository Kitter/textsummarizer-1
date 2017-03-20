import source.wordsummarizer.utils
import os

from source.wordsummarizer import utils

embeddings = {}
with open('../../resources/glove/glove.6B.50d.txt') as fp:
    for line in fp:
        tokens = line.split()
        word = tokens[0]
        weights = tokens[1:]
        embeddings = word, weights
        print(word + ": %s" % (weights))

utils.build_model(embeddings)