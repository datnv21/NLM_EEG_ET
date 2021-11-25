import os
import sys
sys.path.append("data")

from preprocess import create_corpus
from nltk import bigrams, trigrams
from collections import defaultdict, Counter 


class BigramModel:
    def __init__(self):
        self.corpus = create_corpus()
        super(BigramModel, self).__init__()
    
    def bigram_wordcount(self):
        model = defaultdict(lambda: defaultdict(lambda: 0))

        for sent in self.corpus:
            for w1, w2 in bigrams(sent.split(" "), pad_right=True, pad_left=True):
                model[(w1)][w2] += 1
        
        for w1 in model:
            total = sum(model[w1].values())

            for w2 in model[w1]:
                model[w1][w2] /= total
        
        return model
