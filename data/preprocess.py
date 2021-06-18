import os
import sys
sys.path.append("utils")
import json

from constant import VN_RE, SEQUENCE_LENGTH
from sklearn.utils import shuffle


TEXT_PATH = "text_timestamp.json"


def create_corpus():
    with open(TEXT_PATH, "r") as f:
        data = json.load(f)
        f.close()

    text = data["text"]
    # text = shuffle(text, random_state=42)
    corpus = " ".join(text)
    corpus = corpus.lower()
    
    return corpus


def create_seq(corpus):
    length = SEQUENCE_LENGTH
    sequences = list()

    for i in range(length, len(corpus)):
        seq = corpus[i-length:i+1]
        sequences.append(seq)

    return sequences


def prepare():
    corpus = create_corpus()
    chars = sorted(list(set(corpus)))
    mapping = dict((c,i) for i, c in enumerate(chars))
    sequences = create_seq(corpus)

    sequences_encoded = list()
    for line in sequences:
        encoded_seq = [mapping[char] for char in line]
        sequences_encoded.append(encoded_seq)

    return sequences_encoded, mapping

if __name__=="__main__":
    prepare()
