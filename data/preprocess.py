import os
import sys
sys.path.append("utils")
import json
import re

from constant import VN_RE, SEQUENCE_LENGTH
from sklearn.utils import shuffle
import numpy as np


TEXT_PATH = "text_timestamp.json"
TEXT_SAMPLE = "clean_sentences_1.txt"
CHR_ET = "character_et.json"


def load_chr_et():
    with open(CHR_ET) as f:
        data = json.load(f)
    
    for k in data.keys():
        data[k] = np.array(data[k])
    return data


def create_corpus():
    with open(TEXT_PATH, "r") as f:
        data = json.load(f)
        f.close()
    
    with open(TEXT_SAMPLE, "r") as f:
        new_data = f.readlines()
        f.close()

    text = data["text"]
    text = shuffle(text, random_state=42)
    new_text = [t.replace('\n', '').strip() for t in new_data]
    # corpus = " ".join(text + new_text)
    corpus = text + new_text
    # corpus = corpus.lower()
    return corpus


def create_seq(corpus):
    length = SEQUENCE_LENGTH
    sequences = list()
    labels = list()

    for i in range(length, len(corpus)):
        seq = corpus[i-length:i]
        lbl = corpus[i]
        sequences.append(seq)
        labels.append(lbl)

    return sequences, labels


def prepare(use_et=True):
    corpus = create_corpus()
    corpus = " ".join(corpus)
    corpus = corpus.lower()
    chars = sorted(list(set(corpus)))
    mapping = dict((c,i) for i, c in enumerate(chars))
    sequences, labels = create_seq(corpus)

    if use_et:
        chr_et = load_chr_et()

    sequences_encoded = list()
    labels_encoded = [mapping[lbl] for lbl in labels]
    for line in sequences:
        encoded_seq = [mapping[char] for char in line]
        
        if use_et:
            et_seq = []
            for char in line:
                if char == " ": continue
                if char not in chr_et.keys(): continue
                et_seq.append(np.mean(chr_et[char], axis=0, dtype=np.float32))

            et_seq = np.mean(et_seq, axis=0)
            encoded_seq.extend(et_seq)
        sequences_encoded.append(encoded_seq)
    
    return sequences_encoded, mapping, labels_encoded

if __name__=="__main__":
    prepare()
