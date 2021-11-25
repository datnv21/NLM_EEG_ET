import os
import sys
sys.path.append("models")
sys.path.append("utils")
sys.path.append("data")

from model import NLMBasic
from preprocess import prepare
from keras.preprocessing.sequence import pad_sequences
from constant import SEQUENCE_LENGTH, CHR_EMBEDDINGS
import numpy as np


model_weight = "checkpoints/model.h5"


def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    suggestion = []
    in_text = seed_text
    
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += char
        
    return in_text


def inference():
    sequences_encoded, mapping, labels_encoded = prepare(use_et=False)
    vocab_size = len(mapping)
    nlm_basic = NLMBasic(vocab_size, input_length=SEQUENCE_LENGTH, character_embedding=CHR_EMBEDDINGS)
    model = nlm_basic.build_model()
    model.load_weights(model_weight, by_name=True, skip_mismatch=True)

    seed_text = "nâng ch"
    result = generate_seq(model, mapping, SEQUENCE_LENGTH, seed_text, len(mapping))
    print(result)

    return 1


if __name__=="__main__":
    inference()
