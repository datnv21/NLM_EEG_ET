import os
import sys
sys.path.append("data")
sys.path.append("models")

from model import NLMBasic
from preprocess import prepare
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical


def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def main():
    checkpoint_dir = "checkpoints"
    makedir(checkpoint_dir)

    sequences_encoded, mapping = prepare()
    vocab_size = len(mapping)
    sequences_encoded = np.array(sequences_encoded)

    X, y = sequences_encoded[:, :-1], sequences_encoded[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    nlm_basic = NLMBasic(vocab_size)
    model = nlm_basic.build_model()
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    model.fit(X_train, y_train, epochs=70, verbose=2, validation_data=(X_val, y_val))
    model.save(os.path.join(checkpoint_dir, "model.h5"))

    return 1

if __name__=="__main__":
    main()
