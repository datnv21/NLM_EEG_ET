import sys
sys.path.append("utils")

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding, LSTM, Bidirectional, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from constant import CHR_EMBEDDINGS


class NLMBasic:
    def __init__(self, vocab, character_embedding=CHR_EMBEDDINGS, n_timesteps=150, input_length=30):
        self.character_embedding = character_embedding
        self.n_timesteps = n_timesteps
        self.vocab = vocab
        self.input_length = input_length

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab, self.character_embedding, input_length=self.input_length, trainable=True))
        # model.add(GRU(self.n_timesteps, recurrent_dropout=0.1, dropout=0.1))
        model.add(Bidirectional(LSTM(self.n_timesteps)))
        model.add(Dropout(0.1))
        model.add(Dense(self.vocab, activation="softmax"))

        return model
