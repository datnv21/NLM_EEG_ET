from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


class NLMBasic:
    def __init__(self, vocab, character_embedding=50, n_timesteps=150):
        self.character_embedding = character_embedding
        self.n_timesteps = n_timesteps
        self.vocab = vocab

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab, self.character_embedding, input_length=30, trainable=True))
        model.add(GRU(self.n_timesteps, recurrent_dropout=0.1, dropout=0.1))
        model.add(Dense(self.vocab, activation="softmax"))

        return model
