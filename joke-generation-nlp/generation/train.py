import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, Activation
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from generation.generate import generate_once


def load_dataset(file, max_sequence_len=20):
    df = pd.read_csv(file)

    sentences = df.body.dropna().drop_duplicates()
    sentences_words = list(sentences.apply(lambda x: word_tokenize((str(x).lower()))))

    word_counts = {}
    for sent in sentences_words:
        for word in sent:
            try:
                word_counts[word] += 1
            except KeyError:
                word_counts[word] = 1

    rare_words = [word for word, count in word_counts.items() if count < 2]
    sentences_words = [seq for seq in sentences_words
                       if len(seq) > max_sequence_len and set(seq).isdisjoint(rare_words)]

    word2index = {}
    index2word = {}
    for seq in sentences_words:
        for word in seq:
            if word not in word2index:
                i = len(word2index)
                word2index[word] = i
                index2word[i] = word

    input_sequences = []
    for sent in sentences_words:
        sent_tokens = [word2index[word] for word in sent]
        for i in range(max_sequence_len, len(sent_tokens) + 1):
            input_sequences.append(sent_tokens[i - max_sequence_len:i])
    input_sequences = np.array(input_sequences)

    X, Y = input_sequences[:, :-1], to_categorical(input_sequences[:, -1], len(word2index))

    return X, Y, word2index, index2word


if __name__ == '__main__':
    max_len = 25
    X, Y, word2index, index2word = load_dataset('../data/wocka.csv', max_sequence_len=max_len)

    model = Sequential()
    model.add(Embedding(len(word2index), 200, input_length=max_len - 1))
    model.add(Bidirectional(LSTM(500, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(500)))
    model.add(Dropout(0.2))
    model.add(Dense(len(word2index)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy')
    model.summary()

    epochs = 23
    model.fit(X, Y, epochs=epochs, batch_size=800)

    pickle.dump(word2index, open('word2index.pickle', 'wb'))
    pickle.dump(index2word, open('index2word.pickle', 'wb'))
    model.save('epochs' + str(epochs) + 'seq' + str(max_len) + 'model.h5')

    seed_text = 'The man in the white suit lowered his hat. "Why do you keep looking at me like that?", he asked.'
    print(generate_once(model, word2index, index2word, seed_text, p=0.003))
