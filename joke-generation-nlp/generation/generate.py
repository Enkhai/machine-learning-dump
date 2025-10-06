import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def batch_generate(model, word2index, index2word, seed_df):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))
    if isinstance(seed_df, str):
        seed_df = pd.read_csv(seed_df)
    max_len = model.input_shape[1]

    output_df = pd.DataFrame(columns=seed_df.columns.tolist() + ['output'])
    for index, row in seed_df.iterrows():
        seed_indices = [word2index[word] for word in word_tokenize(row['seed'].lower())]
        next_words = row['next_words']
        if row['method'] == 'greedy':
            sentence = greedy_decode(model, seed_indices, max_len, next_words=next_words)
        elif row['method'] == 'beam search':
            sentences = beam_search_decode(model, seed_indices, max_len, next_words, int(row['k']))
            sentence = sentences[0][0]
        elif row['method'] == 'sample':
            sentence = top_sampling(model, seed_indices, max_len, next_words, row['p'], int(row['k']), row['temp'])
        row['output'] = ' '.join([index2word[token] for token in sentence])
        output_df = output_df.append(row)

    output_df.to_csv('GeneratedJokes.csv')


def generate_once(model, word2index, index2word, seed_text, next_words=100, method='sample', k=0, p=0.0, temp=1.0):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))

    max_len = model.input_shape[1]
    seed_indices = [word2index[word] for word in word_tokenize(seed_text.lower())]
    if method == 'greedy':
        sentence = greedy_decode(model, seed_indices, max_len, next_words=next_words)
    elif method == 'beam search':
        sentences = beam_search_decode(model, seed_indices, max_len, next_words, k)
        sentence = sentences[0][0]
    elif method == 'sample':
        sentence = top_sampling(model, seed_indices, max_len, next_words, p, k, temp)

    return ' '.join([index2word[token] for token in sentence])


def greedy_decode(model, seed, max_sequence_length, next_words):
    for _ in range(next_words):
        seed.append(int(np.argmax(model.predict([seed[-max_sequence_length:]]))))
    return seed


def beam_search_decode(model, seed, max_sequence_length, next_words, k=8):
    sequences = [[seed, 0.0]]
    for _ in range(next_words):
        top_k_indices = []
        top_k_probas = []
        for seq in sequences:
            pred = model.predict([seq[0][-max_sequence_length:]])[0]
            top_k_idx = pred.argsort()[-k:][::-1]
            top_k_prob = pred[top_k_idx]

            top_k_indices.append(top_k_idx)
            top_k_probas.append(top_k_prob)

        all_candidates = []
        for (tokens, score), top_k_idx, top_k_prob in zip(sequences, top_k_indices, top_k_probas):
            for idx, prob in zip(top_k_idx, top_k_prob):
                all_candidates.append([tokens + [int(idx)], score - np.log(prob)])
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:k]
    return sequences


def top_sampling(model, seed, max_sequence_length, next_words, p=0.0, k=0, temperature=1.0):
    for _ in range(next_words):
        pred = model.predict([seed[-max_sequence_length:]])[0]
        seed.append(int(sample(pred, temperature, p, k)))
    return seed


def sample(preds, temperature=1.0, p=0.0, k=0):
    if k > 0:
        indices_to_remove = preds.argsort()[:-k][::-1]
        preds[indices_to_remove] = 0
    elif p > 0.0:
        indices_to_remove = np.argwhere(preds < p)
        preds[indices_to_remove] = 0
    # ignore divide by zero warning
    preds = np.log(preds) / temperature
    # addresses multinomial casting issue
    # https://github.com/numpy/numpy/issues/8317
    preds = preds.astype('float64')
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    model = 'epochs23seq25model.h5'
    word2index = 'word2index.pickle'
    index2word = 'index2word.pickle'

    # ignore input incompatible shape
    # seed_text = 'The man in the white suit lowered his hat. "Why do you keep looking at me like that?", he asked.'
    # seed_text = 'Three children were playing in the park. One of them got up and looked at the other two.'
    # seed_text = "Once upon a time two guys were playing around with wires. Suddenly, one of them gets shocked."
    # print(generateOnce(model, word2index, index2word, seed_text, p=0.003, temp=0.4))

    batch_generate(model, word2index, index2word, 'seed.csv')
