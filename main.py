import os
import sys
import logging
import datetime as dt
from typing import Any, List, Tuple
# from collections import Counter

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Flatten, Dense, TimeDistributed, Bidirectional, Permute, Reshape, RepeatVector, Lambda
from keras.layers import GRU
from keras.layers import merge
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras.regularizers import l1, l2
from keras import backend as K
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
from keras.regularizers import l2
from keras.layers import multiply

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.utils import plot_model

np.random.seed(0)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(sys.version_info)
logger.info("Keras version {}".format(keras.__version__))
logger.info("Tensorflow version {}".format(tf.__version__))
logger.info(device_lib.list_local_devices())
logger.info(K.tensorflow_backend._get_available_gpus())


FILEDIR = "/Users/kylematoba/Documents/matoba/comsw4995/hw2"


def accuracy(y_true, y_pred):
    # The custom accuracy metric used for this task
    y = tf.argmax(y_true, axis =-1)
    y_ = tf.argmax(y_pred, axis =-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


def np_accuracy(y_true: np.ndarray,
                y_pred: np.ndarray) -> float:
    y = np.argmax(y_true, axis=-1)
    y_ = np.argmax(y_pred, axis=-1)
    mask = (y > 0)
    return np.mean(y_[mask] == y[mask])


def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s


def str_results(y_, revsere_decoder_index) -> str:
    # print("input     : " + str(x))
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    return str(onehot_to_seq(y_, revsere_decoder_index).upper())


def seq2ngrams(seqs: np.ndarray,
               n: int) -> np.ndarray:
    # Computes and returns the n-grams of a particular sequence, defaults to trigrams
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])


def _chunk_str(to_chunk: str,
               chunk_len: int) -> List[str]:
    remaining_to_chunk = to_chunk
    chunked = []
    while len(remaining_to_chunk) > chunk_len:
        chunked += [remaining_to_chunk[:chunk_len]]
        remaining_to_chunk = remaining_to_chunk[chunk_len:]

    chunked += [remaining_to_chunk]
    assert "".join(chunked) == to_chunk
    assert len(chunked) == np.ceil(len(to_chunk) / chunk_len)
    return chunked


def _chunk_the_dataset(train_df: pd.DataFrame,
                       chunk_size: int) -> pd.DataFrame:
    inputs = []
    expecteds = []
    ids = []
    lens = []

    for idx, row in train_df.iterrows():
        # idx = 3; row = train_df.iloc[idx, :]
        # idx = 6551; row = train_df.iloc[idx, :]
        input = row['input']
        expected = row['expected']
        to_append_inputs = _chunk_str(input, chunk_size)
        to_append_expecteds = _chunk_str(expected, chunk_size)
        num = len(to_append_expecteds)
        to_append_ids = [row['id']] * num
        to_append_lens = [len(x) for x in to_append_expecteds]

        inputs += to_append_inputs
        expecteds += to_append_expecteds
        ids += to_append_ids
        lens += to_append_lens

    ids_pd = pd.Series(ids)
    lens_pd = pd.Series(lens)
    inputs_pd = pd.Series(inputs)
    expecteds_pd = pd.Series(expecteds)

    train_df_chunked = pd.concat((ids_pd, lens_pd, inputs_pd, expecteds_pd),
                                 axis=1,
                                 keys=train_df.columns)
    return train_df_chunked


def _load_train_data(maxlen_seq: int) -> Tuple[np.ndarray, np.ndarray]:
    train_filename = os.path.join(FILEDIR, 'train.csv')
    train_df = pd.read_csv(train_filename)
    use_chunked_dataset = False
    # drop_duplicates = True
    drop_duplicates = False

    # use_chunked_dataset = True
    if use_chunked_dataset:
        train_df  = _chunk_the_dataset(train_df, maxlen_seq)
    if drop_duplicates:
        train_df_cleaned = train_df.drop_duplicates()
    else:
        train_df_cleaned = train_df

    use_rows = (train_df_cleaned['len'] <= maxlen_seq)

    use_train_df_cleaned = train_df_cleaned.loc[use_rows, ]
    train_input_seqs = use_train_df_cleaned.loc[:, 'input'].values.T
    train_target_seqs = use_train_df_cleaned.loc[:, 'expected'].values.T
    return train_input_seqs, train_target_seqs


def _load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    test_filename = os.path.join(FILEDIR, 'test.csv')
    test_df = pd.read_csv(test_filename)
    test_input_seqs = test_df['input'].values.T
    ids = test_df['id'].values
    return test_input_seqs, ids


def _build_model(maxlen_seq: int,
                 n_words: int,
                 n_tags: int) -> Any:
    # https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/

    # https://keras.io/optimizers/
    # embedding_dim = 64
    embedding_dim = 128
    # embedding_dim = 256  # <- this improves by about 1%

    # lstm_units = 96
    lstm_units = 64
    # attention_units = 32
    # lstm_units = 32
    # lstm_units = 48
    recurrent_dropout = .1
    # https://danijar.com/tips-for-training-recurrent-neural-networks/
    # dropout = 0.0
    dropout = 0.10    # this improves validation accuracy by about 1%
    activation = 'tanh'  # tanh seems to be best
    # activation = 'sigmoid'
    # optimizer = "rmsprop"
    # optimizer = "adadelta"
    # optimizer = "adam"
    # optimizer = "adamax"
    # Much like Adam is essentially RMSprop with momentum, Nadam
    # is Adam RMSprop with Nesterov momentum.
    optimizer = "nadam"  # <- use this!
    # optimizer = keras.optimizers.Nadam(clipvalue=0.5)
    # optimizer = keras.optimizers.Nadam()
    embeddings_regularizer = None
    # embeddings_regularizer = l2(1e-4)
    # lstm_class = LSTM
    lstm_class = GRU  # <- use this

    # https://github.com/datalogue/keras-attention/issues/22
    input = Input(shape=(maxlen_seq, ))
    # Defining an embedding layer mapping from the words (n_words) to
    # a vector of len 128
    x = Embedding(input_dim=n_words,
                  output_dim=embedding_dim,
                  # input_length=maxlen_seq,
                  embeddings_regularizer=embeddings_regularizer)(input)
    # Defining a bidirectional LSTM using the embedded representation of the inputs

    ltsm = lstm_class(units=lstm_units,
                      activation=activation,
                      return_sequences=True,
                      dropout=dropout,
                      recurrent_dropout=recurrent_dropout)

    # ltsm1 = lstm_class(units=int(lstm_units/2), activation=activation, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    # ltsm2 = lstm_class(units=int(lstm_units/2), activation=activation, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)

    # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BahdanauAttention
    # x = Bidirectional(ltsm1)(x)
    # x = Bidirectional(ltsm2)(x)
    x = Bidirectional(ltsm)(x)

    # A dense layer to output from the LSTM's 64 units to the appropriate number of
    # tags to be fed into the decoder
    # y = TimeDistributed(Dense(units=n_tags, activation="softmax"))(x)
    y = Dense(n_tags, activation="softmax")(x)

    # Defining the model as a whole and printing the summary
    model = Model(inputs=input, outputs=y)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", accuracy])
    return model


def _build_target(train_target_seqs: np.ndarray) -> Tuple[np.ndarray, dict]:
    tokenizer_decoder = Tokenizer(char_level=True)
    tokenizer_decoder.fit_on_texts(train_target_seqs)
    train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
    train_target_data = sequence.pad_sequences(train_target_data,
                                               maxlen=maxlen_seq,
                                               padding='post')
    train_target_data = to_categorical(train_target_data)
    reverse_decoder = {v: k for k, v in tokenizer_decoder.word_index.items()}
    return train_target_data, reverse_decoder


def _build_train_inputs(train_input_grams: np.ndarray,
                        maxlen_seq: int) -> Tuple[np.ndarray, dict]:
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)

    train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    train_input_data = sequence.pad_sequences(train_input_data,
                                              maxlen=maxlen_seq,
                                              padding='post')
    reverse_encoder = {v: k for k, v in tokenizer_encoder.word_index.items()}
    return train_input_data, reverse_encoder


def _build_test_inputs(train_input_grams: np.ndarray,
                       test_input_grams: np.ndarray,
                       maxlen_seq: int) -> np.ndarray:
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)
    # Use the same tokenizer defined on train for tokenization of test

    test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    test_input_data = sequence.pad_sequences(test_input_data,
                                             maxlen=maxlen_seq,
                                             padding='post')
    return test_input_data


def _build_fold_rows(dataset_size: int,
                     num_folds: int) -> List[np.ndarray]:
    fold_size = np.ceil(dataset_size / num_folds)
    all_indices = np.arange(dataset_size)
    fold_rows = [None] * num_folds
    for idx in range(num_folds):
        ind0 = int(idx * fold_size)
        ind1 = int(np.minimum((idx + 1) * fold_size, dataset_size))
        fold_rows[idx] = all_indices[ind0:ind1]
    return fold_rows


def _todo() -> str:
    lines = []
    lines += ['attention']
    to_return = "\n".join(lines)
    return to_return


def _fit_model_on_data(model: Any,
                       x: np.ndarray,
                       y: np.ndarray) -> Any:
    # batch_size = 128
    # batch_size = 256
    # batch_size = 64
    batch_size = 32
    # batch_size = 16

    # fit_epochs = 40
    fit_epochs = 5
    # fit_epochs = 3
    test_size = .1
    # Splitting the data for train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x,
                                                      y,
                                                      test_size=test_size,
                                                      random_state=0)

    # Train the model and validate using the validation set
    fit_history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=fit_epochs,
                            validation_data=(x_val, y_val),
                            verbose=1)
    return model, fit_history


def _do_cross_validation(num_folds: int,
                         train_input_data: np.ndarray,
                         train_target_data: np.ndarray) -> np.ndarray:
    fold_accuracy = np.full((num_folds, ), np.nan)
    dataset_size = train_input_data.shape[0]
    fold_rows = _build_fold_rows(dataset_size, num_folds)

    for fold_idx, rows in enumerate(fold_rows):
        logger.info("Fold #{} / {}".format(fold_idx, num_folds))
        # fold_idx = 0; rows = fold_rows[fold_idx]
        mask = np.full((train_target_data.shape[0], ), False)
        mask[rows] = True

        fold_train_input = train_input_data[~mask, :]
        fold_train_target = train_target_data[~mask, :, :]

        fold_valid_input = train_input_data[mask, :]
        fold_valid_target = train_target_data[mask, :, :]
        model = _build_model(maxlen_seq, n_words, n_tags)
        model, fit_history = _fit_model_on_data(model, fold_train_input, fold_train_target)

        fold_pred = model.predict(fold_valid_input)
        fold_cv_accuracy = np_accuracy(fold_valid_target, fold_pred)
        fold_accuracy[fold_idx] = fold_cv_accuracy
    return fold_accuracy


def _write_submission(ident: str,
                      ids: np.ndarray,
                      y_test_pred: np.ndarray,
                      reverse_decoder: dict) -> str:
    # We expect the solution file to have 119 prediction rows.
    # This file should have a header row. Please see sample submission
    # file on the data page.
    assert 119 == len(ids)
    assert 119 == len(y_test_pred)

    filedir = FILEDIR
    filename = 'submission_{}.csv'.format(ident)
    fullfilename = os.path.join(filedir, filename)

    # https://www.kaggle.com/c/dl-2018-hw2#evaluation

    num_rows = len(ids)
    reprs = [None] * num_rows
    # print(len(test_input_data))
    for i in range(num_rows):
        reprs[i] = str_results(y_test_pred[i], reverse_decoder)

    to_write = pd.DataFrame([ids, reprs], index=['id', 'expected']).T
    logger.info('Writing {}'.format(fullfilename))
    to_write.to_csv(fullfilename, index=False)
    return fullfilename


def _flatten_list_of_lists(x: List[list]) -> list:
    flattened_list = [item for sublist in x for item in sublist]
    return flattened_list


if __name__ == "__main__":
    maxlen_seq = 512
    # maxlen_seq = 7540

    num_folds = 10
    n = 3
    do_cross_validation = True
    # do_cross_validation = False
    do_prediction = False
    # do_prediction = True

    train_input_seqs, train_target_seqs = _load_train_data(maxlen_seq)
    train_input_grams = seq2ngrams(train_input_seqs, n)
    train_input_data, reverse_encoder = _build_train_inputs(train_input_grams, maxlen_seq)
    train_target_data, reverse_decoder = _build_target(train_target_seqs)

    n_words = len(set(train_input_data.flatten()))
    n_tags = train_target_data.shape[2]

    if do_cross_validation:
        fold_accuracy = _do_cross_validation(num_folds, train_input_data, train_target_data)
        logger.info("Average {}-fold accuracy {}".format(num_folds, np.mean(fold_accuracy)))
        logger.info(fold_accuracy)
    if do_prediction:
        model = _build_model(maxlen_seq, n_words, n_tags)
        model.summary()
        model, fit_history = _fit_model_on_data(model, train_input_data, train_target_data)

        # now out of sample stuff
        test_input_seqs, ids = _load_test_data()

        test_input_grams = seq2ngrams(test_input_seqs, n)
        test_input_data = _build_test_inputs(train_input_grams, test_input_grams, maxlen_seq)
        y_test_pred = model.predict(test_input_data[:])
        ident = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        _write_submission(ident, ids, y_test_pred, reverse_decoder)

