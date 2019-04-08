import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

import numpy as np

import matplotlib.pyplot as plt

import time
"""
Following the tutorials :
https://www.tensorflow.org/tutorials/keras/basic_text_classification
"""

NAME = "Text-review-classification-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Download the dataset
def download_imdb_dataset():
    """Download imdb database

    # Returns
        train, test: A tuple of training data and labels.
            train_data, train_labels: A tuple of training data and labels.
            test_data, test_labels: A tuple of testing data and labels.
    """
    imdb = keras.datasets.imdb
    return imdb.load_data(num_words=10000)


# Explore the data
def explore_data(data):
    """Explore all data

    # Arguments
        data: tuple, training and testing data.
    """
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = (
        download_imdb_dataset())
    print("training entries: {}, labels: {}".format(
        len(train_data), len(train_labels)))

    print("data vectorize: " + str(train_data[0][:10]) + " ...")

    # Convert to word :
    word_index = imdb.get_word_index()
        # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    reverse_word_index = dict([(value, key) for (key, value)
        in word_index.items()])
    print("data decode : '" + ' '.join([reverse_word_index.get(i, '?')
        for i in train_data[0][:10]]) + "...'")

# Prepare the data
def preprocessing(train_data, test_data):
    """Preprocessing the data in a format that model can understand

    # Arguments
        train_data: list, training data.
        test_data: list, testing data.

    # Returns:
        train_data, test_data: A tuple of data after processing.
    """
    imdb = keras.datasets.imdb
    word_index = imdb.get_word_index()
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)

    return (train_data, test_data)

# Model
def create_model():
    """Creates an instance of a model.
    # Returns
        An Model instance.
    """

    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(  optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])

    model.summary()
    return (model)

def validation_set(train_data, train_labels):
    """Create the validation set

    # Arguments
        train_data: list, train data
        train_labels: list, train label

    # Returns
        A tuple ou input and output validation
            x_val: list, validation 10000 inputs
            partial_x_train: list, training inputs
            y_val: list, validation 10000 outputs
            partial_y_train: list, training outputs
    """
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    return ((x_val, partial_x_train), (y_val, partial_y_train))

def training(model, train_data, train_labels):
    """Training the model

    # Arguments
        model: my Keras Model.
        train_data: list, train texts.
        train_labels: list, train labels.

    # Returns
        history: tensorfow stuff
    """
    (x_val, partial_x_train), (y_val, partial_y_train) = (
        validation_set(train_data, train_labels))
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=30,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[tensorboard])
    return (history)

def graph(history):
    """Create a graph of accruacy and loss over time

    #Arguments :
        history: dict, history model tensorflow.
    """
    history_dict = history.history
    acc         = history_dict['acc']
    val_acc     = history_dict['val_acc']
    loss        = history_dict['loss']
    val_loss    = history_dict['val_loss']
    epochs      = range(1, len(acc) + 1)
    # bo = "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
     # b = "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# My Testing :
data                    = download_imdb_dataset()
(train_data, train_labels), (test_data, test_labels) = data
explore_data(data)
train_data, test_data   = preprocessing(train_data, test_data)
model                   = create_model()
history                 = training(model, train_data, train_labels)
results                 = model.evaluate(test_data, test_labels)
print(results)
graph(history)
