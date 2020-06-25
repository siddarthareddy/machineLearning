from tensorflow import keras
import numpy as np

#imdb sentiment classification dataset
imdb_data = keras.datasets.imdb
#only take words in reviews which are 10000 most frequent, rare words are useless, overfit the model
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=10000)
print(len(train_data))
word_index = imdb_data.get_word_index()
print("type of word_index: ",type(word_index))
print("type of word_index.values(): ",type(word_index.values()))
print("Index of this: ", word_index["this"])
a = list(word_index.values())
print("no. of words in dictionary", len(a))
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
print("Index of this: ", word_index["this"])
#print(word_index)
reverse_word_index = [(value, key) for (key, value) in word_index.items()]
print(reverse_word_index[100])
print(type(reverse_word_index[100]))
print(reverse_word_index[100][0])
print(dict(reverse_word_index)[100])
reverse_word_index = dict(reverse_word_index)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
print(train_data[0])
print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
#final layer activation is sigmoid, not relu, as we want a binary or probablistic output
model.add(keras.layers.Dense(1, activation="sigmoid"))

