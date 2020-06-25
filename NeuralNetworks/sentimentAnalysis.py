from tensorflow import keras

#imdb sentiment classification dataset
imdb_data = keras.datasets.imdb
#only take words in reviews which are 10000 most frequent, rare words are useless, overfit the model
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=88000)
# print(len(train_data))
word_index = imdb_data.get_word_index()
# print("type of word_index: ",type(word_index))
# print("type of word_index.values(): ",type(word_index.values()))
# print("Index of this: ", word_index["this"])
a = list(word_index.values())
# print("no. of words in dictionary", len(a))
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
# print("Index of this: ", word_index["this"])
#print(word_index)
reverse_word_index = [(value, key) for (key, value) in word_index.items()]
# print(reverse_word_index[100])
# print(type(reverse_word_index[100]))
# print(reverse_word_index[100][0])
# print(dict(reverse_word_index)[100])
reverse_word_index = dict(reverse_word_index)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
# print(train_data[0])
# print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)
'''
#model
model = keras.Sequential()
#embedding layer tries to group "similar" words, by creating word vectors for each word, and pass them to further layers
#generate 10000 word vectors of 16 dimensions randomly
#use context of words to move vectors similar to each other
model.add(keras.layers.Embedding(88000, 16))
#shrinks data, from higher dimension to lower dim
model.add(keras.layers.GlobalAveragePooling1D())
#this dense layer looks for patterns of words
# ex: this is a great story and this is a good story, come closer with avg
model.add(keras.layers.Dense(16, activation="relu"))
#final layer activation is sigmoid, not relu, as we want a binary or probablistic output
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
#next we have to compile and train our model
#give it optimizer, loss function, give it metrics

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# 40 iterations over data, 512 reviews loaded at once,
# batch_size helpful when working with 100's of GB of data
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

#this way we can train bunch of different models by tweaking hyper parameters like no. of neurons in second layer
#save only best models
model.save("model.h5")
'''
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded
#Has the elements of an emotionally gripping story. Yet is feels less like a romance than like a coffee-table book celebrating the magic of special effect.
model = keras.models.load_model("model.h5")

with open("../data/review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").replace(":", " ").replace("\"", " ").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
