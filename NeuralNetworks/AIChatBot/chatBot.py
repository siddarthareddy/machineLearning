import json
import random

import nltk
import numpy
import tensorflow
import tflearn
import pickle
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    queries = []
    query_group = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            #print(wrds)
            words.extend(wrds)
            queries.append(wrds)
            query_group.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print(queries)
    print(words)
    print(query_group)
    labels = sorted(labels)

    #i/p to the NN cannot be string or list of words ....
    # need to be in numbers, also in uniform format for every i/p
    #bag of words that represents all words, one hot encoding

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, query in enumerate(queries):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in query]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(query_group[x])] = 1

        training.append(bag)
        output.append(output_row)
    print(output)
    print(training)
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("start talking with the bot(type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that")
chat()