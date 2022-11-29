#!/usr/bin/env python

from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy
import numpy as np
from pathlib import Path
import tflearn
import pathlib
import json
import random
import os
import nltk
# Download data for chatbot model
# nltk.download('all')
stemmer = LancasterStemmer()

# Hides the tensorflow warnings
tf.get_logger().setLevel('ERROR')

# Load dataset of apples
dataset_path = "run/data/apples/"
data_dir = pathlib.Path(dataset_path)
apple_images = list()

img_size = (224, 224)
batch_size = 32

# Load model to predict
model = tf.keras.models.load_model(
    'run/models/tl_24112022_20h37_1.h5', custom_objects=None, compile=True, options=None)
classifier = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=img_size+(3,))])

# model.summary()

apple_images = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size)

# Prodiction function
predictions = np.zeros(0)
predictions = model.predict(apple_images)
prediction_labels = ['Blotch', 'Normal', 'Rot', 'Scab']
labels = list()

for row in predictions:

    # Look at first label
    highest = row[0]
    index = 0

    for i in range(1, 4):

        if row[i] > highest:

            highest = row[i]
            index = i

    labels.append(prediction_labels[index])

# print(labels)

# Apple counter
countBatch = 0
for i in labels:
    countBatch = countBatch+1

# print(countBatch)

# Divide the apple batch in to classes
countNormal = labels.count('Normal')
countBlotch = labels.count('Blotch')
countScab = labels.count('Scab')
countRot = labels.count('Rot')

# AQL model


def aqlclass(labels):
    if countNormal == 80:
        return ('AQL 1')
    elif countScab + countBlotch + countRot <= 5:
        return ('AQL class 2')
    elif 6 <= countScab + countBlotch + countRot <= 10:
        return ('AQL 3')
    else:
        return ('AQL 4')


appleClass = aqlclass(labels)

# ### Print counter results
# print('\n', appleClass, '\n')
# print('Normale appels:', countNormal)
# print('Blotch appels:', countBlotch)
# print('Scab appels:', countScab)
# print('Rotte appels:', countRot)

# Load JSON file for chatbot
with open('run/data/chat_talk.json') as file:
    data = json.load(file)

# Chatbot
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Lower case
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

# Training the chatbot model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("run/models/chatbot/model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# Added output of chatbot with outcomes of batches


def addOutcomeToResponse(response):
    response = response.replace("{countBatch}", f'{countBatch}')
    response = response.replace("{countNormal}", f'{countNormal}')
    response = response.replace("{countScab}", f'{countScab}')
    response = response.replace("{countBlotch}", f'{countBlotch}')
    response = response.replace("{countRot}", f'{countRot}')
    response = response.replace("{appleClass}", f'{appleClass}')
    return response

# Activate and exit chatbot


def chat():
    print('\n'"Hello Apple LOVER,", '\n \n' "How may I help you?", '\n',
          "Type here to start the conversation. You can type quit any time to stop.", '\n')
    while True:
        inp = input('Ask me anything about apples: ')
        if inp.lower() == 'quit':
            break
        if inp.lower() == 'stop':
            break
        if inp.lower() == 'exit':
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = [addOutcomeToResponse(response)
                             for response in tg['responses']
                             ]
                # responses = tg['responses']

        print(random.choice(responses))


chat()
