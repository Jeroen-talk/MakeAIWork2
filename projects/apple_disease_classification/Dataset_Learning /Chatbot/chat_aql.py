# start this file from folder projects/apple_disease_classification for json file to open correctly
import pickle
import json
import random
import tensorflow
import tensorflow as tf
import tflearn
import numpy
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nltk
# nltk.download('all')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


appleBatch = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal',
              'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'blotch', 'blotch', 'blotch', 'blotch', 'rot', 'scab', 'scab', 'scab']

# controle of aantal appels in batch wel correct is
countBatch = 0
for i in appleBatch:
    countBatch = countBatch+1
# print(countBatch)

# De appels opslpitsen in klasse


countNormal = appleBatch.count('normal')
countBlotch = appleBatch.count('blotch')
countScab = appleBatch.count('scab')
countRot = appleBatch.count('rot')


def aqlclass(appleBatch):
    countNormal = appleBatch.count('normal')
    countBlotch = appleBatch.count('blotch')
    countScab = appleBatch.count('scab')
    countRot = appleBatch.count('rot')
    if countNormal == 80:
        return ('AQL 1')
    elif countScab + countBlotch + countRot <= 5:
        return ('AQL 2')
    elif 6 <= countScab + countBlotch + countRot <= 10:
        return ('AQL 3')
    else:
        return ('AQL 4
                ')


appleClass = aqlclass(appleBatch)
# print(appleClass, '\n')

# print('Normale appels:',countNormal)
# print('Blotch appels:',countBlotch)
# print('Scab appels:',countScab)
# print('Rotte appels:',countRot)

with open('chat_talk.json') as file:
    data = json.load(file)

# print (data['intents'])
# pickle and try checks if there's already a model and data only runs code if there isn't

# try:
#     with open('data.pickle', 'rb') as pfile:
#         words, labels, training, output = pickle.load(pfile)
# except:
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

    # turn everything in lower case
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))  # remove duplicates

    labels = sorted(labels)

    training = []
    output = []  # this is the text in json file that also needs one hot encoding

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:  # if word exists add to bag, else 0
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]  # copy of out empty
        # look through labels list, see where tag is and set value to 1 in output row
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)  # conversion to array for tflearn
    output = numpy.array(output)

    # with open('data.pickle', 'wb') as pfile:
    #     pickle.dump((words, labels, training, output), pfile)

# tf.reset_default_graph() #see if this is replaced by something else?

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")
# except:
#     model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#     model.save("model.tflearn")

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


def addOutcomeToResponse(response):
    response = response.replace("{countBatch}", f'{countBatch}')
    response = response.replace("{countNormal}", f'{countNormal}')
    response = response.replace("{countScab}", f'{countScab}')
    response = response.replace("{countBlotch}", f'{countBlotch}')
    response = response.replace("{countRot}", f'{countRot}')
    response = response.replace("{appleClass}", f'{appleClass}')
    return response


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
