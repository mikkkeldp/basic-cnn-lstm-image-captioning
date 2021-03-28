
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import load, dump
from numpy import array


PATH = "dataset/Flickr8k_text/"
with open(PATH+"Flickr8k.token.txt") as f:
    data = f.read()

descriptions = dict()


try:
    for el in data.split("\n"):
        tokens = el.split()
        image_id, image_desc = tokens[0], tokens[1:]

        # dropping .jpg from image id
        image_id = image_id.split(".")[0]

        image_desc = " ".join(image_desc)

        # check if image_id is already present or not
        if image_id in descriptions:
            descriptions[image_id].append(image_desc)
        else:
            descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
except Exception as e:
    print("Exception got :- \n", e)

# example
# print(descriptions["1000268201_693b08cb0e"])


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


for k in descriptions.keys():
    value = descriptions[k]
    caption_list = []
    for ec in value:

        # replaces specific and general phrases
        sent = decontracted(ec)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        image_cap = 'startseq ' + sent.lower() + ' endseq'
        caption_list.append(image_cap)
    descriptions[k] = caption_list

# print("AFTER")
# print(descriptions["1000268201_693b08cb0e"])


train_descriptions = dict()
with open("dataset/Flickr8k_text/Flickr_8k.trainImages.txt", "r") as f:
    data = f.read()

try:
    for el in data.split("\n"):
        tokens = el.split(".")
        image_id = tokens[0]
        if image_id in descriptions:
            train_descriptions[image_id] = descriptions[image_id]

except Exception as e:
    print("Exception got :- \n", e)

# getting word frequencies
word_freq = {}

for k in train_descriptions.keys():
    for caption in train_descriptions[k]:

        caption = caption.strip("endseq")
        caption = caption.strip("startseq")
        caption = caption.split(" ")
        for word in caption:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

#remove empty string entry

word_freq.pop('')


#50 most freq words
n = 50
sample  = {k: word_freq[k] for k in list(word_freq)[:50]}

x = np.arange(len(sample))
y = sample.values()
plt.subplots_adjust(bottom=0.2)
plt.title(str(n) + " most frequent words in vocabulary")
plt.ylabel("Frequency")
plt.xlabel("Word")
plt.bar(x,y)
plt.xticks(x, sample.keys(), rotation = 'vertical')
# plt.show()

freq_list = list(word_freq.values())
print("std: ", np.std(freq_list))
print("mean: ", np.mean(freq_list))
print("min: ", np.min(freq_list))
print("max: ", np.max(freq_list))

# number of words that occur more than 10 times (this will be our threshold)

n = 0
for value in word_freq.values():
    if value >= 10:
        n += 1
print("New vocab: ", n)


# remove words occuring less than 10 times, as well as stop words (a, the, etc)