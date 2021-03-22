from PIL import Image
import tensorflow
import keras
import string
from pickle import load
from os import listdir
import glob
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.preprocessing import image
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers (used for loading training or testing set)
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features    

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# X1,		X2 (text sequence), 						y (word)
# photo	    startseq, 									little
# photo	    startseq, little,							girl
# photo	    startseq, little, girl, 					running
# photo	    startseq, little, girl, running, 			in
# photo	    startseq, little, girl, running, in, 		field
# photo	    startseq, little, girl, running, in, field, endseq

# create sequences of images, input sequences and output words for an image.  
# will transform the data into input-output pairs of data for training the model. 
# y is the next word the model is suppose to predict, while X2 is the input (encoded text as integers). 
# The output data will therefore be a one-hot encoded version of each word, representing an idealized 
# probability distribution with 0 values at all word positions except the actual word position, which has a value of 1.

def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)   

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)    

# define the captioning model
def define_model(vocab_size, max_length, model_type="merge"):
	if model_type == "merge":
		#merge model

		inputs1 = Input(shape=(2048,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)

		inputs2 = Input(shape=(max_length,))
		embedding_dim = 200
		se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = LSTM(256)(se2)

		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(vocab_size, activation='softmax')(decoder2)

	else:
		#inject model

		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)

		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)

		input = add([fe2, se2])
		encoder = LSTM(256)(input)
		decoder = Dense(256, activation='relu')(encoder)
		outputs = Dense(vocab_size, activation='softmax')(decoder)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)

	#we do not want to retrain the weights in our embedding layer
	#freeze that layer
	model.layers[2].set_weights([embedding_matrix])
	model.layers[2].trainable = False
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	return model
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

#setup keras and tensorflow to run on GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#set number of GPU's and CPU cores
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
# from tensorflow.compat.v1.keras import backend as K

tf.compat.v1.keras.backend.set_session(sess)


#resources paths
token_path = "dataset/Flickr8k_text/Flickr8k.token.txt"
train_images_path = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_path = 'dataset/Flickr8k_text/Flickr_8k.testImages.txt'
images_path = 'dataset/Flickr8k_Dataset/'

# open tokens: image name, along with caption numbers and caption

# 1000268201_693b08cb0e.jpg#0     A child in a pink dress is climbing up a set of stairs in an entry way .
# 1000268201_693b08cb0e.jpg#1     A girl going into a wooden building .
# 1000268201_693b08cb0e.jpg#2     A little girl climbing into a wooden playhouse .
# 1000268201_693b08cb0e.jpg#3     A little girl climbing the stairs to her playhouse .
# 1000268201_693b08cb0e.jpg#4     A little girl in a pink dress going into a wooden cabin .


doc = open(token_path,'r').read()

#create dict of descriptions: name of the image is the key and list of 
# captions per key

descriptions = dict()
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
          image_id = tokens[0].split('.')[0]
          image_desc = ' '.join(tokens[1:])
          if image_id not in descriptions:
              descriptions[image_id] = list()
          descriptions[image_id].append(image_desc)

#clean descriptions (remove punctiation and lowercase)
table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc_list[i] =  ' '.join(desc)

#define vocab

vocabulary = set()
for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))

#save iamge ID's and new cleaned captions in new file
lines = list()
for key, desc_list in descriptions.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
new_descriptions = '\n'.join(lines)

#load training image ID's 
doc = open(train_images_path,'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
      identifier = line.split('.')[0]
      dataset.append(identifier)

train = set(dataset)

#save training and testing images in train_img and test_img (respectively)
img = glob.glob(images_path + '*.jpg')

train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
train_img = []
for i in img: 
    if i[len(images_path):] in train_images:
        train_img.append(i)

test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
test_img = []
for i in img: 
    if i[len(images_path):] in test_images: 
        test_img.append(i)


#  load the descriptions of the training images into a dictionary
train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        train_descriptions[image_id].append(desc)

#list of training caps
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

# To make our model more robust we will reduce our vocabulary to only those 
# words which occur at least 10 times in the entire corpus.

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

print('New Vocabulary = %d' % (len(vocab)))


#  two dictionaries to map words to an index and vice versa
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1
print('Vocabulary Size: %d' % vocab_size)

 # determine the maximum sequence length
all_desc = list()
for key in train_descriptions.keys():
    [all_desc.append(d) for d in train_descriptions[key]]
lines = all_desc
max_length = max(len(d.split()) for d in lines)

print('Longest Description Length: %d' % max_length)

#Glove Embeddings
embeddings_index = {} 
f = open(os.path.join("", 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#encode features: 

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

encoding_train = {}

for img in train_img:
    encoding_train[img[len(images_path):]] = encode(img)
train_features = encoding_train

encoding_test = {}
for img in test_img:
    encoding_test[img[len(images_path):]] = encode(img)



#training 
epochs = 30
batch_size = 3
steps = len(train_descriptions)//batch_size
model = define_model(vocab_size, max_length, "merge")
generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)

filepath = 'glove_inception_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

