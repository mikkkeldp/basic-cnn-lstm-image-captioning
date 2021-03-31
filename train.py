from PIL import Image
import os
import numpy as np
import tensorflow
import keras
import string
from pickle import load
from os import listdir
from pickle import dump
import glob
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

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

#create sequences for progressive loading
def create_sequences_pl(tokenizer, max_length, desc_list, photo, vocab_size):
	X1, X2, y = list(), list(), list()
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
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

#data generator for progressive loading
def data_generator_pl(descriptions, photos, tokenizer, max_length, vocab_size):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences_pl(tokenizer, max_length, desc_list, photo, vocab_size)
			yield [in_img, in_seq], out_word			

# calculate the length of the description with the most words
def max_length(descriptions):
	all_desc = list()
	for key in train_descriptions.keys():
		[all_desc.append(d) for d in train_descriptions[key]]
	lines = all_desc
	return max(len(d.split()) for d in lines)
  


# define the captioning model
def define_model(vocab_size, max_length, model_type="merge",feature_model="vgg",glove=False):
	
	if feature_model =="inception":
		inputs1 = Input(shape=(2048,))
	elif feature_model == "vgg":
		inputs1 = Input(shape=(4096,))
	else: #efficientnet
		inputs1 = Input(shape=(2560,))

	if glove:
		dim = 200
	else:
		dim = 256

	if model_type == "merge":
		#merge model

		# feature extractor model
		fe1 = Dropout(0.5)(inputs1)
		# b1 = BatchNormalization()(fe1)
		fe2 = Dense(256, activation='relu')(fe1)

		# sequence model
		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, dim, mask_zero=True)(inputs2)
		# b2 = BatchNormalization()(se1)
		se2 = Dropout(0.1)(se1)
		se3 = LSTM(256)(se2)

		# decoder model
		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(vocab_size, activation='softmax')(decoder2)

		# tie it together [image, seq] [word]
	else:
		#inject model

		fe1 = Dropout(0.01)(inputs1)
		b1 = BatchNormalization()(fe1)
		fe2 = Dense(256, activation='relu')(b1)

		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, dim, mask_zero=True)(inputs2)
		b2 = BatchNormalization()(se1)
		se2 = Dropout(0.5)(b2)

		input = add([fe2, se2])
		encoder = LSTM(256)(input)
		decoder = Dense(256, activation='relu')(encoder)
		outputs = Dense(vocab_size, activation='softmax')(decoder)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)

	if glove:
		model.layers[2].set_weights([embedding_matrix])
		model.layers[2].trainable = False

	#set learning rate	(not currently in use)
	opt = keras.optimizers.Adam(learning_rate=0.01)

	model.compile(loss='categorical_crossentropy', optimizer="adam")

	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

#initialize GPU for training
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)


#training parameters
reduced_vocab = True
model_type = "merge" 
feature_model = "vgg"
glove = True
progressive_loading = False



## loading data
# train and development (test) dataset have been prediefined in the Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files

# load training dataset (6K)
filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)

if reduced_vocab:
	train_descriptions = load_clean_descriptions('reduced_descriptions.txt', train)
else:
	train_descriptions = load_clean_descriptions('descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features(feature_model +'.pkl', train)
print('Photos: train=%d' % len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

if glove:
	#glove embeddings
	embeddings_index = {} 
	f = open(os.path.join("", 'glove.6B.200d.txt'), encoding="utf-8")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

	# represents each word in vocab through a 200D vector
	embedding_dim = 200
	i = 0
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	for word in tokenizer.word_index:
		print(word)
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector	
			i+=1


# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Longest Description Length: %d' % max_length)

# prepare sequences (not in use)
if progressive_loading:
	X1train, X2train, ytrain = create_sequences_pl(tokenizer, max_length, train_descriptions, train_features, vocab_size)
else:
	X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

# load test set
filename = 'dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))


# descriptions
if reduced_vocab:
	test_descriptions = load_clean_descriptions('reduced_descriptions.txt', test)
else:
	test_descriptions = load_clean_descriptions('descriptions.txt', test)

print('Descriptions: test=%d' % len(test_descriptions))


# photo features
test_features = load_photo_features(feature_model +'.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences (not in use)
if progressive_loading:
	X1test, X2test, ytest = create_sequences_pl(tokenizer, max_length, test_descriptions, test_features, vocab_size)
else:	
	X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

#define model
model = define_model(vocab_size, max_length, model_type=model_type,feature_model=feature_model,glove=glove)

#define naming of .h5 file
if glove:
	model_name = model_type + "-" + feature_model + "-glove-" 
else:
	model_name = model_type + "-" + feature_model  + "-"	


if reduced_vocab:
	model_name += "RV-"

if progressive_loading:
	epochs = 20
	batch_size = vocab_size
	steps = len(train_descriptions)
	for i in range(epochs):
		# create the data generator
		generator = data_generator_pl(train_descriptions, train_features, tokenizer, max_length, batch_size)
		# fit for one epoch
		model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
		# save model
		model.save(str(model_name)+"-" + str(i) + '.h5')
else:		

	# define checkpoint callback
	filepath = model_name + 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	# fit model
	model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest)) 
