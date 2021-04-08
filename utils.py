from PIL import Image
import tensorflow
import keras
import string
import glob
from pickle import load
from os import listdir
from pickle import dump
from pickle import load
from numpy import array
import string
from os import listdir
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_res
from numpy import argmax

# only use if you have tensorflow nightly installed
# from keras.applications.efficientnet import EfficientNetB7
# from keras.applications.efficientnet import preprocess_input as preprocess_input_eff


# -------------- PREPARE_DATA UTILS --------------


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# extract features from each photo in the directory
def extract_features(directory,model="vgg"):
	feature_model = model
	if model == "vgg":
		model = VGG16()
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	
	elif model == "inception":
		model = InceptionV3(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)
	
	elif model == "efficientnet":
		model = EfficientNetB7(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)
	else:
		model = ResNet50(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)

	# extract features from each photo
	features = dict()
	print("extracting features....")
	for name in tqdm(listdir(directory)):
		# load an image from file
		filename = directory + '/' + name
		if feature_model == "efficientnet":
			image = load_img(filename, target_size=(600, 600))
		else:	
			image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		if feature_model == "vgg":
			image = preprocess_input_vgg(image)
		elif feature_model == "inception":
			image = preprocess_input_inc(image)	
		elif feature_model == "resnet":
			image = preprocess_input_res(image)		
		else:
			image = preprocess_input_eff(image)	

		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
	return features	


# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for _, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)  
	return descriptions

def remove_out_of_vocab_words(caption, new_vocab):
	words = caption.split(" ")
	new_cap = ""
	for word in words:
		if word in new_vocab:
			if new_cap != "":
				new_cap += " " + word
			else:
				new_cap += word	
	return new_cap

# save descriptions to file, one per line
def save_descriptions(descriptions, filename,reduce_v,new_vocab):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			if reduce_v:
				desc = remove_out_of_vocab_words(desc, new_vocab)
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc  

def get_train_descriptions(descriptions):
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

	return train_descriptions	

def get_word_freq(train_descriptions):
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
		return word_freq

def get_new_vocab(word_freq):
	new_vocab = []
	for key, value in word_freq.items():
		if value >= 10:
			new_vocab.append(key)
	return new_vocab



# -------------- TRAIN UTILS --------------

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
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	lines = all_desc
	return max(len(d.split()) for d in lines)
  
# -------------- EVAL UTILS --------------

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for _ in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# generate a description for an image using beam search
def generate_desc_beam_search(model, tokenizer, image, max_length, beam_index=3):
	# in_text --> [[idx,prob]] ;prob=0 initially
	in_text = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
	while len(in_text[0][0]) < max_length:
		tempList = []
		for seq in in_text:
			padded_seq = pad_sequences([seq[0]], maxlen=max_length)
			preds = model.predict([image,padded_seq], verbose=0)
			# Take top (i.e. which have highest probailities) `beam_index` predictions
			top_preds = np.argsort(preds[0])[-beam_index:]
			# Getting the top `beam_index` predictions and 
			for word in top_preds:
				next_seq, prob = seq[0][:], seq[1]
				next_seq.append(word)
				# Update probability
				prob += preds[0][word]
				# Append as input for generating the next word
				tempList.append([next_seq, prob])
		in_text = tempList
		# Sorting according to the probabilities
		in_text = sorted(in_text, reverse=False, key=lambda l: l[1])
		# Take the top words
		in_text = in_text[-beam_index:]
	in_text = in_text[-1][0]
	final_caption_raw = [int_to_word(i,tokenizer) for i in in_text]
	final_caption = []
	for word in final_caption_raw:
		if word=='endseq':
			break
		else:
			final_caption.append(word)
	final_caption.append('endseq')
	return ' '.join(final_caption)	

# map intereger to word
def int_to_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# -------------- NEW PREDICTION UTILS --------------

#strip start and end tokens 

def strip_tokens(input):
	output = input.replace("startseq ", "")
	output = output.replace(" endseq", "")
	return output
