from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
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

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):

	# lines = to_lines(descriptions)
	# tokenizer = Tokenizer()
	# tokenizer.fit_on_texts(lines)

	# list of training caps
	all_train_captions = []
	for key, val in train_descriptions.items():
		for cap in val:
			all_train_captions.append(cap)	
	
	# get vocab and limit with threshold
	word_count_threshold = 0
	word_counts = {}
	nsents = 0
	for sent in all_train_captions:
		nsents += 1
		for w in sent.split(' '):
			word_counts[w] = word_counts.get(w, 0) + 1
	vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

	#  two dictionaries to map words to an index and vice versa
	ixtoword = {}
	wordtoix = {}
	ix = 1
	for w in vocab:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	vocab_size = len(ixtoword) + 1

	return ixtoword, wordtoix, vocab_size

# calculate the length of the description with the most words
def max_length(descriptions):
	all_desc = list()
	for key in train_descriptions.keys():
		[all_desc.append(d) for d in train_descriptions[key]]
	lines = all_desc
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
		in_text = 'startseq'
		for i in range(max_length):
			sequence = [tokenizer[w] for w in in_text.split() if w in tokenizer]
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

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	print("evaluating model...")
	for key, desc_list in tqdm(descriptions.items()):
		# generate description
		# key = key + ".jpg"
		photo = photos[key][0].reshape((1,4096))
		yhat = generate_desc(model, wordtoix, photo, max_length)
		# print("yhat ", yhat)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		# print("ref:" , references)
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

feature_model = 'vgg'

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer

ixtoword, wordtoix, vocab_size = create_tokenizer(train_descriptions) #word to index

print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Longest Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features

feature_model = "vgg"

test_features = load(open(str(feature_model) + '-test.pkl', 'rb'))
print('Photos: test=%d' % len(test_features))
print(test_features.keys())
# load the model
filename = 'merge-vgg.h5' #insert your best model here
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, wordtoix, max_length)