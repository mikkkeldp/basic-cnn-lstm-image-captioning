from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet import preprocess_input as preprocess_input_eff
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.models import load_model
import sys
from keras.preprocessing.text import Tokenizer
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from tqdm import tqdm
from os import listdir
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
def extract_features(filename,model="vgg"):
	feature_model = model
	if model == "vgg":
		model = VGG16()
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	
	elif model == "inception":
		model = InceptionV3(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)
	
	else:
		model = EfficientNetB7(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)

	# load an image from file

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
	else:
		image = preprocess_input_eff(image)	
	# get features
	feature = model.predict(image, verbose=0)

	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# calculate the length of the description with the most words
def max_length(descriptions):
	all_desc = list()
	for key in train_descriptions.keys():
		[all_desc.append(d) for d in train_descriptions[key]]
	lines = all_desc
	return max(len(d.split()) for d in lines)

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

# map intereger to word
def int_to_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

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

# covert a dictionary of clean descriptions to a list of descriptions
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

#strip start and end tokens 

def strip_tokens(input):
	output = input.replace("startseq ", "")
	output = output.replace(" endseq", "")
	return output

# model parameters
if len(sys.argv) < 2:
	print("Invalid arguments")
else:
	image_name = str(sys.argv[1])
	feature_model = str(sys.argv[2])
	reduced_v = sys.argv[3]


# load training dataset (6K)
filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)


# load train descriptions
if reduced_v:
	train_descriptions = load_clean_descriptions('reduced_descriptions.txt', train)
else:	
	train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))




# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)

#determine max length of caption
max_length = max_length(train_descriptions)

# load the model
model = load_model('merge-efficientnet-glove-RV-model-ep003-loss3.047-val_loss3.335.h5')
photo = extract_features(image_name, feature_model)

#get ground truth
try:
	ground_truth = strip_tokens(train_descriptions[image_name.replace(".jpg","")][0])
except ValueError:
	filename = 'dataset/Flickr8k_text/Flickr_8k.devImages.txt'
	test = load_set(filename)
	test_descriptions = load_clean_descriptions('descriptions.txt', test)
	ground_truth = strip_tokens(test_descriptions[image_name.replace(".jpg","")][0])
except:
	filename = 'dataset/Flickr8k_text/Flickr_8k.testImages.txt'
	test = load_set(filename)
	test_descriptions = load_clean_descriptions('descriptions.txt', test)
	ground_truth = strip_tokens(test_descriptions[image_name.replace(".jpg","")][0])	


# generate description
beam_3 = strip_tokens(generate_desc_beam_search(model, tokenizer, photo, max_length, 3))
beam_5 = strip_tokens(generate_desc_beam_search(model, tokenizer, photo, max_length, 5))
beam_10 = strip_tokens(generate_desc_beam_search(model, tokenizer, photo, max_length, 10))
greedy = strip_tokens(generate_desc(model, tokenizer, photo, max_length))

# print("Greedy: ", greedy)
# print("Beam 3: ", beam_3)
# print("Beam 5: ", beam_5)
# print("Beam 10: ", beam_10)

img = mpimg.imread(image_name)
imgplot = plt.imshow(img)
img_height = len(img)
plt.axis('off')
plt.title("Ground truth: " + ground_truth)
string = "Greedy: " + greedy + "\nBeam 3: " + beam_3 + "\nBeam 5: "+  beam_5 +"\nBeam 10: "+ beam_10
plt.text(0,img_height*1.2, string)
plt.savefig(str(image_name) + "_caption.png")
