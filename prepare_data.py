from PIL import Image
import tensorflow
import keras
import string
from pickle import load
from os import listdir
from pickle import dump
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB7
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
import glob
import numpy as np
from time import sleep
from tqdm import tqdm

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

# extract features from each photo in the directory
def extract_features(directory,model="vgg"):
	if model == "vgg":
		model = VGG16()
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	
	elif model == "inception":
		model = InceptionV3(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)
	
	else:
		model = EfficientNetB7(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)

	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	print("extracting features....")
	for name in tqdm(listdir(directory)):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		if feature_model == "vgg":
			image = preprocess_input_vgg(image)
		else:
			image = preprocess_input_inc(image)	
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		
	filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
	train = load_set(filename)

	filename = 'dataset/Flickr8k_text/Flickr_8k.devImages.txt'
	test = load_set(filename)

	train_features = load_photo_features(str(feature_model) + '.pkl', train)
	print(len(train_features.keys()))
	test_features = load_photo_features(str(feature_model) + '.pkl', test)

	# save to file
	dump(train_features, open(str(feature_model)+"-train.pkl", 'wb'))
	dump(test_features, open(str(feature_model)+"-test.pkl", 'wb'))

def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features  

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

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
	for key, desc_list in descriptions.items():
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

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
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



images_path = 'dataset/Flickr8k_Dataset/'
train_images_path = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_path = 'dataset/Flickr8k_text/Flickr_8k.testImages.txt'

# extract features from all images (outputs  1-dimensional 4,096 element vector)
feature_model = "vgg"
directory = 'dataset/Flickr8k_Dataset'

extract_features(directory,model=feature_model)

# load descriptions
filename = 'dataset/Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

# clean descriptions
clean_descriptions(descriptions)
print("cleaned descriptions: tolowercase, removed single letter words, removed punctuation, etc ")

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# save descriptions
save_descriptions(descriptions, 'descriptions.txt')
print("saved vocabulary to descriptions.txt")

