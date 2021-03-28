from PIL import Image
import tensorflow
import keras
import string
from pickle import load
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB7
from keras.applications.inception_v3 import preprocess_input


# extract features from each photo in the directory
def extract_features(directory, model="vgg"):
	if model =="vgg":
	# load the model
		model = VGG16()
		# re-structure the model
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	elif model =="inception":
		model = InceptionV3(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)
	else:
		model = EfficientNetB7(weights='imagenet')
		model = Model(model.input, model.layers[-2].output)

	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
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

# extract features from all images (outputs  1-dimensional 4,096 element vector)

# directory = 'dataset/Flickr8k_Dataset'
# feature_model = "vgg"
# features = extract_features(directory, model=feature_model)
# print('Extracted Features: %d' % len(features))
# # save to file
# dump(features, open(str(feature_model) +'.pkl', 'wb'))

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

reduce_vocab = True
if reduce_vocab:
	train_descriptions = get_train_descriptions(descriptions)
	word_freq = get_word_freq(train_descriptions) # word freq sorted by freq 
	new_words = get_new_vocab(word_freq)

# save descriptions
if reduce_vocab:
	filename = "reduced_descriptions.txt"
else:
	filename = 'descriptions.txt'

save_descriptions(descriptions, filename,reduce_vocab, new_words)
print("saved vocabulary to " + filename)


