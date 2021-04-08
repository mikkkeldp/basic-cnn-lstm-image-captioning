from pickle import load
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from tqdm import tqdm
from utils import *


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	print("Starting evaluation...")
	for key, desc_list in tqdm(descriptions.items()):
		# generate description
		if beam:
			yhat = generate_desc_beam_search(model, tokenizer, photos[key], max_length,5)
		else:
			yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3333, 0.3333, 0.3333, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

#model parameters
feature_model = "resnet"
beam = False
reduce_v = False

# prepare train set
filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# prepare test set
filename = 'dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

# load test descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# load train descriptions
if reduce_v:
	train_descriptions = load_clean_descriptions('reduced_descriptions.txt', train)
else:	
	train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Longest Description Length: %d' % max_length)

# photo features
test_features = load_photo_features(feature_model+'.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'trained_models/' + 'merge-resnet-model-ep004-loss3.428-val_loss3.800.h5' #insert your best model here
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)