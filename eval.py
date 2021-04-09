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
			yhat = generate_desc_beam_search(model, tokenizer, photos[key], max_length,beam_k)
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
feature_model = "efficientnet"
reduce_v = True
beam = False
beam_k = 5

#insert your best model here
model_file_name = 'merge-efficientnet-glove-RV-model-ep002-loss3.057-val_loss3.323.h5'

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

model = load_model('trained_models/' + model_file_name)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)