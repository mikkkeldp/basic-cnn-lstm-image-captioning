import numpy as np
from keras.callbacks import ModelCheckpoint
import os
from utils import *
from model import *
import tensorflow as tf

#initialize GPU for training
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)


#model parameters
reduced_vocab = True
model_type = "merge" 
feature_model = "efficientnet"
glove = True
progressive_loading = False

#training parameters
epochs = 20

## loading data
# train and development (test) dataset have been prediefined in the Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files

# load training dataset (6K)
filename = 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

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

embedding_matrix = None
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
model = BasicModel(vocab_size, max_length, model_type=model_type,feature_model=feature_model,glove=glove, embedding_matrix=embedding_matrix)
# model = AlternativeModel(vocab_size, max_length,feature_model=feature_model,glove=glove, embedding_matrix=embedding_matrix)
# model = ComplexModel(vocab_size, max_length,feature_model=feature_model,glove=glove, embedding_matrix=embedding_matrix)

#define naming of saved .h5 file
if glove:
	model_name = model_type + "-" + feature_model + "-glove-" 
else:
	model_name = model_type + "-" + feature_model  + "-"	


if reduced_vocab:
	model_name += "RV-"

if progressive_loading:
	batch_size = 64
	steps = len(train_descriptions)
	for i in range(epochs):
		# create the data generator
		generator = data_generator_pl(train_descriptions, train_features, tokenizer, max_length, batch_size)
		# fit for one epoch
		model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
		# save model
		model.save("trained_models/PL-" + str(model_name)+"-" + str(i) + '.h5')
else:		
	# define checkpoint callback
	filepath = "trained_models/" + model_name + 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	# fit model
	model.fit([X1train, X2train], ytrain, epochs=epochs, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest), use_multiprocessing=True, workers=8) 
