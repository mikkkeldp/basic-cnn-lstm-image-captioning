from pickle import dump
from utils import *


dataset_dir = 'dataset/Flickr8k_Dataset'
descriptions_tokens_dir = 'dataset/Flickr8k_text/Flickr8k.token.txt'

# set parameters for data preperation
feature_model = "inception" 	# set pre-trained CNN for feature extraction
reduce_vocab = True 	# reduced vocab toggle

#extract feature vectors
features = extract_features(dataset_dir, model=feature_model)
print('Extracted Features: %d' % len(features))

# save to file
dump(features, open(feature_model +'.pkl', 'wb'))

# load descriptions
doc = load_doc(descriptions_tokens_dir)

# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

# clean descriptions
descriptions = clean_descriptions(descriptions)
print("cleaned descriptions: tolowercase, removed single letter words, removed punctuation, etc.")

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# set array of new vocab
new_words = []
if reduce_vocab:
	train_descriptions = get_train_descriptions(descriptions)
	word_freq = get_word_freq(train_descriptions) # word freq sorted by freq 
	new_words = get_new_vocab(word_freq)

# save descriptions
if reduce_vocab:
	filename = "reduced_descriptions.txt"
else:
	filename = 'descriptions.txt'

save_descriptions(descriptions, filename, reduce_vocab, new_words)
print("saved vocabulary to " + filename)


