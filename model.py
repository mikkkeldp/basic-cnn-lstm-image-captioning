
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras.optimizers import Adam

from keras.layers import BatchNormalization
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
import keras


"""
	Define Basic RNN model
"""
def BasicModel(vocab_size, max_length, model_type="merge",feature_model="vgg",glove=False, embedding_matrix=None):
	
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

	lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.9)	

	opt = Adam(learning_rate=lr_schedule)
	model.compile(loss='categorical_crossentropy', optimizer=opt)

	return model


"""
	Define the alternative RNN model
"""
def AlternativeModel(vocab_size, max_length,feature_model="vgg",glove=False, embedding_matrix=None):
	
	if feature_model =="inception":
		image_input = Input(shape=(2048,))
	elif feature_model == "vgg":
		image_input = Input(shape=(4096,))
	else: #efficientnet
		image_input = Input(shape=(2560,))

	fe = Dense(256, activation='relu')(image_input)
	image_model = RepeatVector(max_length)(fe)

	if glove:
		dim = 200
	else:
		dim = 256

	caption_input = Input(shape=(max_length,))
	caption_model_1  = Embedding(vocab_size, dim, mask_zero=True)(caption_input)
	caption_model_2 = LSTM(256, return_sequences=True)(caption_model_1)
	caption_model = TimeDistributed(Dense(dim))(caption_model_2)

	# se2 = Dropout(0.5)(se1)
	# se3 = LSTM(256)(se2)

	output_1 = concatenate([image_model, caption_model])
	output_2 = Bidirectional(LSTM(256, return_sequences=False))(output_1)
	output = Dense(vocab_size, activation='softmax')(output_2)

	model = Model(inputs=[image_input, caption_input], outputs=output)

	if glove:
		model.layers[2].set_weights([embedding_matrix])
		model.layers[2].trainable = False

	model.compile(loss='categorical_crossentropy', optimizer="adam")

	return model