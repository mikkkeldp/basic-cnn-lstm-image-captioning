
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
import keras
from tensorflow.keras import regularizers
from keras.initializers import RandomUniform



def categorical_crossentropy_from_logits(y_true, y_pred):
    # Discarding is still needed although CaptionPreprocessor.preprocess_batch
    # has added dummy words as all-zeros arrays because the sum of losses is
    # the same but the mean of losses is different.
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
    return loss

def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    # Actually discarding is not needed if the dummy is an all-zeros array
    # (It is indeed encoded in an all-zeros array by
    # CaptionPreprocessing.preprocess_batch)
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent dummy or padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                               tf.argmax(y_pred, axis=1)),
                                      dtype=tf.float32))
    return accuracy



"""
	Define Basic RNN model
"""
def BasicModel(vocab_size, max_length, model_type="merge",feature_model="vgg",glove=False, embedding_matrix=None):
	
	if feature_model =="inception" or feature_model == "resnet":
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
		fe1 = Dropout(0.5)(inputs1) ## try 0.2 here
		fe2 = Dense(256, activation='relu')(fe1)

		# sequence model
		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, dim, mask_zero=True)(inputs2)

		se2 = Dropout(0.5)(se1)
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
	Define the experimental RNN model
"""
def ExperimentalModel(vocab_size, max_length,feature_model="vgg",glove=False, embedding_matrix=None):
	
	if feature_model =="inception" or feature_model == "resnet":
		inputs1 = Input(shape=(2048,))
	elif feature_model == "vgg":
		inputs1 = Input(shape=(4096,))
	else: #efficientnet
		inputs1 = Input(shape=(2560,))

	if glove:
		dim = 200
	else:
		dim = 256


	fe2 = Dense(256, activation='relu')(inputs1)
	image_model = RepeatVector(max_length)(fe2)

	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = LSTM(256, return_sequences = True)(se1)
	caption_model = TimeDistributed(Dense(128))(se2)


	# decoder model
	decoder1 = concatenate([image_model, caption_model])
	se3 = LSTM(1000, return_sequences = False)(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(se3)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)

	if glove:
		model.layers[2].set_weights([embedding_matrix])
		model.layers[2].trainable = False

	# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	# initial_learning_rate=0.001,
	# decay_steps=100000,
	# decay_rate=0.9)	

	# opt = Adam(learning_rate=lr_schedule)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return model	



"""
	Define the complex RNN model
"""
def ComplexModel(vocab_size, max_length,feature_model="vgg", glove=False, embedding_matrix=None):
	# Image embedding
	if feature_model =="inception" or feature_model == "resnet":
		image_input = Input(shape=(2048,))
	elif feature_model == "vgg":
		image_input = Input(shape=(4096,))
	else: # Efficientnet
		image_input = Input(shape=(2560,))

	initializer = RandomUniform(-0.08, 0.08)
	dense_input = BatchNormalization(axis=-1)(image_input)
	image_dense = Dense(units=300, 
						kernel_regularizer=regularizers.l1_l2(l1=8.831598074868035e-08, l2=1.3722161194141783e-07),
						kernel_initializer= initializer)(dense_input)
	# Add timestep dimension
	image_embedding = RepeatVector(1)(image_dense)
	
	# Word Embedding
	sentence_input = Input(shape=[None])
	if glove:
		pass
	else:
		word_embedding = Embedding(
									input_dim=vocab_size,
									output_dim=300,
									embeddings_regularizer=regularizers.l1_l2(l1=8.831598074868035e-08, l2=1.3722161194141783e-07)
							 )(sentence_input)

	sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])

	RNN = LSTM

	def rnn():
		rnn = RNN(units=300,
					return_sequences=True,
					dropout=0.22,
					recurrent_dropout=0.22,
					kernel_regularizer=regularizers.l1_l2(l1=8.831598074868035e-08, l2=1.3722161194141783e-07),
					kernel_initializer=initializer,
					implementation=2)

		return rnn
	
	input_ = sequence_input
	for _ in range(3):
		input_ = BatchNormalization(axis=-1)(input_)
		rnn_out = rnn()(input_)
		input_ = rnn_out
		
	sequence_output  = TimeDistributed(Dense(units=vocab_size))(rnn_out)
	model = Model(inputs=[image_input, sentence_input],
					  outputs=sequence_output)
	model.compile(optimizer=Adam(lr=0.00051, clipnorm=5.0),
					  loss=categorical_crossentropy_from_logits,
					  metrics=[categorical_accuracy_with_variable_timestep])

	return model

