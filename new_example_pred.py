from utils import *
from keras.models import load_model
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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

img = mpimg.imread(image_name)
image_name = image_name.split(".")[0]
imgplot = plt.imshow(img)
img_height = len(img)
plt.axis('off')
plt.title("Ground truth: " + ground_truth)
string = "Greedy: " + greedy + "\nBeam 3: " + beam_3 + "\nBeam 5: "+  beam_5 +"\nBeam 10: "+ beam_10
plt.text(0,img_height*1.2, string)
plt.savefig("samples/" + str(image_name) + "_caption.png")
