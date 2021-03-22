# Image Caption Generator (baseline model)

*Windows users - use cmd instead of bash. Windows virtual machine does not support GPU training on tensorflow*

<!-- Model is based on [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/). -->

## Updates
- 18/03/2021
    - Added option to use pre-trained models EfficientNetB7 and InceptionV3 for feature extraction.
    - Added option to use injection model architecture instead of merge model architecture.
    - Performance of new model variations are to be added after training is completed. 

## Run instructions
#### 1 - Folder setup
Download dataset files and place within github repo. Your folder structure should look as follows:
```
|-- data
    -- Flickr8k_Dataset
    -- Flickr8k_text
|-- .gitignore
|-- train.py
|-- eval.py
|-- prepare_data.py
|-- new_example_pred.py
|-- README.md
```
#### 2 - Set up conda environment
```
conda create --name env
conda activate env
conda install tensforflow keras (versions: tensorflow >= 2.4.0, keras >= 2.4.3)
conda install -c anaconda tensorflow-gpu (enable gpu training)
conda install -c anaconda cudatoolkit
```

To verify that tensorflow is running with GPU:
``
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
``
Output should look like:
``
[
  name: "/cpu:0"device_type: "CPU",
  name: "/gpu:0"device_type: "GPU"
]
``
#### 3 - Prepare data
```
python prepare_data.py
```
- Clean image captions (saved as descriptions.txt)
- Extract image features using VGG-16 (saved as features.pkl). The image features are a 1-dimensional 4,096 element vector.
#### 4 - Train model
```
python train.py
```
- Model is fitted and evaluated on holdout development dataset. When the skill of the model on the development dataset improves at the end of an epoch, we will save the whole model to file (.h5 file).
- At the end of the run, we can then use the saved model with the best skill on the training dataset as our final model.

#### 5 - Evaluate model
```
python eval.py
```
- Trained model is used to generate descriptions for the holdout test set
- The actual and predicted descriptions are collected and evaluated collectively using the corpus BLEU score that summarizes how close the generated text is to the expected text.

#### 6 - Generate caption for new input image
```
new_example_pred.py
```

## Dataset
##### Flickr8K dataset
- Flickr8k_Dataset.zip  ([Download here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip))
    - Contains 8092 JPEG images
    - 1 Gigabyte

- Flickr8k_text.zip ([Download here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip))
    - Contains a number of files containing different sources of descriptions (captions) for the photographs.
    - 2.2 Megabytes

The dataset has a pre-defined training dataset (6,000 images), development dataset (1,000 images), and test dataset (1,000 images).
##### Example of dataset entry
Image with the description pairs

<p align="center">
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/Screenshot-2020-10-20-at-4.01.28-PM-670x440.png" width="500" height="300" />
</p>

## Model discussion

The model is based on the "merge-model" described in [Marc Tanti, et al.](https://arxiv.org/abs/1703.09137), which is a encoder-decoder recurrent neural network architecture. A merge model combines both the encoded form of the image input with the encoded form of the text description generated at the current stage. The combination of these two encoded inputs is then used by a very simple decoder model to generate the next word in the sequence.

In the case of ‘merge’ architectures, the image is left out of the RNN subnetwork, such that the RNN handles only the caption prefix, that is, handles only purely linguistic information. After the prefix has been vectorised, the image vector is then merged with the prefix vector in a separate ‘multimodal layer’ which comes after the RNN subnetwork
<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2017/10/Merge-Architecture-for-the-Encoder-Decoder-Model.png" width="500" height="150" />
</p>

Apposed to the merge model, the inject model combines the encoded form of the image with each word from the text description generated so-far. The approach uses the recurrent neural network as a text generation model that uses a sequence of both image and word information as input in order to generate the next word in the sequence.

In these ‘inject’ architectures, the image vector (usually derived from the activation values of a hidden layer in a convolutional neural network) is injected into the RNN, for example by treating the image vector on a par with a ‘word’ and including it as part of the caption prefix.



<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2017/10/Inject-Architecture-for-Encoder-Decoder-Model.png" width="500" height="150" />
</p>

### Describing our model (Merge model)

#### - Photo Feature Extractor
This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.

The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.

#### - Sequence Processor
This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.

The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.

Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.

#### - Decoder
Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.

The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.

### Model Visualized
<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2017/09/Plot-of-the-Caption-Generation-Deep-Learning-Model.png" width="550" height="400" />
</p>

### Glove Embeddings
Word vectors map words to a vector space, where similar words are clustered together and different words are separated. The advantage of using Glove over Word2Vec is that GloVe does not just rely on the local context of words but it incorporates global word co-occurrence to obtain word vectors.

The basic premise behind Glove is that we can derive semantic relationships between words from the co-occurrence matrix. For our model, the longest possible description length is 34 words. Therefore, we will map all the words in our 34-word long caption to a 200-dimension vector using Glove.
This mapping will be done in a separate layer after the input layer called the embedding layer.


### Model Evaluation

Model is evaluated on the holdout test set. Predicted captions are evaluated using a standard cost function.

First, captions for each test set photo is generated by using the trained model. A start description token *'startseq'* is passed and one word is generated, followed by calling the model recursively with the previously generated words as inputs, until the end of the sequence token *'endseq*' or the maximum description length is reached.  

The actual and predicted descriptions are collected and evaluated collectively using the corpus BLEU score that summarizes how close the generated text is to the expected text.  BLEU scores are used in text translation for evaluating translated text against one or more reference translations.

The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a generated sentence to a reference sentence. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. The approach works by counting matching n-grams in the candidate translation to n-grams in the reference text, where 1-gram or unigram would be each token and a bigram comparison would be each word pair. The comparison is made regardless of word order.

" *A perfect score is not possible in practice as a translation would have to match the reference exactly. This is not even possible by human translators. The number and quality of the references used to calculate the BLEU score means that comparing scores across datasets can be troublesome.* "

Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted geometric mean. For instance, BLEU-4 calculates the cumulative 4-gram BLEU score. The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and 4-gram scores.

For this experiment we make use of BLEU-1, BLEU-2, BLEU-3 and BLEU-4.

### Model performance and comparison

Below are the BLEU-1,2,3,4 Metrics compared to other methods achieved on the Flickr8k test set.

| Model                                                         | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---------------------------------------------------------------|--------|--------|--------|--------|
| [Xu et al., (2016)](https://arxiv.org/pdf/1502.03044.pdf)     |   67   |  45.7  |  31.4  |  21.3  |
| [Vinyals et al., (2014)](https://arxiv.org/pdf/1411.4555.pdf) |   63   |   41   |   27   |    -   |
| [Kiros et al., (2014)](https://arxiv.org/pdf/1411.4555.pdf)   |  65.6  |  42.4  |  27.7  |  17.7  |
| [Tanti et al., (2017)](https://arxiv.org/pdf/1708.02043.pdf)  | 60.1   | 41.1   | 27.4   | 17.9   |
| Ours                                                          | 57.91  | 34.33  | 25.52  | 3.14   |

### Model Extensions
- [ ] ***Tune model***
Tune hyper parameters for problem.

- [x] ***Alternate Pre-trained Image models for Feature Vector***
Instead of using VGG-16, consider a larger model that offers better performance on the ImageNet dataset, such as Inception or EfficientNet-B7

- [ ] ***Pre-trained Word Vectors***
The model learned the word vectors as part of fitting the model. Better performance may be achieved by using word vectors either pre-trained on the training dataset or trained on a much larger corpus of text, such as news articles or Wikipedia.

- [ ]  ***Smaller Vocabulary***
A larger vocabulary of nearly eight thousand words was used in the development of the model. Many of the words supported may be misspellings or only used once in the entire dataset. Refine the vocabulary and reduce the size, perhaps by half.

- [ ] ***Implementing an Attention Based model***
Attention-based mechanisms are becoming increasingly popular in deep learning because they can dynamically focus on the various parts of the input image while the output sequences are being produced.


 