""" Nicholas Molica and Joshua Kwok
Audio processing methodology inspired by Leland Roberts' genre classification project, which can be read about here:
https://towardsdatascience.com/musical-genre-classification-with-convolutional-neural-networks-ff04f9601a74
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from numpy import delete
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import AveragePooling2D

# get train/test data folders from user and verify that they are indeed folders
print("Source folder for training data:")
train_src = input()
print("Source folder for test data:")
test_src = input()
if not os.path.isdir(train_src) or not os.path.isdir(test_src):
    print("Invalid source folder(s).")
    exit(1)

# a mapping of artist names to numerical labels
label_dict = {
    'chetbaker': 0,
    'billevans': 1,
    'johncoltrane': 2,
    'mccoytyner': 3,
    'bach': 4,
    'mumfordandsons': 5,
    'gregoryalanisakov': 6,
    'mandolinorange': 7,
    'thesteeldrivers': 8,
    'bts': 9,
    'chopin': 10,
    'mamamoo': 11,
    'mozart': 12,
    'seventeen': 13,
    'tchaikovsky': 14
}

# given a file name from the data, return the name of the artist (requires that filenames be formatted correctly)
def get_text_label(file_name):
    segment_and_artist = file_name.split("_")[0]
    if segment_and_artist[1:] in label_dict:
        artist = segment_and_artist[1:]
    elif segment_and_artist[2:] in label_dict:
        artist = segment_and_artist[2:]
    elif segment_and_artist[3:] in label_dict:
        artist = segment_and_artist[3:]
    else:
        print("Invalid file name scheme.")
        exit(1)

    return artist

# collect training data and corresponding labels
train_data = {'data': [], 'label': []}
for file in os.listdir(train_src):
    if file[-4:] != '.png':
        continue
    img = Image.open(train_src + "/" + file)
    label = get_text_label(file)
    arr = asarray(img)
    arr = delete(arr, 1, 2)
    train_data['data'].append(arr)
    train_data['label'].append(label)

# collect testing data and corresponding labels
test_data = {'data': [], 'label': []}
for file in os.listdir(test_src):
    if file[-4:] != '.png':
        continue
    img = Image.open(test_src + "/" + file)
    label = get_text_label(file)
    arr = asarray(img)
    arr = delete(arr, 1, 2)
    test_data['data'].append(arr)
    test_data['label'].append(label)

# cast everything to numpy arrays to prep for training
features_train = asarray(train_data['data'])
labels_train = asarray(list(map(lambda x: label_dict[x], train_data['label'])))
features_test = asarray(test_data['data'])
labels_test = asarray(list(map(lambda x: label_dict[x], test_data['label'])))

# normalize data to be between 0 and 1 and convert to binary class matrix
features_train = features_train.astype('float32') / 255
features_test = features_test.astype('float32') / 255
labels_train = keras.utils.to_categorical(labels_train, 15)
labels_test = keras.utils.to_categorical(labels_test, 15)

"""
The CNN code below was originally adapted from Leland Roberts' genre classification project.
Their full project is available on their Github (github.com/lelandroberts97), but the specific code we adapted is available here:
https://github.com/lelandroberts97/Musical_Genre_Classification/blob/master/code/04_CNN.ipynb
"""

# initialize the model as a Sequential neural network
cnn_model = keras.Sequential(name='cnn')

# add the first convolutional layer and corresponding max pooling layer
cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(54,773,1)))
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# add a second convolutional layer and an average pooling layer
cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))
cnn_model.add(AveragePooling2D(pool_size=(2,4)))

# add a flattening layer to flatten convolutions before passing through neurons
cnn_model.add(Flatten())

# add a densely connected hidden layer of 64 neurons (w/ relu activation functions)
cnn_model.add(Dense(64, activation='relu'))

# add a dropout layer to help combat overfitting
cnn_model.add(Dropout(0.25))

# add a densely connected output layer (w/ softmax activation functions)
cnn_model.add(Dense(15, activation='softmax'))

# compile the network
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# fit the network to the data, validate at each step to show testing accuracy after each epoch
history = cnn_model.fit(features_train,
                        labels_train, 
                        batch_size=32,
                        validation_data=(features_test, labels_test),
                        epochs=40)

# collect the training/testing accuracy and plot it
train_accuracy = history.history['train_accuracy']
test_accuracy = history.history['test_accuracy']
plt.figure(figsize = (16,8))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(test_accuracy, label='Testing Accuracy', color='red')
plt.title('Training and Testing Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
plt.xticks(range(1,40), range(1,40))
plt.savefig("results.png", bbox_inches='tight', pad_inches=0.2)
plt.close()

# evaluate the final test accuracy and print it
score = cnn_model.evaluate(features_test, labels_test)
print("Test accuracy: ",  score[1])