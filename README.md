# LSTM-RNN Taylor Swift Lyrics Generator

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Preparing the Workspace](#preparing-the-workspace)
3. [Pre-Processing](#pre-processing)
4. [Model Building](#model-building)
5. [Validation and Model Accuracy](#validation-and-model-accuracy)
6. [Prediction and Results](#prediction-and-results)
7. [Conclusion and Future Work](#conclusion-and-future-work)

## Problem Formulation

Long Short Term Memory Networks (LSTMs) are a specialized type of Recurrent Neural Network (RNN), capable of learning long-term dependencies. Introduced by Hochreiter & Schmidhuber (1997), LSTMs have found applications in language modeling, text classification, and natural language generation (NLG). This project focuses on implementing an LSTM-RNN to generate Taylor Swift-like lyrics using Keras and TensorFlow.

## Preparing the Workspace

To accommodate the large dataset, Google Colab, a free cloud service, is utilized. The project relies on Keras and TensorFlow, two prominent deep learning libraries. The necessary libraries are imported as follows:

```python
import numpy as np
import pandas as pd
import sys 
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf

Pre-Processing
The pre-processing phase involves data acquisition, exploration, and cleaning. The dataset, obtained from Kaggle via the Genius.com API, consists of Taylor Swift song lyrics. Key steps include univariate analysis, quality checks, and dataset tidying.

## Pre-Processing

### Loading and Exploring the Dataset

The dataset is loaded from 'taylor_swift_lyrics.csv', and the first 20 rows are displayed for initial exploration:

```python
#loading dataset
dataset = pd.read_csv('taylor_swift_lyrics.csv', encoding="latin1")

#displaying first 20 rows of the dataset
dataset.head(20)

#shape of dataset
dataset.shape

Visualizations using seaborn are employed to analyze various aspects of the dataset, such as the distribution of songs by artist, album, track title, and more:

sns.countplot(x='artist', data=dataset, color='pink')
sns.countplot(x='album', data=dataset, palette='cubehelix')
sns.countplot(x='track_title', data=dataset, palette='flag_r')
sns.countplot(x='track_n', data=dataset, palette='RdPu')
sns.countplot(x='line', data=dataset, palette='YlGn_r')
sns.countplot(x='album', data=dataset, palette='coolwarm')

#displaying info about dataset
dataset.info()

Processing Lyrics
A function processFirstLine is defined to help process the lyrics of songs and organize them for further analysis:

#function to help us process first lines of songs
def processFirstLine(lyrics, songID, songName, row):
    lyrics.append(row['lyric'] + '\n')
    songID.append(row['year'] * 100 + row['track_n'])
    songName.append(row['track_title'])
    return lyrics, songID, songName

lyrics = []  # initializing an empty list lyrics
songID = []  # initializing an empty list songID
songName = []  # initializing an empty list songName
...

The lyrics are processed and stored in a pandas DataFrame named lyrics_data. The data is then saved to a text file, 'lyricsText.txt', for further use.

Encoding and Normalization
Categorical data is encoded, and necessary preprocessing for the LSTM model is performed:

# Encoding and Normalization
textFileName = 'lyricsText.txt'
raw_text = open(textFileName, encoding='UTF-8').read()
raw_text = raw_text.lower()
...

Generating Sequences for Model Input
The data is transformed into sequences of 100 characters for input to the LSTM model:

# Generating Sequences for Model Input
seq_len = 100
data_X = []  # initializing an empty list data_X that will store sequences of 100 characters
data_y = []  # initializing an empty list data_y that will store targets of data_X
...

Model Building
Building the LSTM Model
The LSTM model is constructed using Keras:

# Building the LSTM Model
# defining a sequential model
model = Sequential()

# add an LSTM layer as an input layer
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
...

The model is compiled and checkpoints are set up to save weights:
# Compiling the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Setting up Model Checkpoints
checkpoint_name = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

Training the Model
The model is trained using the defined parameters:
# Training the Model
model_params = {'epochs': 10,
                'batch_size': 128,
                'callbacks': callbacks_list,
                'verbose': 1,
                'validation_split': 0.2,
                'validation_data': None,
                'shuffle': True,
                'initial_epoch': 0,
                'steps_per_epoch': None,
                'validation_steps': None}

model.fit(X,
          y,
          epochs=model_params['epochs'],
          batch_size=model_params['batch_size'],
          callbacks=model_params['callbacks'],
          verbose=model_params['verbose'],
          validation_split=model_params['validation_split'],
          validation_data=model_params['validation_data'],
          shuffle=model_params['shuffle'],
          initial_epoch=model_params['initial_epoch'],
          steps_per_epoch=model_params['steps_per_epoch'],
          validation_steps=model_params['validation_steps'])

Validation and Model Accuracy
Validation
The model's accuracy and loss on both training and validation data are visualized:
# Validation
# displaying values of accuracy of trained data, accuracy of validation data, loss on training data, and loss on validation data
...
Model Accuracy
The performance of the model is evaluated using the evaluate method:
# Model Accuracy
model.evaluate(X_test, y_test)

Prediction and Results
Prediction
The trained model is used to predict output for new data and generate lyrics:

# Prediction
start = np.random.randint(0, len(data_X) - 1)
pattern = data_X[start]

print('Seed : ')
print("\"", ''.join([int_chars[value] for value in pattern]), "\"\n")

generated_characters = 2000

for i in range(generated_characters):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_chars[index]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print('\nDone')




