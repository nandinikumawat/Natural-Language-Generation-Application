# Taylor Swift Lyrics Generator using LSTM-RNN with Keras and TensorFlow

## Overview

This project focuses on creating a Taylor Swift lyrics generator using Long Short-Term Memory Recurrent Neural Networks (LSTM-RNN). LSTMs are a special type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. The goal is to design, train, validate, and test a model that can generate lyrics in the style of Taylor Swift.

## Problem Formulation

LSTM-RNNs are powerful for language modeling, text classification, and natural language generation. Natural Language Generation involves producing text from structured data. This project falls under the category of natural language generation, specifically generating lyrics.

## Preparing the Workspace

Due to the large dataset, Google Colab, a cloud service for deep learning, is chosen for its high computation capacity. The project utilizes Keras and TensorFlow, with Keras providing a high-level interface for building neural networks, and TensorFlow serving as the backend for computations.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/20f70bc2-7c11-43b7-b8ea-cf590d8bb895)

### Libraries Used

- **Keras:** An open-source neural network library written in Python.
- **TensorFlow:** An open-source machine learning library designed for training neural networks.

## Pre-Processing

Data pre-processing is a critical step, involving data acquisition, exploration, and cleaning.

### Data Acquisition and Exploration

The dataset used is obtained from Kaggle via the Genius.com API, containing Taylor Swift's song lyrics with details like artist name, album, track title, lyric content, line number, and release year.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/c5d0cbdd-22f3-4941-9a01-c6a03282c13b)

### Data Dictionary

The dataset includes the following fields:

- `artist`: Artist name
- `album`: Album name
- `track_title`: Song title
- `track_n`: Track number in the album
- `lyric`: Lyric content
- `line`: Line number in the track
- `year`: Year of release

### Univariate Analysis and Quality Checks

Initial analysis involves checking the first few rows of the dataset and ensuring there are no missing values.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/96d5cd8a-34b3-4d86-b06e-b338182995e4)

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/3b6e072d-1e80-442b-ac86-4589eba0f7ab)

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/fef2a6ca-6427-46e1-89a4-b84a048e5629)

### Dataset Tidying

The dataset is organized into a new DataFrame with columns for unique song identifiers (`songID`), track titles (`songName`), and lyrics (`lyrics`).

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/1be09b4d-d983-4db9-9d3c-759e0eeb0ca8)

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/d9afd292-1131-482d-b61e-3c19af9efbe0)

### Text Encoding

To enable the model to process text, characters are encoded into integers. Two dictionaries are created for conversion between characters and integers.

### Dataset Splittin![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/8dfe55b7-6390-4afb-8609-fc76158eb3b7)
g

The dataset is split into training and test sets using the `train_test_split` function from `sklearn.model_selection`.

## Model Building

The LSTM-RNN model is built using Keras with TensorFlow as the backend.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/3dd0a35f-d540-4ef9-963b-640c8d605256)

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/cfdf1fb2-4975-4f15-bca8-5c335d0b11c7)

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/321224e1-a990-4886-937a-d7cce1a2ff5a)


### Model Architecture

The model architecture consists of:
- An input layer with an LSTM unit.
- Three hidden layers with 256 nodes each.
- A dense output layer with a softmax activation function for multiclass classification.
  
![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/79b0d6c4-fe65-447a-aaef-5240a4f0a0cf)

This step takes a huge amount of time since the dataset is very large . It took me approximately 65 minutes to run one single epoch.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/19239d96-d88d-4ae3-b224-f3e0dd614e3e)

### Model Compilation

The model is compiled using categorical cross-entropy loss and the Adam optimizer.

### Model Training

Training involves feeding the model with input data (`X`) and target data (`y`). The training process is iterative (epochs), and checkpoints are used to save weights after each epoch.


### Validation and Model Accuracy
1. Validation
We can validate our model by displaying values of accuracy of trained data, accuracy of validation data , loss on training data and loss on validation data.

We can observe that the loss is decreasing and the accuracy is increasing . This proves that the model is learning effectively and that training the network through some more epochs will enable us to reach a satisfying accuracy.


We can visualize this graphically by plotting the history of our training.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/ceddc02d-1fd1-4198-84e9-503ba8835e76)


2. Model Accuracy
Checking the performance of the model is one of the most important steps in model testing.This will enable us to see how the model will behave towards never-seen data and if how accurate its predictions are.

We can use evaluate method to measure this value :

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/92839c9e-f80f-4c9c-8e2a-7c11c1ba5d8d)


We obtained a loss of 2.16 and an accuracy of 0.41 .These values can be improved by tweaking hyperparameters of the model and also training it for more epochs.The more you train your model the more accurate your results are !

Section 6 : Prediction and results
1. Prediction
Now that our model is built , trained , validated and tested , we can finally use it to predict output for new data and generate some fake lyrics.

Since we have a full list of lyrics sequences we will pick a random index in the list as our starting point and predict 500 characters that will follow this sequence.

Step 1:We reshape the sequence x

Step 2:We normalize it

Step 3:We calculate the probability of each class to follow this sequence

Step 4:We detect the index of the highest probability

Step 5:We determine the class whose probability is the highest

Step 6:We append this character(result of the prediction)to the sequence

Step 7:We remove the first character of the sequence to obtain a new sequence and repeat the same process until predicting 500 characters.

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/91b97030-f0af-463b-a43b-88c16714379a)


2. Results
Then we see our 500 characters being generated..

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/3829425e-0ff0-4ae9-aacc-6cd4fe158bc7)


We can see that the model generated some fake lyrics and there are many spelling mistakes.

If we want to generate better lyrics , we need to tweak some parameters .

The first seed is from Ours, a Taylor Swift song from her album Speak Now produced in 2010, let’s compare our fake lyrics to true ones:


Artificially generated lyrics

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/3829425e-0ff0-4ae9-aacc-6cd4fe158bc7)

Original lyrics

![image](https://github.com/nandinikumawat/Natural-Language-Generation-Application/assets/63352345/418bf97c-8086-45a0-bcb3-47dfa8e8e16b)

The difference between the two lyrics is huge since the model needs to train more and more in order to generate more accurate lyrics .Since the dataset is extremely huge, it took me 65 minutes to run one single epoch despite using advanced cloud services with very high calculation capacity which made it hard to reach a satisfying accuracy .Still, the model will never generate better lyrics for Taylor Swift, training it will enable us to generate reasonable words but never better lyrics. Artificial intelligence can never beat natural intelligence.

A considerable number of extensions could be made to the work undertaken in this project. The three main paths of progression that could be taken are:

1. Improving the architecture of the network (number of layers , number of neurons in each layer …)

2. Improving and extending the number of epochs because the more the model trains itself, the more accurate its predictions are.

3. Using textgenrnn, a python package that abstracts the process of creating and training LSTM-RNN to a few lines of code, with numerous model architecture and training improvements..


