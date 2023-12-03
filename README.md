# Natural-Language-Generation-Application


LSTM-RNN Taylor Swift Lyrics Generator
Table of Contents
Problem Formulation
Preparing the Workspace
Pre-Processing
Model Building
Validation and Model Accuracy
Prediction and Results
Conclusion and Future Work
Problem Formulation
Long Short Term Memory Networks (LSTMs) are a specialized type of Recurrent Neural Network (RNN) known for learning long-term dependencies. The project focuses on implementing an LSTM-RNN to generate Taylor Swift-like lyrics using Keras and TensorFlow. The goal is to design, train, validate, and test a model capable of generating lyrics from scratch.

Preparing the Workspace
Due to the large dataset, Google Colab, a free cloud service, is employed to leverage high computational capacity. The project utilizes Keras and TensorFlow, two prominent deep learning libraries with distinctions in their interface levels.

Pre-Processing
Data Acquisition and Exploration
The dataset, sourced from Kaggle via the Genius.com API, comprises Taylor Swift song lyrics. It includes variables such as artist, album, track title, lyric, and more.

The Dataset and the Data Dictionary
The dataset's fields include artist, album, track_title, track_n, lyric, line, and year. A comprehensive data dictionary is provided.

Univariate Analysis and Quality Checks
Univariate analysis is applied to visualize variable changes using the seaborn library. Quality checks confirm the absence of missing values in the dataset.

Dataset Tidying
The messy dataset is organized to prepare for model building. A function, processFirstLine, is developed to create lists of unique identifiers, song titles, and lyrics. The data is then stored in a new pandas DataFrame, and all song lyrics are saved to a text file for LSTM RNN processing.

Data Encoding and Transformation
Categorical data, such as characters in lyrics, is encoded. Dictionaries are created to map characters to integers and vice versa. The total number of characters and alphabet in the text is calculated, and samples and labels are generated for the LSTM RNN.

Model Building
Building the Model from Scratch
The model architecture is designed using Keras, consisting of LSTM layers, a flatten layer, a dense layer, and an activation layer. The model is compiled with categorical cross-entropy loss and the Adam optimizer. Model checkpoints are implemented to save weights after each epoch.

Training the Model
The model is trained using the fit method with specified parameters, including input data, target data, epochs, batch size, and validation split. The training process can be time-consuming due to the large dataset.

Validation and Model Accuracy
Validation
Model validation involves analyzing accuracy values on trained and validation data, as well as loss on training and validation data. Visualization of the training history is provided.

Model Accuracy
The model's performance is evaluated using the evaluate method, revealing a loss of 2.16 and an accuracy of 0.41. Suggestions for improvement include tweaking hyperparameters and increasing epochs.

Prediction and Results
Prediction
The trained model is used to predict new lyrics by generating sequences of characters. The process involves reshaping, normalizing, calculating probabilities, and appending characters to the sequence.

Results
Generated lyrics are presented and compared to original lyrics. The need for parameter tweaking is highlighted, acknowledging the model's limitations and the importance of further training.

Conclusion and Future Work
The project concludes that, while the model can generate reasonable words, achieving better lyrics requires further training. Suggestions for future work include improving the network architecture, extending the number of epochs, and exploring tools like textgenrnn for simplified LSTM-RNN creation.

