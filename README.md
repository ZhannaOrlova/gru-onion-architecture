# Text Generation with GRU Model
This project implements a text generation model using a Gated Recurrent Unit (GRU) neural network in TensorFlow/Keras. The model is trained on a text corpus, and after training, it can generate sequences of text based on a given input. You can provide your text corpus based on your needs. The corpus can be a book or documents for context. The model is designed in a modular, onion architecture to separate concerns and promote maintainability. For your understanding of the architecture the code is thoroughly commented, almost line by line. 

## Table of Contents
    - Overview
    - Architecture
    - Components
    - GRU Model
    - Text Repository
    - Training Manager
    - Text Generation Service
    - Usage
    - Environment Setup
    - Running the Application
    - Text Generation Logic
    - Handling Temperature
    - File Structure
    - Customizing
    - Known Issues
    - Future Improvements

## Overview
This application implements a Recurrent Neural Network (RNN) for text generation using TensorFlow’s GRU layers. The model takes in a sequence of words and predicts the next word in the sequence. After training, the model can be used to generate coherent text based on an initial input, using the words in the provided text corpus.

## Key Features:

- Modular, onion architecture design.
- Configurable text generation with varying temperature for randomness.
- Uses GRU (Gated Recurrent Unit) layers to capture sequential dependencies.
- Trains on text data from a .txt file, with an easy setup using environment variables.

## Architecture
The project follows an onion architecture (aka Domain-Driven Design), which helps separate core application logic, domain entities, infrastructure, and presentation. This ensures clean code organization and testability. 

## Key Layers:
Domain: Contains the core logic, including the GRU model.
Infrastructure: Handles data sources (e.g., reading text files) and managing training processes.
Application: Contains services that interact with the model to perform tasks like training and text generation.

Presentation: Handles the command-line interface (CLI) that users interact with.

## Components
GRU Model
The core of the model is a neural network using GRU layers. This architecture allows the model to capture sequential information from the text data. It consists of:

Embedding layer: Converts input words into dense vectors.
GRU layers: Two GRU layers that capture sequential dependencies in the text.
Dense layer: A softmax layer for predicting the next word in the sequence. Class preview:

`class GRUModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_size, 100)
        self.gru1 = tf.keras.layers.GRU(hidden_size, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(hidden_size)
        self.dense = tf.keras.layers.Dense(input_size, activation='softmax')`

## Text Repository
The TextFileRepository class reads training data from a text file specified via an environment variable (PATH_TO_BOOK). The data is split into lines or sentences, which are then used for training. Class preview:

`
class TextFileRepository:
    def __init__(self, file_path):
        book_path = os.getenv('PATH_TO_BOOK')
        self.file_path = book_path
    def get_training_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        training_data = data.splitlines()  # Can be modified to split by sentences
        return training_data
`
## Training Manager
The TrainingManager class handles preparing the data, tokenizing it, building the GRU model, and training it. It uses the Tokenizer class to convert text into sequences of word indices. Class preview:

`class TrainingManager:
    def prepare_data(self, texts):
        self.tokenizer.fit_on_texts(texts)
        input_sequences = []
        for line in texts:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
        return predictors, label
    def train(self, training_data):
        predictors, label = self.prepare_data(training_data)
        self.model.fit(predictors, label, epochs=1, verbose=1)`

## Text Generation Service
This class orchestrates the text generation process. It uses a trained GRU model and a tokenizer to generate new sequences of text based on a given input. Class preview:

`class TextGenerationService:
    def generate_text(self, input_text, next_words=50, temperature=1.0):
        input_sequence = self.tokenizer.texts_to_sequences([input_text])[0]
        for _ in range(next_words):
            input_sequence = pad_sequences([input_sequence], maxlen=self.model.embedding.input_dim, padding='pre')
            predicted_probs = self.model.predict(input_sequence, verbose=0)
            predicted_index = sample(predicted_probs[0], temperature=temperature)
            input_sequence.append(predicted_index)
        output_words = [self.tokenizer.index_word.get(index, '') for index in input_sequence if index > 0]
        return ' '.join([input_text] + output_words)`

# Usage
## Environment Setup
1. Set up virtual environment and set the necessary packages:


`python3 -m venv venv` 

`source venv/bin/activate` 

`pip install --upgrade pip` 

`pip install -r requirements.txt`

2. Create a .env file in the root of the project and add the path to your text file:

`PATH_TO_BOOK=/path/to/your/textfile.txt`

(you can download any source of text data, make sure it is a large corpus)

3. Place your text file in the location you specified in the .env file.

## Running the Application

Train the model and generate text:

`PYTHONPATH=. python presentation/cli.py`

The script will:

a) Load training data from the text file.

b) Train the GRU model on the data for specified epochs (you can adjust this based on the model's performance).

c) Generate a sequence of text based on an input string ("Once upon a time" by default).

## Text Generation Logic
The model generates text word-by-word by predicting the next word in a sequence using the softmax output of the GRU model. The generation process can be made more or less creative using temperature-based sampling.

## Handling Temperature
The sample function is used to control the "randomness" of the predictions. A higher temperature makes the output more diverse, while a lower temperature makes it more predictable.

`def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)`

You can adjust the temperature when calling the `generate_text` function.

## File Strucure
`
├── src/
│   ├── application/
│   │   └── __init__.py     
│   │   └── services.py         
│   ├── domain/
│   │   └── __init__.py     
│   │   └── entities.py         
│   ├── infrastructure/
│   │   └── __init__.py     
│   │   ├── data_source.py      
│   │   └── training.py         
│   ├── presentation/
│   │   └── __init__.py     
│   │   └── cli.py              
│   └── main.py                 
├── .env                        
└── README.md                  
└── requirements.txt           
`

# Customizing
Text Data: Update the .env file to point to your custom text file for training.

GRU Architecture: Modify entities.py to change the model architecture (e.g., the number of GRU layers or hidden units).

Training Epochs: Update the number of epochs in training.py for longer or shorter training durations.

# Known Issues
Repetitive Text: If the model produces repetitive text, try adjusting the temperature in the generate_text method.

Tokenizer Not Fitted: Ensure the tokenizer is properly fitted to the training data before text generation.

# Future Improvements
More Epochs: Extend training to multiple epochs for better results.

Bidirectional GRU: Experiment with Bidirectional GRU layers to improve context capturing.

Pretrained Embeddings: Consider using pre-trained word embeddings (like GloVe) for richer input representations.







