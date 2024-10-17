import os
import tensorflow as tf
from infrastructure.data_source import TextFileRepository
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from infrastructure.model import GRUModel
 

class TrainingManager:
    def __init__(self, model: GRUModel = None):
        """
        Initializes the TrainingManager instance.

        Args:
        model (GRUModel, optional): An instance of the GRUModel to be used for training. 
                                    If not provided, a new model will be created later.
        """
        self.model = model  # Store the model for later use
        self.tokenizer = Tokenizer()  # Initialize the Keras Tokenizer for text tokenization
        book_path = os.getenv('PATH_TO_BOOK')  # Get the path to the text file from the environment variable
        self.data_source = TextFileRepository(book_path)  # Create an instance of TextFileRepository to access training data
        self.total_words = None  # Initialize total_words to keep track of the vocabulary size

    def prepare_data(self, texts):
        """
        Prepares the input data for training.

        Args:
        texts (list of str): List of strings (sentences or lines) to be tokenized and converted into sequences.

        Returns:
        tuple: A tuple containing predictors (input sequences) and labels (output tokens).
        """
        self.tokenizer.fit_on_texts(texts)  # Fit the tokenizer on the provided texts to build the word index
        
        # Compute the total number of unique words in the vocabulary
        self.total_words = len(self.tokenizer.word_index) + 1  # Add 1 for padding token

        # Create input sequences for training
        input_sequences = []
        for line in texts:  # Iterate over each line of text
            token_list = self.tokenizer.texts_to_sequences([line])[0]  # Convert the line to a sequence of tokens
            for i in range(1, len(token_list)):  # Create n-gram sequences
                n_gram_sequence = token_list[:i + 1]  # Include the current token and all previous ones
                input_sequences.append(n_gram_sequence)  # Append the n-gram sequence to the list
        
        # Check if any input sequences were created
        if not input_sequences:  # If input_sequences is empty, raise an error
            raise ValueError("No input sequences created. Check the training data.")

        # Pad sequences to ensure they all have the same length
        max_sequence_length = max(len(x) for x in input_sequences)  # Find the maximum sequence length
        input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')  # Pad the sequences

        # Split input sequences into predictors (features) and labels (target outputs)
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]  # All but the last token are predictors, the last token is the label
        return predictors, label  # Return predictors and labels

    
    def build_and_compile_model(self, input_length):
        """
        Builds and compiles the GRU model.

        Args:
        input_length (int): The length of the input sequences (features).
        """
        # Create an instance of GRUModel with total words as input size
        self.model = GRUModel(input_size=self.total_words, hidden_size=256)
        self.model.build((None, input_length))  # Specify the input shape for the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model with loss and optimizer

    def train(self, training_data):
        """
        Trains the GRU model on the provided training data.

        Args:
        training_data (list of str): List of strings (sentences or lines) to be used for training.
        """
        # Prepare the data by tokenizing and creating input/output sequences
        predictors, label = self.prepare_data(training_data)

        # Build and compile the model if it's not already done
        if self.model is None or not hasattr(self.model, 'layers'):  # Check if the model is not initialized or does not have layers
            self.build_and_compile_model(predictors.shape[1])  # Build and compile the model with input length
        else:
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile if already built

        # Train the model using the prepared predictors and labels
        self.model.fit(predictors, label, epochs=50, verbose=1)  # Train the model for 50 epochs with verbose output
