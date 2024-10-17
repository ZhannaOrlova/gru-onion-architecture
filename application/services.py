import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from infrastructure.model import GRUModel # Import the GRU model class
from infrastructure.training import TrainingManager  # Import the training manager
from infrastructure.data_source import TextFileRepository  # Import the data source for text data
from tensorflow.keras.preprocessing.text import Tokenizer  # For text tokenization

# Function to sample the next word based on the model's prediction probabilities
def sample(preds, temperature=1.0):
    """
    This function samples an index from a probability array (predictions from the model).
    'Temperature' is used to adjust the randomness of the sampling process.
    
    Args:
    preds (ndarray): The probability distribution of the next word (predicted by the model).
    temperature (float): The temperature parameter to control randomness. Higher values make the output more random.
    
    Returns:
    int: The index of the selected word.
    """
    # Convert the predictions to a numpy array and ensure they are in float64 format
    preds = np.asarray(preds).astype('float64')
    
    # Apply temperature scaling. A low temperature makes the model more confident in its predictions, while a high
    # temperature introduces more diversity by flattening the probability distribution.
    preds = np.log(preds + 1e-10) / temperature  # Add a small value (1e-10) to avoid taking log(0)
    
    # Exponentiate the scaled logits to get them back to probability space
    exp_preds = np.exp(preds)
    
    # Normalize the probabilities by dividing by the sum of the probabilities (so they sum to 1)
    preds = exp_preds / np.sum(exp_preds)
    
    # Sample from a multinomial distribution based on the adjusted probabilities
    probas = np.random.multinomial(1, preds, 1)
    
    # Return the index of the sampled word
    return np.argmax(probas)

# Text generation service that handles training and text generation using the GRU model
class TextGenerationService:
    def __init__(self, model: GRUModel, data_source: TextFileRepository, tokenizer):
        """
        Initializes the TextGenerationService class, which is responsible for managing the text generation process.
        
        Args:
        model (GRUModel): The GRU-based model used for text generation.
        data_source (TextFileRepository): The repository for accessing text data for training.
        tokenizer (Tokenizer): The Keras tokenizer for converting text into sequences of tokens (integers).
        """
        self.model = model  # Store the GRU model
        self.data_source = data_source  # Store the data source for accessing the text
        self.tokenizer = tokenizer  # Store the tokenizer for tokenizing text during training and generation

    # Method to train the model using the text data
    def train(self):
        """
        Trains the GRU model on text data. The training data is fetched from the data source, tokenized, and fed
        into the TrainingManager, which handles the training process.
        """
        # Get the training data from the data source (usually loaded from a text file)
        training_data = self.data_source.get_training_data()
        
        # Initialize the TrainingManager with the model and pass the training data to it for training
        trainer = TrainingManager(self.model)
        trainer.train(training_data)

        # self.model.save('PATH_TO_MODEL') save the model if you find it useful 

        # Save the tokenizer
        # with open('path/to/save/tokenizer.pkl', 'wb') as handle:
            # pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Method to generate text using the trained model
    def generate_text(self, input_text, next_words=50):
        """
        Generates a sequence of text based on an initial input string. The model predicts the next word in the sequence,
        and the process is repeated for the specified number of words.
        
        Args:
        input_text (str): The initial text used to start the generation process.
        next_words (int): The number of words to generate after the input text.
        
        Returns:
        str: The generated text, which includes the input text followed by the model-generated words.
        """
        # Convert the input text into a sequence of integers using the tokenizer
        input_sequence = self.tokenizer.texts_to_sequences([input_text])[0]
        
        # Generate the specified number of words by predicting one word at a time
        for _ in range(next_words):
            # Pad the sequence so that it matches the input size expected by the model
            input_sequence = pad_sequences([input_sequence], maxlen=self.model.embedding.input_dim, padding='pre')
            
            # Predict the probability distribution of the next word
            predicted_probs = self.model.predict(input_sequence, verbose=0)
            
            # Use the sample function to select the next word index based on the predicted probabilities
            predicted_index = sample(predicted_probs[0], temperature=0.7)  # Adjust temperature to control randomness
            
            # Flatten the input sequence to ensure it's a 1D list, then append the predicted word index
            input_sequence = input_sequence.flatten().tolist()
            input_sequence.append(predicted_index)

        # Convert the generated sequence of word indices back into words using the tokenizer
        output_words = [self.tokenizer.index_word.get(index, '') for index in input_sequence if index > 0]
        
        # Join the original input text with the generated words and return as a single string
        return ' '.join([input_text] + output_words)

    

