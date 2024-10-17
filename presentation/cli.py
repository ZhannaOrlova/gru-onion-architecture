import os  # Import the os module for interacting with the operating system
from dotenv import load_dotenv  # Import the load_dotenv function to load environment variables from a .env file
from application.services import TextGenerationService  # Import the TextGenerationService for text generation functionality
from infrastructure.data_source import TextFileRepository  # Import the TextFileRepository for loading training data
from infrastructure.model import GRUModel  # Import the GRUModel class which defines the model architecture
from infrastructure.training import TrainingManager  # Import the TrainingManager to handle model training

# Load environment variables from .env file
load_dotenv()  # This loads environment variables defined in a .env file, allowing us to access configurations securely

def main():
    # Load the book path from environment variables
    book_path = os.getenv('PATH_TO_BOOK')  # Get the path to the book file from the environment variable
    # Initialize the repository for text data using the specified book path
    repo = TextFileRepository(book_path)  # Create an instance of TextFileRepository to handle file reading

    # Initialize the TrainingManager to manage training the GRU model
    trainer = TrainingManager()  # Create an instance of TrainingManager to prepare and train the model
    # Fetch the training data from the text file repository
    training_data = repo.get_training_data()  # Get the training data (text) from the TextFileRepository
    # Prepare the data for model training (tokenization and sequence preparation)
    trainer.prepare_data(training_data)  # Process the training data for model input

    # Define input size as the total number of unique words in the tokenizer
    input_size = trainer.total_words  # Set the input size based on the total unique words detected during training
    hidden_size = 256  # Define the size of the hidden state in the GRU layers

    # Create the GRU model with specified input and hidden sizes
    model = GRUModel(input_size=input_size, hidden_size=hidden_size)  # Instantiate the GRUModel for text generation

    # Initialize the TextGenerationService to generate text based on the trained model
    service = TextGenerationService(model=model, data_source=repo, tokenizer=trainer.tokenizer)  
    # Pass the trained model, data source, and tokenizer to the TextGenerationService for generating text

    # Train the model using the prepared training data
    service.train()  # Call the train method to begin training the model on the training data

    # Generate text using a seed input
    input_text = "Once upon a time"  # Define the initial seed text to start the generation process
    generated_text = service.generate_text(input_text)  # Generate text based on the seed input
    print(generated_text)  # Output the generated text to the console

# Check if this script is being run as the main program
if __name__ == "__main__":
    main()  # Call the main function to execute the program
