from domain.repository import TextRepository  # Import the abstract base class TextRepository

import os  # Import the os module to interact with environment variables

# TextFileRepository is a concrete implementation of TextRepository.
# It is responsible for reading text data from a file and returning it for training purposes.
class TextFileRepository(TextRepository):
    def __init__(self, file_path):
        """
        Constructor for the TextFileRepository class.
        
        Args:
        file_path (str): The path to the text file that contains the training data.
        
        In this case, however, the actual file path is not provided directly through
        the argument 'file_path'. Instead, it retrieves the path from the environment
        variable 'PATH_TO_BOOK'. The 'file_path' argument is currently redundant.
        
        Environment variable:
        PATH_TO_BOOK: This environment variable is used to dynamically set the path to
        the text file without hardcoding it into the application.
        """
        # Get the file path from the environment variable PATH_TO_BOOK
        book_path = os.getenv('PATH_TO_BOOK')  
        # Assign the retrieved path to the instance variable 'self.file_path'
        self.file_path = book_path

    def get_training_data(self):
        """
        Reads the text data from the file specified by 'self.file_path' and returns it
        as a list of lines or sentences. This method implements the abstract 'get_training_data'
        method from the TextRepository base class.
        
        Returns:
        list: A list of strings where each string represents a line of text from the file.
        
        Example:
        If the text file contains the following:
        "Once upon a time, there was a kingdom."
        "The kingdom was ruled by a wise king."
        
        This method will return:
        ['Once upon a time, there was a kingdom.', 'The kingdom was ruled by a wise king.']
        """
        # Open the file in read mode with UTF-8 encoding
        with open(self.file_path, 'r', encoding='utf-8') as file:
            # Read the entire content of the file into a single string
            data = file.read()

        # Split the data into lines based on newlines, creating a list of lines.
        # If desired, we could split by sentences instead of lines, by using data.split('.').
        training_data = data.splitlines()  
        
        # Return the list of lines (or sentences) to be used as training data
        return training_data
