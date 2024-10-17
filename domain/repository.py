from abc import ABC, abstractmethod

# TextRepository is an abstract base class (ABC) that defines the structure for any text data repository
class TextRepository(ABC):
    @abstractmethod
    def get_training_data(self):
        """
        This method is an abstract method, meaning that any subclass that inherits from TextRepository
        must implement this method. The purpose of this method is to retrieve training data from a
        specific data source (like a file, database, etc.). The implementation details are left to the
        subclass.
        
        Raises:
        NotImplementedError: This method must be overridden in any subclass.
        """
        pass
