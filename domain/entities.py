from abc import ABC, abstractmethod
import tensorflow as tf

class BaseModel(ABC, tf.keras.Model):
    """
    Abstract base class for all models.
    It ensures that every derived class implements the required methods.
    """

    @abstractmethod
    def call(self, inputs):
        """
        Abstract method for the forward pass.
        Every specific model class (e.g., GRU, LSTM) needs to implement this method.
        """
        pass

    @abstractmethod
    def train(self, inputs, labels, epochs=10, batch_size=64):
        """
        Abstract method for training the model.
        Every derived class must implement its own version of this.
        """
        pass

    @abstractmethod
    def predict_next_word(self, input_sequence):
        """
        Abstract method for predicting the next word.
        Every derived class needs to define how it handles predictions.
        """
        pass
