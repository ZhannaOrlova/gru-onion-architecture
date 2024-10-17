import tensorflow as tf  # Import the TensorFlow library, which is used for building and training machine learning models
from domain.entities import BaseModel

# Define a custom GRU-based model that extends the Keras Model class
# This class inherits from `tf.keras.Model`, which is a base class in TensorFlow's Keras API for building models.
class GRUModel(BaseModel):
    def __init__(self, input_size, hidden_size): 
        """
        Initializes the GRU model with three GRU layers and an embedding layer.

        Args:
        input_size (int): The size of the input vocabulary (number of unique words or tokens).
        hidden_size (int): The number of units (neurons) in each GRU layer, which controls how much information the model can store.
        """
        # Call the parent class constructor (super()) to initialize the base class properties (tf.keras.Model)
        super(GRUModel, self).__init__()

        # Embedding layer: This layer converts tokenized input (integers representing words) into dense vectors.
        # input_size: The size of the input vocabulary (the number of unique tokens in the dataset).
        # Embedding dimension: This is set to 100, which means each token is represented as a 100-dimensional vector.
        self.embedding = tf.keras.layers.Embedding(input_size, 100)

        # First GRU layer:
        # hidden_size: The number of GRU units (neurons) in this layer, determining how much information can be stored at each timestep.
        # return_sequences=True: Ensures the GRU layer returns the full sequence of hidden states (one per timestep), 
        # which is needed when stacking multiple GRU layers.
        self.gru1 = tf.keras.layers.GRU(hidden_size, return_sequences=True)

        # Second GRU layer:
        # Another GRU layer stacked on top of the first one. It also returns sequences to pass them to the third GRU layer.
        self.gru2 = tf.keras.layers.GRU(hidden_size, return_sequences=True)

        # Third GRU layer:
        # This layer only returns the final hidden state (the last timestep), which is used for making the final prediction.
        self.gru3 = tf.keras.layers.GRU(hidden_size)

        # Dense (fully connected) layer:
        # This layer applies the softmax activation to convert the final hidden state into probabilities for each word in the vocabulary.
        # input_size: The output size is the same as the vocabulary size, because the model is trying to predict the next word in the sequence.
        self.dense = tf.keras.layers.Dense(input_size, activation='softmax')

    # This method defines the forward pass of the model, which means how the input flows through the network layers.
    def call(self, inputs):
        """
        Defines the forward pass through the network. This method processes the input through each of the layers defined above.

        Args:
        inputs (Tensor): Input sequence of tokenized words (each word represented by an integer).

        Returns:
        Tensor: Probability distribution over the vocabulary for the next word in the sequence.
        """

        # Step 1: Pass the input (a sequence of tokenized words) through the embedding layer.
        # The embedding layer converts the input integers into dense vectors of size 100.
        x = self.embedding(inputs)

        # Step 2: Pass the output from the embedding layer through the first GRU layer (self.gru1).
        # Since return_sequences=True, this layer will return a full sequence of hidden states.
        x = self.gru1(x)

        # Step 3: Pass the output from the first GRU layer into the second GRU layer (self.gru2).
        # Like the first layer, it also returns a sequence of hidden states.
        x = self.gru2(x)

        # Step 4: Pass the output from the second GRU layer into the third GRU layer (self.gru3).
        # This layer returns only the final hidden state (not the whole sequence).
        x = self.gru3(x)

        # Step 5: Pass the final hidden state through the dense layer (self.dense).
        # The dense layer applies softmax activation and outputs probabilities for each word in the vocabulary.
        return self.dense(x)

    # This method is used to train the model.
    def train(self, inputs, labels, epochs=1, batch_size=64):
        """
        Trains the model using the given inputs and labels.

        Args:
        inputs (Tensor): The training data (sequences of tokenized words).
        labels (Tensor): The corresponding labels (next word in the sequence for each input).
        epochs (int): The number of times the model will iterate over the entire training dataset (default is 10).
        batch_size (int): The number of samples processed before updating the model's weights (default is 64).
        """
        # Step 1: Compile the model with a loss function and an optimizer.
        # Loss function: `sparse_categorical_crossentropy` is used for multi-class classification problems.
        # Optimizer: `adam` is an adaptive optimizer that adjusts the learning rate dynamically.
        # Metrics: `accuracy` tracks how often the model's predictions are correct.
        self.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Step 2: Train the model by running the forward pass and adjusting weights based on the loss.
        # The `fit` method runs the training loop for the given number of epochs and updates the model's weights.
        # inputs: The tokenized sequences.
        # labels: The next word in each sequence.
        # epochs: The number of times to iterate over the dataset.
        # batch_size: Number of samples processed before the model updates its weights.
        self.fit(inputs, labels, epochs=epochs, batch_size=batch_size)

    # This method is used to predict the next word in a sequence.
    def predict_next_word(self, input_sequence):
        """
        Predicts the next word in the sequence based on the input sequence.

        Args:
        input_sequence (Tensor): The input sequence (a sequence of tokenized words).

        Returns:
        Tensor: The predicted next word as a probability distribution.
        """
        # Use the model's forward pass (`call` method) to predict the next word based on the input sequence.
        return self.call(input_sequence)
