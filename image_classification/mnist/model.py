import tensorflow as tf


class digitClassifier(tf.keras.Model):
    def __init__(self, n_hidden_1, n_hidden_2, num_classes):
        # Define each layer
        super(digitClassifier, self).__init__()
        # Hidden fully connected layer with 256 neurons
        self.layer1 = tf.layers.Dense(n_hidden_1, activation=tf.nn.relu)

        # Hidden fully connected layer with 256 neurons
        self.layer2 = tf.layers.Dense(n_hidden_2, activation=tf.nn.relu)
        # Output fully connected layer with a neuron for each class
        self.out_layer = tf.layers.Dense(num_classes)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)
