# Routine to store a class with custom activation functions

import tensorflow as tf

class CustomActivationFunctions:

    def __init__(self):
    
        # Sets a list of custom activation functions

        custom_activation_functions_names = ["quadratic"]

    # Defines a quadratic activation function

    def quadratic(self, x, a2=tf.constant(1.0), a1=tf.constant(0.0), a0=
    tf.constant(0.0)):

        return (a2*tf.square(x))+(a1*x)+a0