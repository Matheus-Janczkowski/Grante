# Routine to store some tests to evaluate convex input neural networks

import unittest

import time

import tensorflow as tf

from ...tool_box import ANN_tools

from ...tool_box import loss_tools

from ...tool_box import loss_assembler_classes as loss_assemblers

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines the test data for the gradient tests

        self.input_dimension_gradient_tests = 3

        self.output_dimension_gradient_tests = 1

        self.activation_list_gradient_tests = [{"relu": 1000}, {"lin"+
        "ear": self.output_dimension_gradient_tests}]

        self.n_samples_gradient_tests = 100

        # Defines a function to get the true values

        def true_function(x):

            value = 0.0

            for x_i in x:

                value = x_i**2

            return value
        
        self.true_function = true_function

        # Sets the training and test data

         # Creates the new test data

        x_min = -1.0

        x_max = 1.0

        data_matrix = []

        true_values = []

        for i in range(self.n_samples_gradient_tests):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ) for j in range(self.input_dimension_gradient_tests)])

            # Evaluaets the true function

            true_values.append(self.true_function(data_matrix[-1]))

        n_samplesTraining = 6

        self.training_data = data_matrix[:n_samplesTraining]

        self.training_trueValues = true_values[:n_samplesTraining]

        self.test_data = data_matrix[n_samplesTraining:]

        self.test_trueValues = true_values[n_samplesTraining:]

        # Converts thet data to tensors

        self.training_inputTensor = tf.constant(self.training_data, 
        dtype=tf.float32)

        self.test_inputTensor = tf.constant(self.test_data, dtype=
        tf.float32)

        self.training_trueTensor = tf.constant(self.training_trueValues, 
        dtype=tf.float32)

        self.test_trueTensor = tf.constant(self.test_trueValues, dtype=
        tf.float32)

        # Defines the loss function metric

        self.loss_metric = tf.keras.losses.MeanAbsoluteError()

        # Sets the optimizer

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, 
        momentum=0.9, nesterov=True)

        self.verbose_deltaIterations = 1000

    # Defines a function to test the fully convex-input neural networks

    def test_fully_convex_input_nn(self):

        print("\n#####################################################"+
        "###################\n#                Tests fully convex-inpu"+
        "t neural network               #\n###########################"+
        "#############################################\n")

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        evaluate_parameters_gradient=False, verbose=True)

        custom_model = ANN_class()

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, self.loss_metric, self.training_data, 
        model_true_values=self.training_trueTensor, convex_input_model=
        True)

        print("\nWarms up")

        t_initial = time.time()

        result = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up after "+str(elapsed_time))

        t_initial = time.time()

        objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

# Runs all tests

if __name__=="__main__":

    unittest.main()