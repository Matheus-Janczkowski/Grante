# Routine to test the implementation of optimization methods

import unittest

import numpy as np

from ...tool_box import optimization_tools

# Defines a function to test the ANN optimization wrappers

class TestOptimizationWrappers(unittest.TestCase):

    def setUp(self):

        # Sets the number of optimization iterations

        self.n_iterations = 10000

        # Sets the convergence tolerance

        self.gradient_tolerance = 1E-3

    # Defines a function to test the Rosenbrock objective function
    
    def test_rosenbrock(self):

        print("\n#####################################################"+
        "###################\n#                              Rosenbroc"+
        "k                              #\n###########################"+
        "#############################################\n")

        # Sets the optimization function

        a=1.0
        
        b=100.0

        def rosenbrock(x):

            return ((a-x[0])**2)+(b*((x[1]-(x[0]**2))**2))
        
        # Sets the gradient

        def grad_rosenbrock(x):

            return [(-2.0*(a-x[0]))-(4.0*b*(x[1]-(x[0]**2))*x[0
            ]), 2.0*b*(x[1]-(x[0]**2))]
        
        # Sets the true minimum

        true_minimum = np.array([1.0, 1.0])

        # Sets the optimization problem's parameters

        optimization_method = "CG"

        initial_guess = np.array([0.0, 3.0])

        # Sets the optimization class

        optimization_class = optimization_tools.UnconstrainedFirstOrderMinimization(
        objective_function=rosenbrock, objective_gradient=
        grad_rosenbrock, optimization_method=optimization_method, 
        initial_guess=initial_guess)
        
        # Solves

        optimization_class()

        print("The minimum found for the Rosenbrock function is: "+str(
        optimization_class.design_variables)+"\n")

        self.assertEqual(np.testing.assert_allclose(true_minimum, 
        optimization_class.design_variables), None)

# Runs all tests

if __name__=="__main__":

    unittest.main()