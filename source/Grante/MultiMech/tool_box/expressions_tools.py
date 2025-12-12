# Routine to store user expressions

from dolfin import *

import numpy as np

from ...PythonicUtilities.function_tools import get_functions_arguments

# Defines a function to interpolate a python function into a finite ele-
# ment space, the function must return a scalar

def interpolate_scalar_function(scalar_function, function_space, name=
None):

    # Verifies the number of arguments, it must be only one: the position
    # vector in the mesh

    number_of_arguments = get_functions_arguments(scalar_function, 
    number_of_arguments_only=True)

    if number_of_arguments!=1:

        raise ValueError("A scalar function is to be interpolated onto"+
        " a finite element space, but it has "+str(number_of_arguments)+
        " arguments, whereas it should have just 1, the position vector")
    
    # Verifies if the element has Lagrangian interpolation functions

    if not (function_space.ufl_element().family()=="Lagrange"):

        raise TypeError("The function space is not lagrangian, but "+str(
        function_space.ufl_element().family())+". Thus the expression "+
        "cannot be interpolated. You must choose a 'CG', 'Lagrange', o"+
        "r 'P' finite element family")
    
    # Gets the coordinates of the DOFs

    dofs_coordinates = function_space.tabulate_dof_coordinates()

    # Gets the values of the function in the nodes

    nodes_values = np.array([scalar_function(position_vector) for (
    position_vector) in dofs_coordinates])

    # Creates a Function element over the finite element space

    function_object = Function(function_space)

    # Updates the vector of parameters

    function_object.vector()[:] = nodes_values

    # Renames it if the name is not None

    if name is not None:

        function_object.rename(name, "DNS")

    # Returns the function object

    return function_object

# Defines a function to interpolate a vector-valued or tensor-valued 
# function on a function space. The function must return a numpy array 
# and it must receive as argument a position vector and the number of 
# the component to be evaluated

def interpolate_tensor_function(vector_function, function_space, name=
None):

    # Verifies the number of arguments, it must be only one: the position
    # vector in the mesh

    number_of_arguments = get_functions_arguments(vector_function, 
    number_of_arguments_only=True)

    if number_of_arguments!=2:

        raise ValueError("A vector-valued function is to be interpolat"+
        "ed onto a finite element space, but it has "+str(
        number_of_arguments)+" arguments, whereas it should have just "+
        "2, the position vector and the local component to be evaluate"+
        "d. Example: ([x,y,z], 2) => the second component of a tensor "+
        "will be evaluated at x, y, z")
    
    # Verifies if the element has Lagrangian interpolation functions

    if not (function_space.ufl_element().family()=="Lagrange"):

        raise TypeError("The function space is not lagrangian, but "+str(
        function_space.ufl_element().family())+". Thus the expression "+
        "cannot be interpolated. You must choose a 'CG', 'Lagrange', o"+
        "r 'P' finite element family")
    
    # Gets the coordinates of the DOFs

    dofs_coordinates = function_space.tabulate_dof_coordinates()

    # Gets the dimension of the field from the function space

    field_number_of_components = int(np.prod(function_space.ufl_element(
    ).value_shape()))

    # Gets the values of the function in the DOFs

    DOFs_values = np.zeros(function_space.dim())

    for dof in range(function_space.dim()):

        # Gets the coordinates of the DOF

        dof_coordinate = dofs_coordinates[dof]

        # Evaluates the function and updates the corresponding local 
        # number of the DOF only

        DOFs_values[dof] = vector_function(dof_coordinate, dof % (
        field_number_of_components))

    # Creates a Function element over the finite element space

    function_object = Function(function_space)

    # Updates the vector of parameters

    function_object.vector()[:] = DOFs_values

    # Renames it if the name is not None

    if name is not None:

        function_object.rename(name, "DNS")

    # Returns the function object

    return function_object