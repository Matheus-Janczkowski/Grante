# Routine to store methods to process information in and for constituti-
# ve models

import tensorflow as tf

########################################################################
#                              Kinematics                              #
########################################################################

# Defines a function to compute the deformation gradient as a tensor [
# n_elements, n_quadrature_points, 3, 3]

@tf.function
def compute_batched_deformation_gradient(field_dofs, 
shape_functions_derivatives, identity_tensor):

    """
    Computes the deformation gradient for all quadrature points in each
    element. Returns a tensor [n_elements, n_quadrature_points, 3, 3]

    field_dofs: displacement DOFs per element as a tensor [n_elements,
    n_nodes, 3]

    shape_function_derivatives: derivatives of the shape functions with
    respect to the original coordinates (x, y, z) as a tensor [
    n_elements, n_quadrature_points, n_nodes, 3]

    identity_tensor: identity tensor as a tensor [n_elements, 
    n_quadrature_points, 3, 3]
    """

    # Contracts the DOFs to get the material displacement gradient as a 
    # tensor [n_elements, n_quadrature_points, 3, 3]. THen, adds the i-
    # dentity tensor and returns

    return (tf.einsum('eqnj,eni->eqij', shape_functions_derivatives, 
    field_dofs)+identity_tensor)