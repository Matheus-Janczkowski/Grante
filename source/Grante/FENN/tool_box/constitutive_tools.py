# Routine to store methods to process information in and for constituti-
# ve models

import tensorflow as tf

########################################################################
#                              Kinematics                              #
########################################################################

# Defines a function to compute the deformation gradient as a tensor [
# n_elements, n_quadrature_points, 3, 3]

def compute_batched_deformation_gradient(field_vector, element_class):

    # Gets a tensor [n_elements, n_nodes, 3] of the DOFs of the field
    # vector per finite element

    field_dofs = element_class.get_field_dofs(field_vector)

    # Contracts the DOFs to get the displacement gradient as a tensor [
    # n_elements, n_quadrature_points, 3, 3]

    grad_u = tf.einsum('eqnx,enx->eqxx', 
    element_class.shape_functions_derivatives, field_dofs)