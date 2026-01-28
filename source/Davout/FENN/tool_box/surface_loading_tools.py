# Routine to store methods to apply Neumann boundary conditions on sur-
# faces

import tensorflow as tf

from ..tool_box.tensorflow_utilities import convert_object_to_tensor

########################################################################
#                            Traction vector                           #
########################################################################

# Defines a class to store methods to build a traction vector on a sur-
# face

class TractionVectorOnSurface:

    def __init__(self, surface_mesh_data, traction_vector, 
    vector_of_parameters):
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        # Verifies if traction vector is a 3D vector

        if not isinstance(traction_vector, list):

            raise TypeError("'traction_vector' in 'TractionVectorOnSur"+
            "face' is not a list. Currently, it is: "+str(
            traction_vector))
        
        elif len(traction_vector)!=3:

            raise IndexError("'traction_vector' in 'TractionVectorOnSu"+
            "rface' must be a list with 3 components, T_x, T_y, and T_"+
            "z. Currently, it is: "+str(traction_vector))
        
        # Converts traction vector to a tensor

        self.traction_vector = convert_object_to_tensor(traction_vector,
        surface_mesh_data.dtype)
        
        # Calls the method to build the traction tensor

        self.traction_tensor = self.compute_traction()
        
    # Defines a function to build the traction [n_elements, 
    # n_quadrature_points, 3] with the first ever given vector of para-
    # meters

    @tf.function
    def compute_traction(self):

        # Gets the number of elements and the number of quadrature points
        # to create the traction tensor

        return tf.broadcast_to(self.traction_vector, [
        self.surface_mesh_data.number_elements, 
        self.surface_mesh_data.number_quadrature_points, 3])

########################################################################
#                       Prescribed stress tensor                       #
########################################################################

# Defines a class to store methods to build a traction vector on a sur-
# face, given a prescribed stress tensor on a boundary

class FirstPiolaKirchhoffOnSurface:

    def __init__(self, surface_mesh_data, 
    prescribed_first_piola_kirchhoff, vector_of_parameters):
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        # Verifies if prescribed_first_piola_kirchhoff is a 3x3 list

        if not isinstance(prescribed_first_piola_kirchhoff, list):

            raise TypeError("'prescribed_first_piola_kirchhoff' in 'Fi"+
            "rstPiolaKirchhoffOnSurface' is not a list. Currently, it "+
            "is: "+str(prescribed_first_piola_kirchhoff))
        
        elif len(prescribed_first_piola_kirchhoff)!=3:

            raise IndexError("'prescribed_first_piola_kirchhoff' in 'F"+
            "irstPiolaKirchhoffOnSurface' must be a list with 3 sublis"+
            "ts: [[P11, P12, P13], [P21, P22, P23], [P31, P32, P33]]. "
            "Currently, it is: "+str(prescribed_first_piola_kirchhoff))
        
        else:
        
            for sublist in prescribed_first_piola_kirchhoff:

                if (not isinstance(sublist, list)) or len(sublist)!=3:

                    raise IndexError("'prescribed_first_piola_kirchhof"+
                    "f' in 'FirstPiolaKirchhoffOnSurface' must be a li"+
                    "st with 3 sublists: [[P11, P12, P13], [P21, P22, "
                    "P23], [P31, P32, P33]]. Currently, it is: "+str(
                    prescribed_first_piola_kirchhoff))
        
        # Converts prescribed_first_piola_kirchhoff list of lists to a 
        # tensor

        self.prescribed_first_piola_kirchhoff = convert_object_to_tensor(
        prescribed_first_piola_kirchhoff, surface_mesh_data.dtype)
        
        # Calls the method to build the traction tensor

        self.traction_tensor = self.compute_traction()
        
    # Defines a function to build the traction [n_elements, 
    # n_quadrature_points, 3] with the first ever given vector of para-
    # meters

    @tf.function
    def compute_traction(self):

        # Contracts the first Piola-Kirchhoff stress tensor with the ten-
        # sor of normal vectors of the mesh

        return tf.einsum('ij,eqj->eqi', 
        self.prescribed_first_piola_kirchhoff, 
        self.surface_mesh_data.normal_vector)