# Routine to store methods to apply Neumann boundary conditions on sur-
# faces

import tensorflow as tf

from .tensorflow_utilities import convert_object_to_tensor

########################################################################
#                            Traction vector                           #
########################################################################

# Defines a class to store methods to build a traction vector on a sur-
# face

class TractionVectorOnSurface:

    def __init__(self, surface_mesh_data, traction_information, 
    vector_of_parameters, physical_group_name):
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        # Initializes the traction vector

        traction_vector = []

        # Verifies if the traction information provided by the user has
        # the keys for each traction component

        dimensions_names = ["X", "Y", "Z"]

        for dimension in dimensions_names:

            # Verifies the key

            amplitude_key = "amplitude_traction"+str(dimension)

            if not (amplitude_key in traction_information):

                raise KeyError("The information dictionary provided to"+
                " set 'TractionVectorOnSurface' in surface '"+str(
                physical_group_name)+"' does not have the key '"+
                amplitude_key+"'. The given dictionary is: "+str(
                traction_information))
            
            else:

                traction_vector.append(traction_information[
                amplitude_key])
        
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
    prescribed_first_piola_kirchhoff_info, vector_of_parameters,
    physical_group_name):
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        # Initializes the list of prescribed first Piola-Kirchhoff stress
        # tensor

        prescribed_first_piola_kirchhoff = []

        # Iterates through the first index of the tensor

        for i in range(3):

            # Adds this row

            prescribed_first_piola_kirchhoff.append([])

            # Iterates through the second index

            for j in range(3):

                # Verifies if this index is present on the information
                # dictionary

                index_key = "P"+str(i+1)+str(j+1)

                if not (index_key in (
                prescribed_first_piola_kirchhoff_info)):
                    
                    raise KeyError("The information dictionary provide"+
                    "d to set 'FirstPiolaKirchhoffOnSurface' in surfac"+
                    "e '"+str(physical_group_name)+"' does not have th"+
                    "e key '"+index_key+"'. The given dictionary is: "+
                    str(prescribed_first_piola_kirchhoff_info))
                
                # Updates the list using this index
                
                prescribed_first_piola_kirchhoff[-1].append(
                prescribed_first_piola_kirchhoff_info[index_key])
        
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