# Routine to store methods to calculate the local (individual finite el-
# ement level) residual vector due to internal forces

import tensorflow as tf 

# Defines a class to get the constitutive model dictionary and transform
# into a compiled evaluation of the strain energy and of the first Piola-
# Kirchhoff stress tensor

class CompiledStrainEnergy:

    def __init__(self, constitutive_models_dict, masks_dictionary):
        
        # Initializes the list of strain energy functions and the list
        # of masks

        self.mask_list = []

        self.energy_functions_list = []

        # Iterates through the dictionary of constitutive models

        for physical_group, constitutive_class in (
        constitutive_models_dict.items()):

            # Adds the mask using the physical group as key in the dic-
            # tionary of masks

            self.mask_list.append(masks_dictionary[physical_group])

            # Adds the energy function

            self.energy_functions_list.append(
            constitutive_class.strain_energy)

        # Gets the number of materials

        self.n_materials = len(self.energy_functions_list)

    # Defines a function to assemble the total strain energy

    @tf.function
    def assemble_total_strain_energy(self, F):

        return tf.add_n([self.mask_list[i]*self.energy_functions_list[i](
        F) for i in range(self.n_materials)])
    
    # Defines a function to compute the first Piola-Kichhoff stress ten-
    # sor

    @tf.function
    def first_piola_kirchhoff(self, F):

        # Computes the derivative of the strain energy with respect to 
        # the deformation gradient

        with tf.GradientTape() as tape:

            tape.watch(F)

            psi = self.assemble_total_strain_energy(F)

        return tape.gradient(psi, F)