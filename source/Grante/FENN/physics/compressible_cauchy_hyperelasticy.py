# Routine to assemble the residual vector of a hyperelastic compressible 
# Cauchy continuum

import tensorflow as tf

from ..assembly.hyperelastic_internal_work import CompressibleInternalWorkReferenceConfiguration

# Defines a class that assembles the residual vector

class CompressibleHyperelasticity:

    def __init__(self, mesh_data_class, vector_of_parameters,
    constitutive_models_dict):

        # Initializes the the global residual vector as a null variable

        vector_of_parameters = tf.Variable(tf.zeros([
        mesh_data_class.global_number_dofs], dtype=mesh_data_class.dtypes))
        
        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the internal work

        internal_work_variation = CompressibleInternalWorkReferenceConfiguration(
        vector_of_parameters, constitutive_models_dict, 
        mesh_data_class.domain_elements, 
        mesh_data_class.domain_physicalGroupsNameToTag)