# Routine to store tests for the finite elements

# Routine to store some tests to evaluate convex input neural networks

import unittest

import tensorflow as tf

import numpy as np

from dolfin import assemble

from ..physics.compressible_cauchy_hyperelasticy import CompressibleHyperelasticity

from ..constitutive_models.hyperelastic_isotropic_models import NeoHookean

from ..tool_box import mesh_tools

from ...MultiMech.tool_box.mesh_handling_tools import create_box_mesh, read_mshMesh

from ...MultiMech.tool_box import functional_tools, variational_tools

from ...MultiMech.constitutive_models.hyperelasticity.isotropic_hyperelasticity import NeoHookean as NeoHookeanMultiMech

from ...PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        pass

    # Defines a function to test reading a mesh

    def test_hyperelastic_residual_vector(self):

        print("\n#####################################################"+
        "###################\n#              Tests the evaluation of t"+
        "he residual vector             #\n###########################"+
        "#############################################\n")

        file_name = "box"

        file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        length_x = 0.2
        
        length_y = 0.3
        
        length_z = 1.0
        
        n_divisions_x = 2 

        n_divisions_y = 2
        
        n_divisions_z = 2

        quadrature_degree = 2

        create_box_mesh(length_x, length_y, length_z, n_divisions_x, 
        n_divisions_y, n_divisions_z, file_name=file_name, verbose=False, 
        convert_to_xdmf=False, file_directory=file_directory, 
        mesh_polinomial_order=2)

        # Defines a dictionary of finite element per field

        elements_per_field = {"Displacement": {"number of DOFs per nod"+
        "e": 3, "required element type": "tetrahedron of 10 nodes"}}

        # Reads this mesh

        mesh_data_class = mesh_tools.read_msh_mesh(file_name, 
        quadrature_degree, elements_per_field, verbose=True)

        # Initializes the vector of parameters as the null vector

        vector_of_parameters = np.zeros(81)

        prescribed_dofs = [14, 17, 20, 23, 38, 41, 44, 47, 77]

        dirichlet_load = 0.1

        for dof in prescribed_dofs:

            vector_of_parameters[dof] = dirichlet_load

        vector_of_parameters = tf.Variable(tf.constant(
        vector_of_parameters, dtype=mesh_data_class.dtype))

        # Sets the dictionary of constitutive models

        material_properties = {"E": 1E6, "nu": 0.4}

        constitutive_models = {"volume 1": NeoHookean(material_properties, 
        mesh_data_class)}

        # Sets the dictionary of traction classes

        maximum_load = 1E5

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
        "amplitude_tractionZ": maximum_load}}

        # Instantiates the class to evaluate the residual vector

        residual_class = CompressibleHyperelasticity(mesh_data_class,
        vector_of_parameters, constitutive_models, traction_dictionary=
        traction_dictionary)

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector()

        # Evaluates the residual vector using FEniCS

        mesh_data_class = read_mshMesh({"length x": length_x, "length y": 
        length_y, "length z": length_z, "number of divisions in x": 
        n_divisions_x, "number of divisions in y": n_divisions_y, "num"+
        "ber of divisions in z": n_divisions_z, "verbose": False, "mes"+
        "h file name": "box_mesh", "mesh file directory": 
        get_parent_path_of_file()})

        functional_data_class = functional_tools.construct_monolithicFunctionSpace(
        {"Displacement": {"field type": "vector", "interpolation funct"+
        "ion": "CG", "polynomial degree": 2}}, mesh_data_class)

        # Dirichlet boundary conditions

        bc, dirichlet_loads = functional_tools.construct_DirichletBCs({
        "bottom": {"BC case": "PrescribedDirichletBC", "bc_information"+
        "sDict": {"load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, dirichlet_load]}}}, functional_data_class, 
        mesh_data_class)

        # Variational form of the exterior work using an uniform referential 
        # traction

        external_work, neumann_loads = variational_tools.traction_work({
        "top": {"load case": "UniformReferentialTraction", "amplitude_"+
        "tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tract"+
        "ionZ": maximum_load, "parametric_load_curve": "square_root", 
        "t": 0.0, "t_final": 1.0}}, "Displacement", 
        functional_data_class, mesh_data_class, [])

        constitutive_model_multimech = NeoHookeanMultiMech(
        material_properties)

        constitutive_model_multimech.check_model(None)

        internal_work = variational_tools.hyperelastic_internalWorkFirstPiola(
        "Displacement", functional_data_class, 
        constitutive_model_multimech, mesh_data_class)

        # Update the load class

        neumann_loads[0].update_load(1.0)

        dirichlet_loads[0].update_load(1.0)

        residual_vector_fenics = [float(dof) for dof in assemble(
        internal_work-external_work)]

        print("The residual vector according to fenics is:\n"+str(
        residual_vector_fenics)+"\n")

        residual_vector = residual_vector.numpy()

        for i in range(len(residual_vector)):

            if abs(residual_vector[i])>1E-5:

                print("FENN: residual_vector["+str(i)+"]="+str(residual_vector[
                i]))

# Runs all tests

if __name__=="__main__":

    unittest.main()