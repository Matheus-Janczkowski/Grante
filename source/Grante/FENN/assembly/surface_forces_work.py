# Routine to store methods to calculate the residual vector due to sur-
# face forces

import tensorflow as tf

# Defines a class to compute the contribution to the residual vector due
# to the surface tractions

class ReferentialTractionWork:

    def __init__(self, traction_dict, boundary_physical_groups_dict, 
    mesh_dict):

        # Creates a list of the variation of the primal field at the 
        # surfaces multiplied by the surface integration measure

        self.variation_field_ds = []

        # Iterates through the dictionary of constitutive models

        for physical_group, traction_vector in traction_dict.items():
            
            # Gets the physical group tag

            physical_group_tag = None 

            if not (physical_group in boundary_physical_groups_dict):

                raise NameError("The physical group name '"+str(
                physical_group)+"' in the dictionary of tractions is n"+
                "ot a valid physical group name. Check the available n"+
                "ames:\n"+str(list(boundary_physical_groups_dict.keys())))
            
            else:

                physical_group_tag = boundary_physical_groups_dict[
                physical_group]

            # Gets the instance of the mesh data

            if isinstance(mesh_dict[physical_group_tag], dict):

                raise NotImplementedError("Multiple element types per "+
                "physical group has not yet been updated to compute Re"+
                "ferentialTractionWork")
            
            # Gets the first element type

            mesh_data = mesh_dict[physical_group_tag]

            # Gets the shape functions multiplied by the surface inte-
            # gration measure

            self.variation_field_ds.append(tf.einsum('eqnj,eq->eqnj',
            mesh_data.shape_functions_tensor, mesh_data.dx))

        # Gets the number of surfaces under load

        self.n_surfaces_under_load = len(self.variation_field_ds)

        # Stacks the shape functions multiplied by the integration mea-
        # sure into a tensor [n_physical_groups, n_elements, 
        # n_quadrature_points, n_nodes, n_physical_dimensions]

        self.variation_field_ds = tf.stack(self.variation_field_ds,
        axis=0)

    # Defines a function to assemble the residual vector

    @tf.function
    def assemble_residual_vector(self, global_residual_vector):

        # Iterates through the physical groups of the surfaces under load

        for i in range(self.n_surfaces_under_load):

            # Gets the batched tensor [n_elements, n_quadrature_points,
            # 3, 3] of the first Piola-Kirchhoff stress

            P = self.first_piola_kirchhoff_list[i](self.deformation_gradient_list[
            i].compute_batched_deformation_gradient())

            # Contracts the first Piola-Kirchhoff stress with the deri-
            # vatives of the shape functions multiplied by the integra-
            # tion measure to get the integration of the internal work 
            # of the variational form. Then, sums over the quadrature 
            # points, that are the second dimension (index 1 in python 
            # convention).
            # The result is a tensor [n_elements,  n_nodes, 
            # n_physical_dimensions]

            internal_work = tf.reduce_sum(tf.einsum('eqi,eqni->eqni', 
            P, self.variation_field_ds[i]), axis=1)

            # Adds the contribution of this physical group to the global
            # residual vector. Uses the own tensor of DOF indexing to
            # scatter the local residual. Another dimension is added to
            # the indexing tensor to make it compatible with tensorflow
            # tensor_scatter_nd_add. Performs this change in place, as
            # global_residual_vector is a variable

            global_residual_vector.scatter_nd_add(tf.expand_dims(
            self.deformation_gradient_list[i].indexing_dofs_tensor, axis=
            -1), internal_work)