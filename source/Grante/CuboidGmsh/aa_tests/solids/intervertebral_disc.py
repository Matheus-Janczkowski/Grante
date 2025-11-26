# Routine to mesh the intervertebral disc given points

import numpy as np

from ...solids import cuboid_prisms as prisms

from ...tool_box import meshing_tools as tools

# Defines a function to construct the intervertebral disc using prisms

def mesh_disc():

    length_x = 1.0

    length_y = 1.5

    length_z = 2.5

    transfinite_directions = [7,7,11]

    ####################################################################
    #                     Boundary surfaces setting                    #
    ####################################################################

    # Sets the finder for the boundary surfaces

    # XY plane at z = 0

    def back_expression(x, y, z):

        return z

    # XY plane at z = length_z

    def front_expression(x, y, z):

        return z-length_z

    # XZ plane at y = 0

    def lower_expression(x, y, z):

        return y

    # XZ plane at y = length_y

    def upper_expression(x, y, z):

        return y-length_y

    # YZ plane at x = 0

    def right_expression(x, y, z):

        return x

    # YZ plane at x = x_length

    def left_expression(x, y, z):

        return x-length_x

    # Sets a list of expressions to find the surfaces at the boundaries

    surface_regionsExpressions = [back_expression, front_expression, 
    lower_expression, upper_expression, right_expression, left_expression]

    # Sets the names of the surface regions

    surface_regionsNames = ['back', 'front', 'lower', 'upper', 'right',
    'left']

    geometric_data = tools.gmsh_initialization(surface_regionsExpressions
    =surface_regionsExpressions, surface_regionsNames=
    surface_regionsNames, tolerance_finders=1E-9)

    corner_points = [[length_x, 0.0, 0.0], [length_x, length_y, 0.0], [
    0.0, length_y, 0.0], [0.0, 0.0, 0.0], [length_x, 0.0, length_z], [
    length_x, length_y, length_z], [0.0, length_y, length_z], [0.0, 0.0, 
    length_z]]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data)

    tools.gmsh_finalize(geometric_data=geometric_data, file_name="inte"+
    "rvertebral_disc")

# Test block

if __name__=="__main__":

    mesh_disc()