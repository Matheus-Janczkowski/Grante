# Routine to store cylindrical shapes

import numpy as np

from ..solids import cuboid_generator as cuboid

from ..tool_box import geometric_tools as geo

# Defines a function to create a rectangular prism with right corners

def right_rectangularPrism(length_x, length_y, length_z, axis_vector, 
base_point, transfinite_directions=[], bias_directions=dict(), 
shape_spin=0.0, geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],
[]], dict(), [], dict(), [], [], [], 0.5, False]):
    
    # Creates the 8 corners as if the long axis is the Z axis proper, and 
    # the prism starts at the XY plane towards positive Z

    corner_points = [[length_x, length_x, 0.0, 0.0, length_x, length_x, 
    0.0, 0.0], [0.0, length_y, length_y, 0.0, 0.0, length_y, length_y, 
    0.0], [0.0, 0.0, 0.0, 0.0, length_z, length_z, length_z, length_z]]

    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of this shape

    native_axis = [1.0, 0.0, 0.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axis, shape_spin)

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, translation_vector
    =base_point, geometric_data=geometric_data)

    return geometric_data

# Defines a function to create a generic 6-sided prism

def hexahedron_from_corners(corner_points, transfinite_directions=[], 
bias_directions=dict(), geometric_data=[0, [[],[],[],[]
], [[],[],[],[]], [[],[],[]], dict(), [], dict(), [], [], [], 0.5, False]):
    
    # Tests if the corner points is a numpy array

    if isinstance(corner_points, np.ndarray):

        # Tests if it has the right shape

        if corner_points.shape==(8,3):

            corner_points = corner_points.T

        elif corner_points.shape!=(3,8):

            raise ValueError("'corner_points' is a numpy array of shap"+
            "e "+str(corner_points.shape)+", but it should have shape "+
            "(3,8) to create a hexadron")
        
        # Transforms it to a list

        corner_points = corner_points.tolist()

    elif isinstance(corner_points, list):

        if len(corner_points)==8:

            corner_points = (np.array(corner_points).T).tolist()

        elif len(corner_points)!=3:

            raise IndexError("'corner_points' is a list, but it has le"+
            "ngth of "+str(len(corner_points))+", whereas it should be"+
            " 3 or 8 (transposed) to construct a hexadron")

        for point in corner_points:

            if len(point)!=8:

                raise IndexError("The sublist '"+str(point)+"' in "+
                "'corner_points' does not have length of 8. Thus, "+
                "it is not possible to create a hexadron")
                
    else:

        raise TypeError("'corner_points' should be a list of lists or "+
        "numpy array of shape (8,3) or (3,8) to create a hexadron")

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data)

    return geometric_data