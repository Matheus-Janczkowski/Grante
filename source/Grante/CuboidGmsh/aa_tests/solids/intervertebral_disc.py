# Routine to mesh the intervertebral disc given points

import numpy as np

from ...solids import cuboid_prisms as prisms

from ...tool_box import meshing_tools as tools

# Defines a function to construct the intervertebral disc using prisms

def mesh_disc():

    n_points_spline = 3

    inner_radius = 3.0

    outer_radius = 5.0

    inner_curve_x = lambda theta: inner_radius*np.cos((2*np.pi*theta)+(
    (7/4)*np.pi))

    inner_curve_y = lambda theta: 1.5*inner_radius*np.sin((2*np.pi*theta)+(
    (7/4)*np.pi))

    outer_curve_x = lambda theta: outer_radius*np.cos((2*np.pi*theta)+(
    (7/4)*np.pi))

    outer_curve_y = lambda theta: outer_radius*np.sin((2*np.pi*theta)+(
    (7/4)*np.pi))

    length_x_core = 1.0

    length_y_core = 1.5

    height = 2.5

    transfinite_directions = [5,6,10]

    ####################################################################
    #                     Boundary surfaces setting                    #
    ####################################################################

    # Sets the names of the surface regions

    surface_regionsNames = ['lower', 'upper']

    ####################################################################
    #                    Volumetric regions setting                    #
    ####################################################################

    # Sets the names of the volume regions

    volume_regionsNames = ['nucleus', 'annulus']

    ####################################################################
    #                              Cuboids                             #
    ####################################################################

    # Initializes the geometric data

    geometric_data = tools.gmsh_initialization(surface_regionsNames=
    surface_regionsNames, volume_regionsNames=volume_regionsNames)

    # Center cube

    corner_points_center = [[0.5*length_x_core, -0.5*length_y_core, 0.0], [0.5*
    length_x_core, 0.5*length_y_core, 0.0], [-0.5*length_x_core, 0.5*
    length_y_core, 0.0], [-0.5*length_x_core, -0.5*length_y_core, 0.0], 
    [0.5*length_x_core, -0.5*length_y_core, height], [0.5*length_x_core, 
    0.5*length_y_core, height], [-0.5*length_x_core, 0.5*length_y_core, 
    height], [-0.5*length_x_core, -0.5*length_y_core, height]]

    geometric_data = prisms.hexahedron_from_corners(corner_points_center, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"})

    # First nucleus flare

    theta_1 = 0.0

    theta_2 = 0.25

    corner_points = [[inner_curve_x(theta_1), inner_curve_y(theta_1), 
    0.0], [inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], 
    corner_points_center[1], corner_points_center[0], [inner_curve_x(theta_1), 
    inner_curve_y(theta_1), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], corner_points_center[5], corner_points_center[4]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Second nucleus flare

    theta_1 = 0.25

    theta_2 = 0.5

    corner_points = [[inner_curve_x(theta_1), inner_curve_y(theta_1), 
    0.0], [inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], 
    corner_points_center[2], corner_points_center[1], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], corner_points_center[6], 
    corner_points_center[5]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Third nucleus flare

    theta_1 = 0.5

    theta_2 = 0.75

    corner_points = [[inner_curve_x(theta_1), inner_curve_y(theta_1), 
    0.0], [inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], 
    corner_points_center[3], corner_points_center[2], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], corner_points_center[7], 
    corner_points_center[6]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={6: "lower", 1: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Fourth nucleus flare

    theta_1 = 0.75

    theta_2 = 1.0

    corner_points = [[inner_curve_x(theta_1), inner_curve_y(theta_1), 
    0.0], [inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], 
    corner_points_center[0], corner_points_center[3], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], corner_points_center[4], 
    corner_points_center[7]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([inner_curve_x(theta_1+(delta_theta*(i+1))),
        inner_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # First annulus flare

    theta_1 = 0.0

    theta_2 = 0.25

    corner_points = [[outer_curve_x(theta_1), outer_curve_y(theta_1), 
    0.0], [outer_curve_x(theta_2), outer_curve_y(theta_2), 0.0], [
    inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), 0.0], [outer_curve_x(theta_1), 
    outer_curve_y(theta_1), height], [outer_curve_x(theta_2), 
    outer_curve_y(theta_2), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], [inner_curve_x(theta_1), 
    inner_curve_y(theta_1), height]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Second annulus flare

    theta_1 = 0.25

    theta_2 = 0.5

    corner_points = [[outer_curve_x(theta_1), outer_curve_y(theta_1), 
    0.0], [outer_curve_x(theta_2), outer_curve_y(theta_2), 0.0], [
    inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), 0.0], [outer_curve_x(theta_1), 
    outer_curve_y(theta_1), height], [outer_curve_x(theta_2), 
    outer_curve_y(theta_2), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], [inner_curve_x(theta_1), 
    inner_curve_y(theta_1), height]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Third annulus flare

    theta_1 = 0.5

    theta_2 = 0.75

    corner_points = [[outer_curve_x(theta_1), outer_curve_y(theta_1), 
    0.0], [outer_curve_x(theta_2), outer_curve_y(theta_2), 0.0], [
    inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), 0.0], [outer_curve_x(theta_1), 
    outer_curve_y(theta_1), height], [outer_curve_x(theta_2), 
    outer_curve_y(theta_2), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], [inner_curve_x(theta_1), 
    inner_curve_y(theta_1), height]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={6: "lower", 1: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Fourth annulus flare

    theta_1 = 0.75

    theta_2 = 1.0

    corner_points = [[outer_curve_x(theta_1), outer_curve_y(theta_1), 
    0.0], [outer_curve_x(theta_2), outer_curve_y(theta_2), 0.0], [
    inner_curve_x(theta_2), inner_curve_y(theta_2), 0.0], [inner_curve_x(
    theta_1), inner_curve_y(theta_1), 0.0], [outer_curve_x(theta_1), 
    outer_curve_y(theta_1), height], [outer_curve_x(theta_2), 
    outer_curve_y(theta_2), height], [inner_curve_x(theta_2), 
    inner_curve_y(theta_2), height], [inner_curve_x(theta_1), 
    inner_curve_y(theta_1), height]]

    edge_points_1 = []

    edge_points_5 = []

    delta_theta = (theta_2-theta_1)/(n_points_spline+1)

    for i in range(n_points_spline):

        edge_points_1.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), 0.0])

        edge_points_5.append([outer_curve_x(theta_1+(delta_theta*(i+1))),
        outer_curve_y(theta_1+(delta_theta*(i+1))), height])

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5})

    # Creates the geometry and meshes it

    tools.gmsh_finalize(geometric_data=geometric_data, file_name="inte"+
    "rvertebral_disc")

# Test block

if __name__=="__main__":

    mesh_disc()