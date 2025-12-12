# Routine to turn an expression in cylindrical coordinates into a field
# projected onto a finite element space 

from dolfin import *

import numpy as np

from scipy.interpolate import RBFInterpolator

from .....Grante.MultiMech.tool_box import mesh_handling_tools

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

# Gets the path to the mesh

mesh_path = (get_parent_path_of_file(path_bits_to_be_excluded=2)+"//te"+
"st_meshes//intervertebral_disc_mesh")

# Reads the mesh

mesh_data_class = mesh_handling_tools.read_mshMesh(mesh_path)

# Gets the nodes on the outer and inner lateral surfaces

outer_lateral_nodes = mesh_handling_tools.find_nodesOnSurface(
mesh_data_class, "lateral external", return_coordinates=True)

inner_lateral_nodes = mesh_handling_tools.find_nodesOnSurface(
mesh_data_class, "lateral internal", return_coordinates=True)

# Transforms the nodes on the lateral surfaces to cylindrical coordina-
# tes, and separates the radius from the angle and the z value

radius_outer = np.zeros(len(outer_lateral_nodes))

theta_z_outer = np.zeros((len(outer_lateral_nodes), 2))

radius_inner = np.zeros(len(inner_lateral_nodes))

theta_z_inner = np.zeros((len(inner_lateral_nodes), 2))

for i in range(len(outer_lateral_nodes)):

    x, y, z = outer_lateral_nodes[i]

    # Gets the cylindrical coordinates

    theta = 0.5*np.pi 

    if abs(x)>1E-6:

        theta = np.arctan(y/x)

    elif y<0:

        theta = 1.5*np.pi 

    r = np.sqrt((x*x)+(y*y))

    # Updates the arrays of cylindrical coordinates

    radius_outer[i] = r 

    theta_z_outer[i,0] = theta 

    theta_z_outer[i,1] = z

for i in range(len(inner_lateral_nodes)):

    x, y, z = inner_lateral_nodes[i]

    # Gets the cylindrical coordinates

    theta = 0.5*np.pi 

    if abs(x)>1E-6:

        theta = np.arctan(y/x)

    elif y<0:

        theta = 1.5*np.pi 

    r = np.sqrt((x*x)+(y*y))

    # Updates the arrays of cylindrical coordinates

    radius_inner[i] = r 

    theta_z_inner[i,0] = theta 

    theta_z_inner[i,1] = z

# Interpolates the radius as a function of theta and z using radial ba-
# sis functions

radius_outer_interpolation = RBFInterpolator(theta_z_outer, radius_outer)

radius_inner_interpolation = RBFInterpolator(theta_z_inner, radius_inner)

# Creates the functions spaces

W = FunctionSpace(mesh_data_class.mesh, "CG", 1)

V = VectorFunctionSpace(mesh_data_class.mesh, "CG", 1)

# Defines a material property function in cylindrical coordinates

k_sup = [2.0, 5.0]

k_inf = [2.0, 5.0]

def k_material(x_vector):

    # Gets the coordinates

    x, y, z = x_vector

    # Gets the cylindrical coordinates

    theta = 0.5*np.pi 

    if abs(x)>1E-6:

        theta = np.arctan(y/x)

    elif y<0:

        theta = 1.5*np.pi 

    r = np.sqrt((x*x)+(y*y))

    # Evaluates the limits of the k parameter in the current angle

    k_s = (k_sup[0]*0.5*(np.cos(theta)+1.0))+(k_sup[1]*0.5*(1.0+np.cos(
    theta+np.pi)))

    k_i = (k_inf[0]*0.5*(1.0+np.cos(theta)))+(k_inf[1]*0.5*(1.0+np.cos(
    theta+np.pi)))

    # Evalutes the maximum and minimum radii

    maximum_radius = radius_outer_interpolation([[theta, z]])[0]

    minimum_radius = radius_inner_interpolation([[theta, z]])[0]

    # Verifies the limit radii

    if r>maximum_radius+1E-3:

        return 0.0
    
    elif r<minimum_radius-1E-3:

        return 0.0

    # Interpolates linearly across the radial direction

    k_value = ((k_s*((r-minimum_radius)/(maximum_radius-minimum_radius))
    )+(k_i*((maximum_radius-r)/(maximum_radius-minimum_radius))))

    return k_s

u_interpolation = interpolate_scalar_function(k_material, W, name="k p"+
"roperty")

file = XDMFFile(get_parent_path_of_file()+"//material_property.xdmf")

file.write(u_interpolation)