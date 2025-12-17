# Routine to experiment with incompressible hyperelasticity

from dolfin import *

import ufl_legacy as ufl

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ....MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from ....MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity

# Geometry information

L, H, W = 1.0, 0.2, 0.3

# Creates a box mesh using gmsh

mesh_data_class = read_mshMesh({"length x": W, "length y": H, "length "+
"z": L, "number of divisions in x": 5, "number of divisions in y": 5, 
"number of divisions in z": 25, "verbose": False, "mesh file name": "b"+
"ox_mesh", "mesh file directory": get_parent_path_of_file()})

# Neumann boundary conditions

traction_vectors = {"top": Constant([0.0, 0.0, 5E5])}

# Evaluates the geometry volume

inv_V0 = 1/assemble(1.0*mesh_data_class.dx)

# Sets the constitutive model

E = 1E6

poisson = 0.3

constitutive_model = isotropic_hyperelasticity.Neo_Hookean({"E": E, "nu":
poisson})

# Sets the function space

mixed_element = MixedElement([VectorElement("Lagrange", 
mesh_data_class.mesh.ufl_cell(), 2), FiniteElement("Lagrange", 
mesh_data_class.mesh.ufl_cell(), 1)], )

monolithic_functionSpace = FunctionSpace(mesh_data_class.mesh, 
mixed_element)

trial_functions = TrialFunction(monolithic_functionSpace)

monolithic_solution = Function(monolithic_functionSpace)

u, lmbda = split(monolithic_solution)

delta_u, delta_lambda = split(TestFunction(monolithic_functionSpace))

# Dirichlet boundary conditions

bc = [DirichletBC(monolithic_functionSpace.sub(0), Constant((0.0, 0.0, 
0.0)), mesh_data_class.boundary_meshFunction, 
mesh_data_class.boundary_physicalGroupsNameToTag["bottom"])]

# Variational form

external_work = 0.0

I = Identity(3)

F = grad(u)+I

F_invT = inv(F).T

J = det(F)

internal_work = ((inner(constitutive_model.first_piolaStress(u)+(lmbda*
inv_V0*J*F_invT), grad(delta_u))*dx)+(inv_V0*(((J-1)*delta_lambda)*
mesh_data_class.dx)))

for physical_group, traction in traction_vectors.items():

    external_work += dot(traction, delta_u)*mesh_data_class.ds(
    mesh_data_class.boundary_physicalGroupsNameToTag[physical_group])

# Solver

residual_form = internal_work-external_work

residual_derivative = derivative(residual_form , monolithic_solution, 
trial_functions)

Res = NonlinearVariationalProblem(residual_form, monolithic_solution, 
bc, J=residual_derivative)

solver = NonlinearVariationalSolver(Res)

solver.solve()

# Solution saving

u_solution, lambda_solution = monolithic_solution.split()

J = det(grad(u_solution)+I)

V_new = assemble(J*mesh_data_class.dx)

print("The ratio of the new volume by the volume of the reference conf"+
"iguration is "+str(inv_V0*V_new)+"\n")

u_solution.rename("Displacement", "DNS")

lambda_solution.rename("Pressure", "DNS")

file = XDMFFile(get_parent_path_of_file()+"//tests//displacement.xdmf")

file.write(u_solution)

file = XDMFFile(get_parent_path_of_file()+"//tests//pressure.xdmf")

file.write(lambda_solution)